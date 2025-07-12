# demo_pipeline_multithread.py

import argparse
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import logging
import threading
import queue
from collections import deque
import csv
import time

# WASB-SBDT modules
from . import load_default_config
from .trackers import build_tracker
from .pipeline_modules import BallPreprocessor, BallDetector, DetectionPostprocessor
from .drawing_utils import draw_on_frame


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class MultithreadedTennisTracker:
    """
    マルチスレッドパイプラインでテニスボールを追跡するクラス。
    I/O, 前処理, 推論, 後処理を独立したスレッドで実行し、スループットを最大化する。
    """
    def __init__(self, args):
        self.args = args
        self.cfg = load_default_config()
        if args.model_path is not None:
            self.cfg.detector.model_path = args.model_path

        # --- 1. 初期化 ---
        self._initialize_device()
        self._initialize_pipeline_modules()

        self.video_writer = None
        self.video_properties = {}
        
        # ワーカースレッド間のデータ受け渡し用キュー
        # (キューのサイズは経験的に調整。大きすぎるとメモリ消費、小さすぎると待機が発生)
        self.preprocess_queue = queue.Queue(maxsize=32)
        self.inference_queue = queue.Queue(maxsize=32)
        
        self.all_tracking_results = []
        self.is_running = threading.Event()
        
        # パフォーマンス計測用
        self.timings = {
            'io_preprocess': [],
            'inference': [],
            'postprocess_write': []
        }

    def _initialize_device(self):
        """デバイスを初期化する"""
        if self.args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.args.device)
        log.info(f"Using device: {self.device}")

    def _initialize_pipeline_modules(self):
        """各種処理モジュールを初期化する"""
        self.preprocessor = BallPreprocessor(self.cfg)
        self.detector = BallDetector(self.cfg, self.device)
        self.postprocessor = DetectionPostprocessor(self.cfg)
        self.tracker = build_tracker(self.cfg)
        log.info("Pipeline modules initialized.")

    def _initialize_video_io(self):
        """入力動画のプロパティを読み込み、出力用のVideoWriterを準備する"""
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened():
            raise RuntimeError(f"動画が開けません: {self.args.video}")

        self.video_properties = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        
        log.info(f"Video: {self.video_properties['width']}x{self.video_properties['height']}, "
                 f"{self.video_properties['fps']:.2f}fps, {self.video_properties['total_frames']} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.args.output, fourcc, self.video_properties['fps'],
                                            (self.video_properties['width'], self.video_properties['height']))

    def _worker_io_preprocess(self):
        """ワーカー1: 動画読み込みと前処理 (CPUバウンド)"""
        log.info("Worker 1 (I/O & Preprocess) started.")
        cap = cv2.VideoCapture(self.args.video)
        frames_in = self.preprocessor.frames_in
        frame_history = deque(maxlen=frames_in)
        batch_sequences = []
        original_frames_batch = []
        
        for frame_idx in range(self.video_properties['total_frames']):
            if not self.is_running.is_set():
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_history.append(frame)
            if len(frame_history) < frames_in:
                continue

            batch_sequences.append(list(frame_history))
            original_frames_batch.append(frame) # バッチの最後のフレームを描画用に保存

            is_last_frame = (frame_idx == self.video_properties['total_frames'] - 1)
            if len(batch_sequences) == self.args.batch_size or (is_last_frame and batch_sequences):
                start_time = time.perf_counter()
                
                batch_tensor, batch_meta = self.preprocessor.process_batch(batch_sequences)
                
                # キューに前処理済みデータと描画用フレームを渡す
                self.preprocess_queue.put((batch_tensor, batch_meta, original_frames_batch.copy()))

                self.timings['io_preprocess'].append(time.perf_counter() - start_time)
                
                batch_sequences.clear()
                original_frames_batch.clear()
        
        # 終了シグナルを送信
        self.preprocess_queue.put(None)
        cap.release()
        log.info("Worker 1 (I/O & Preprocess) finished.")

    def _worker_inference(self):
        """ワーカー2: 推論 (GPUバウンド)"""
        log.info("Worker 2 (Inference) started.")
        while self.is_running.is_set():
            try:
                data = self.preprocess_queue.get(timeout=1)
            except queue.Empty:
                continue

            if data is None: # 終了シグナル
                self.inference_queue.put(None)
                break

            batch_tensor, batch_meta, original_frames_batch = data
            
            start_time = time.perf_counter()
            batch_preds = self.detector.predict_batch(batch_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # GPU処理の完了を待つ
            
            self.timings['inference'].append(time.perf_counter() - start_time)

            self.inference_queue.put((batch_preds, batch_meta, original_frames_batch))
        
        log.info("Worker 2 (Inference) finished.")

    def _worker_postprocess_write(self):
        """ワーカー3: 後処理、トラッキング、動画書き込み (CPUバウンド)"""
        log.info("Worker 3 (Postprocess & Write) started.")
        self.tracker.refresh()
        
        # 最初の `frames_in - 1` フレーム分のダミー結果を追加
        frames_in = self.preprocessor.frames_in
        for _ in range(frames_in - 1):
            self.all_tracking_results.append(self.tracker.update([]))

        total_frames = self.video_properties['total_frames']
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while self.is_running.is_set():
            try:
                data = self.inference_queue.get(timeout=1)
            except queue.Empty:
                continue

            if data is None: # 終了シグナル
                break
                
            batch_preds, batch_meta, original_frames_batch = data

            start_time = time.perf_counter()
            
            batch_detections = self.postprocessor.process_batch(batch_preds, batch_meta, self.device)
            
            for i, detections in enumerate(batch_detections):
                tracking_output = self.tracker.update(detections)
                self.all_tracking_results.append(tracking_output)
                output_frame = draw_on_frame(original_frames_batch[i], tracking_output)
                self.video_writer.write(output_frame)
                pbar.update(1)

            self.timings['postprocess_write'].append(time.perf_counter() - start_time)
        
        pbar.close()
        log.info("Worker 3 (Postprocess & Write) finished.")

    def _save_results_as_csv(self):
        """トラッキング結果をCSVファイルに保存する"""
        log.info(f"Saving tracking results to {self.args.results_csv}...")
        try:
            with open(self.args.results_csv, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'visible', 'score', 'x', 'y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for frame_idx, result in enumerate(self.all_tracking_results):
                    row = {
                        'frame': frame_idx,
                        'visible': 1 if result.get("visi", False) else 0,
                        'score': result.get("score", 0.0),
                        'x': result.get("x", -1),
                        'y': result.get("y", -1)
                    }
                    writer.writerow(row)
        except IOError as e:
            log.error(f"Failed to write CSV file: {e}")

    def _report_timings(self):
        """各ステージの処理時間を集計して表示する"""
        log.info("--- Performance Report ---")
        for name, times in self.timings.items():
            if not times: continue
            total_time = sum(times)
            avg_time = total_time / len(times)
            log.info(f"Stage '{name}':")
            log.info(f"  - Total Batches: {len(times)}")
            log.info(f"  - Total Time: {total_time:.3f} s")
            log.info(f"  - Average Time per Batch: {avg_time:.3f} s")
        log.info("--------------------------")

    def run(self):
        """パイプライン全体を実行する"""
        self._initialize_video_io()
        self.is_running.set()

        # ワーカーズレッドの作成と開始
        thread1 = threading.Thread(target=self._worker_io_preprocess)
        thread2 = threading.Thread(target=self._worker_inference)
        thread3 = threading.Thread(target=self._worker_postprocess_write)
        
        thread1.start()
        thread2.start()
        thread3.start()

        # すべてのスレッドが終了するのを待つ
        thread1.join()
        thread2.join()
        
        # ワーカーが終了したら、メインスレッドも終了
        thread3.join()

        log.info("All workers have finished.")

        # 後処理
        self._save_results_as_csv()
        self._report_timings()

        # リソース解放
        if self.video_writer:
            self.video_writer.release()
        log.info(f"Output video saved to: {self.args.output}")

def main():
    parser = argparse.ArgumentParser(description="WASB-SBDT Tennis Ball Tracking (Advanced Multithreaded)")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--output", default="demo_output_multithread.mp4", help="Output video file")
    parser.add_argument("--results_csv", default="tracking_results.csv", help="Output CSV file for tracking results")
    parser.add_argument("--model_path", default="/content/drive/MyDrive/ColabNotebooks/TennisAnalyzer/third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar", help="Path to a trained model (.pth.tar or .pth)")
    parser.add_argument("--device", default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing frame sequences")
    args = parser.parse_args()

    try:
        tracker_pipeline = MultithreadedTennisTracker(args)
        tracker_pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()