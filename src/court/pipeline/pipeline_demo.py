# pipeline_demo_court.py

import argparse
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import logging
import threading
import queue
import csv
import time
from typing import List, Dict, Any

# コート検出用モジュールをインポート
# pipeline_modules.py が同じディレクトリか、Pythonのパスが通った場所にあることを想定
try:
    from pipeline_modules import CourtPreprocessor, CourtDetector, CourtPostprocessor
except ImportError:
    print("Error: pipeline_modules.py not found.")
    print("Please make sure it is in the same directory or in the Python path.")
    exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# キーポイント描画用の色のリスト (OpenCV BGR形式)
# 15個のキーポイントに対応
KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (75, 25, 230)
]

def draw_keypoints_on_frame(frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """フレームに検出されたキーポイントを描画するヘルパー関数"""
    for i in range(len(keypoints)):
        if scores[i] > threshold:
            px, py = int(keypoints[i][0]), int(keypoints[i][1])
            color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2.circle(frame, (px, py), 5, color, -1)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)
            # cv2.putText(frame, f"{i}", (px + 8, py - 8),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

class MultithreadedCourtDetector:
    """
    マルチスレッドパイプラインでテニスコートのキーポイントを検出するクラス。
    I/O, 前処理, 推論, 後処理を独立したスレッドで実行し、スループットを最大化する。
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

        # --- 1. 初期化 ---
        self._initialize_device()
        self._initialize_pipeline_modules()

        self.video_writer = None
        self.video_properties: Dict[str, Any] = {}
        
        # ワーカースレッド間のデータ受け渡し用キュー
        self.preprocess_queue: queue.Queue = queue.Queue(maxsize=32)
        self.inference_queue: queue.Queue = queue.Queue(maxsize=32)
        
        self.all_keypoint_results: List[Dict[str, np.ndarray]] = []
        self.is_running = threading.Event()
        
        # パフォーマンス計測用
        self.timings: Dict[str, List[float]] = {
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
        log.info("Initializing pipeline modules...")
        self.preprocessor = CourtPreprocessor() # 引数はデフォルト値を使用
        self.detector = CourtDetector(self.args.checkpoint_path, self.device)
        self.postprocessor = CourtPostprocessor() # 引数はデフォルト値を使用
        log.info("Pipeline modules initialized.")

    def _initialize_video_io(self):
        """入力動画のプロパティを読み込み、出力用のVideoWriterを準備する"""
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.args.video}")

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
        frames_batch: List[np.ndarray] = []
        
        total_frames = self.video_properties['total_frames']
        for frame_idx in range(total_frames):
            if not self.is_running.is_set():
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_batch.append(frame_rgb)

            is_last_frame = (frame_idx == total_frames - 1)
            if len(frames_batch) == self.args.batch_size or (is_last_frame and frames_batch):
                start_time = time.perf_counter()
                
                # オリジナルフレームはBGR形式で描画用に渡す
                original_frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_batch]
                
                batch_tensor, batch_meta = self.preprocessor.process_batch(frames_batch)
                
                self.preprocess_queue.put((batch_tensor, batch_meta, original_frames_bgr))

                self.timings['io_preprocess'].append(time.perf_counter() - start_time)
                
                frames_batch.clear()
        
        self.preprocess_queue.put(None) # 終了シグナル
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
            heatmap_preds = self.detector.predict(batch_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  # GPU処理の完了を待つ
            
            self.timings['inference'].append(time.perf_counter() - start_time)

            self.inference_queue.put((heatmap_preds, batch_meta, original_frames_batch))
        
        log.info("Worker 2 (Inference) finished.")

    def _worker_postprocess_write(self):
        """ワーカー3: 後処理、キーポイント描画、動画書き込み (CPUバウンド)"""
        log.info("Worker 3 (Postprocess & Write) started.")
        total_frames = self.video_properties['total_frames']
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while self.is_running.is_set():
            try:
                data = self.inference_queue.get(timeout=1)
            except queue.Empty:
                continue

            if data is None: # 終了シグナル
                break
                
            heatmap_preds, batch_meta, original_frames_batch = data

            start_time = time.perf_counter()
            
            batch_results = self.postprocessor.process_batch(heatmap_preds, batch_meta)
            
            for i, result in enumerate(batch_results):
                self.all_keypoint_results.append(result)
                
                output_frame = draw_keypoints_on_frame(
                    original_frames_batch[i],
                    result['keypoints'],
                    result['scores'],
                    self.args.score_threshold
                )
                self.video_writer.write(output_frame)
                pbar.update(1)

            self.timings['postprocess_write'].append(time.perf_counter() - start_time)
        
        pbar.close()
        log.info("Worker 3 (Postprocess & Write) finished.")

    def _save_results_as_csv(self):
        """キーポイント検出結果をCSVファイルに保存する"""
        log.info(f"Saving keypoint results to {self.args.results_csv}...")
        try:
            with open(self.args.results_csv, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'keypoint_id', 'score', 'x', 'y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for frame_idx, result in enumerate(self.all_keypoint_results):
                    keypoints = result['keypoints']
                    scores = result['scores']
                    for kp_idx in range(len(keypoints)):
                        row = {
                            'frame': frame_idx,
                            'keypoint_id': kp_idx,
                            'score': scores[kp_idx],
                            'x': keypoints[kp_idx][0],
                            'y': keypoints[kp_idx][1]
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
        
        self.is_running.clear()
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
    parser = argparse.ArgumentParser(description="Multithreaded Court Keypoint Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the trained model checkpoint (.ckpt)")
    parser.add_argument("--output", default="demo_output_court.mp4", help="Output video file with keypoints")
    parser.add_argument("--results_csv", default="court_keypoints.csv", help="Output CSV file for keypoint data")
    parser.add_argument("--device", default="auto", choices=["cuda", "cpu", "auto"], help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Minimum score to visualize a keypoint")
    args = parser.parse_args()

    try:
        detector_pipeline = MultithreadedCourtDetector(args)
        detector_pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()