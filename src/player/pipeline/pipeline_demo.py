# pipeline_demo_player.py

import hydra
from omegaconf import DictConfig
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

# プレイヤー検出用モジュールをインポート
from pipeline_module import PlayerPreprocessor, PlayerDetector, PlayerPostprocessor



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ラベルIDとラベル名のマッピング（例）
# これはモデルのconfigに合わせて調整する必要があります。
# LitRtdetrモデル内にid2labelがあればそれを使えます。
ID2LABEL = {0: 'player', 1: 'ball', 2: 'referee'} # 仮のラベルマップ

def draw_detections_on_frame(frame: np.ndarray, detections: Dict[str, np.ndarray]) -> np.ndarray:
    """フレームに検出されたバウンディングボックスを描画するヘルパー関数"""
    scores = detections['scores']
    labels = detections['labels']
    boxes = detections['boxes']

    for score, label_id, box in zip(scores, labels, boxes):
        box_int = [int(i) for i in box]
        x_min, y_min, x_max, y_max = box_int
        
        label_name = ID2LABEL.get(label_id, f"ID:{label_id}")
        
        # バウンディングボックスを描画 (緑色)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (36, 255, 12), 2)
        
        # ラベルとスコアを描画
        label_text = f"{label_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), (36, 255, 12), -1)
        cv2.putText(frame, label_text, (x_min, y_min - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame

class MultithreadedPlayerDetector:
    """
    マルチスレッドパイプラインでプレイヤーを検出するクラス。
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._initialize_device()
        self._initialize_pipeline_modules()

        self.video_writer = None
        self.video_properties: Dict[str, Any] = {}
        
        queue_size = cfg.threading.queue_size_multiplier * cfg.batch_size
        self.preprocess_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.inference_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        
        self.all_detection_results: List[Dict[str, np.ndarray]] = []
        self.is_running = threading.Event()
        
        self.timings: Dict[str, List[float]] = {
            'io_preprocess': [], 'inference': [], 'postprocess_write': []
        }

    def _initialize_device(self):
        """デバイスを初期化する"""
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        log.info(f"Using device: {self.device}")

    def _initialize_pipeline_modules(self):
        """各種処理モジュールを初期化する"""
        log.info("Initializing pipeline modules...")
        self.preprocessor = PlayerPreprocessor()
        self.detector = PlayerDetector(self.cfg.player.checkpoint_path, self.device)
        self.postprocessor = PlayerPostprocessor(confidence_threshold=self.cfg.player.score_threshold)
        log.info("Pipeline modules initialized.")
        # モデルのラベルマップを取得しようと試みる
        global ID2LABEL
        try:
            if hasattr(self.detector.model.model, 'config') and hasattr(self.detector.model.model.config, 'id2label'):
                ID2LABEL = self.detector.model.model.config.id2label
            elif hasattr(self.cfg, 'detection') and hasattr(self.cfg.detection, 'id2label'):
                ID2LABEL = self.cfg.detection.id2label
                log.info(f"Loaded label map from config: {ID2LABEL}")
            else:
                log.info(f"Using default label map: {ID2LABEL}")
        except Exception as e:
            log.warning(f"Could not load label map from model, using default. Error: {e}")


    def _initialize_video_io(self):
        """入力動画のプロパティを読み込み、出力用のVideoWriterを準備する"""
        cap = cv2.VideoCapture(self.cfg.io.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.cfg.io.video}")

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
        self.video_writer = cv2.VideoWriter(self.cfg.io.output, fourcc, self.video_properties['fps'],
                                            (self.video_properties['width'], self.video_properties['height']))

    def _worker_io_preprocess(self):
        """ワーカー1: 動画読み込みと前処理 (CPUバウンド)"""
        log.info("Worker 1 (I/O & Preprocess) started.")
        cap = cv2.VideoCapture(self.cfg.io.video)
        frames_batch: List[np.ndarray] = []
        
        total_frames = self.video_properties['total_frames']
        for frame_idx in range(total_frames):
            if not self.is_running.is_set(): break
            ret, frame = cap.read()
            if not ret: break
            
            frames_batch.append(frame)

            if len(frames_batch) == self.cfg.batch_size or (frame_idx == total_frames - 1 and frames_batch):
                start_time = time.perf_counter()
                
                original_frames_bgr = [f.copy() for f in frames_batch]
                batch_inputs, batch_meta = self.preprocessor.process_batch(frames_batch)
                
                self.preprocess_queue.put((batch_inputs, batch_meta, original_frames_bgr))
                self.timings['io_preprocess'].append(time.perf_counter() - start_time)
                frames_batch.clear()
        
        self.preprocess_queue.put(None)
        cap.release()
        log.info("Worker 1 (I/O & Preprocess) finished.")

    def _worker_inference(self):
        """ワーカー2: 推論 (GPUバウンド)"""
        log.info("Worker 2 (Inference) started.")
        while self.is_running.is_set():
            try: data = self.preprocess_queue.get(timeout=1)
            except queue.Empty: continue
            if data is None:
                self.inference_queue.put(None)
                break

            batch_inputs, batch_meta, original_frames_batch = data
            
            start_time = time.perf_counter()
            outputs = self.detector.predict(batch_inputs)
            if self.device.type == 'cuda': torch.cuda.synchronize()
            self.timings['inference'].append(time.perf_counter() - start_time)

            self.inference_queue.put((outputs, batch_meta, original_frames_batch))
        
        log.info("Worker 2 (Inference) finished.")

    def _worker_postprocess_write(self):
        """ワーカー3: 後処理、描画、動画書き込み (CPUバウンド)"""
        log.info("Worker 3 (Postprocess & Write) started.")
        total_frames = self.video_properties['total_frames']
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while self.is_running.is_set():
            try: data = self.inference_queue.get(timeout=1)
            except queue.Empty: continue
            if data is None: break
                
            outputs, batch_meta, original_frames_batch = data
            start_time = time.perf_counter()
            
            batch_results = self.postprocessor.process_batch(outputs, batch_meta)
            
            for i, result in enumerate(batch_results):
                self.all_detection_results.append(result)
                output_frame = original_frames_batch[i]
                if self.cfg.visualization.enabled:
                    output_frame = draw_detections_on_frame(output_frame, result)
                self.video_writer.write(output_frame)
                pbar.update(1)

            self.timings['postprocess_write'].append(time.perf_counter() - start_time)
        
        pbar.close()
        log.info("Worker 3 (Postprocess & Write) finished.")

    def _save_results_as_csv(self):
        """検出結果をCSVファイルに保存する"""
        log.info(f"Saving detection results to {self.cfg.io.results_csv}...")
        try:
            with open(self.cfg.io.results_csv, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'label_id', 'label_name', 'score', 'x_min', 'y_min', 'x_max', 'y_max']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for frame_idx, result_per_frame in enumerate(self.all_detection_results):
                    for i in range(len(result_per_frame['scores'])):
                        label_id = result_per_frame['labels'][i]
                        box = result_per_frame['boxes'][i]
                        row = {
                            'frame': frame_idx,
                            'label_id': label_id,
                            'label_name': ID2LABEL.get(label_id, 'unknown'),
                            'score': result_per_frame['scores'][i],
                            'x_min': box[0], 'y_min': box[1],
                            'x_max': box[2], 'y_max': box[3]
                        }
                        writer.writerow(row)
        except IOError as e:
            log.error(f"Failed to write CSV file: {e}")

    def _report_timings(self):
        """各ステージの処理時間を集計して表示する"""
        log.info("--- Performance Report ---")
        for name, times in self.timings.items():
            if not times: continue
            total_time, num_batches = sum(times), len(times)
            avg_time = total_time / num_batches
            log.info(f"Stage '{name}': Total Batches: {num_batches}, Total Time: {total_time:.3f} s, Avg Time/Batch: {avg_time:.3f} s")
        log.info("--------------------------")

    def run(self):
        """パイプライン全体を実行する"""
        self._initialize_video_io()
        self.is_running.set()

        thread1 = threading.Thread(target=self._worker_io_preprocess)
        thread2 = threading.Thread(target=self._worker_inference)
        thread3 = threading.Thread(target=self._worker_postprocess_write)
        
        thread1.start(); thread2.start(); thread3.start()
        thread1.join(); thread2.join()
        self.is_running.clear()
        thread3.join()

        log.info("All workers have finished.")
        self._save_results_as_csv()
        self._report_timings()

        if self.video_writer: self.video_writer.release()
        log.info(f"Output video saved to: {self.cfg.io.output}")

@hydra.main(config_path="../../../configs/infer/player", config_name="pipeline_demo", version_base=None)
def main(cfg: DictConfig) -> None:
    # Validate required config
    if cfg.io.video is None:
        raise ValueError("Video path is required. Please set io.video in config or via command line.")
    
    # Set up logging from config
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format=cfg.logging.format
    )
    
    try:
        detector_pipeline = MultithreadedPlayerDetector(cfg)
        detector_pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()