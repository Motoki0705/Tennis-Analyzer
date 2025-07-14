# pipeline_demo_pose.py

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

# パイプラインモジュールをインポート
from pipeline_module import (
    PlayerPreprocessor, PlayerDetector, PlayerPostprocessor,
    PosePreprocessor, PoseEstimator, PosePostprocessor
)
from drawing_utils import draw_results_on_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class MultithreadedPosePipeline:
    """マルチスレッドでプレイヤー検出と姿勢推定を実行するパイプライン"""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._initialize_device()
        self._initialize_pipeline_modules()

        self.video_writer = None
        self.video_properties: Dict[str, Any] = {}
        
        # スレッド間通信用のキュー
        queue_size = cfg.threading.queue_size_multiplier * cfg.batch_size
        self.preprocess_queue = queue.Queue(maxsize=queue_size)
        self.inference_queue = queue.Queue(maxsize=queue_size)
        
        self.all_results_for_csv: List[Dict] = []
        self.is_running = threading.Event()
        
        self.timings = {'io_preprocess': [], 'inference_pipeline': [], 'postprocess_write': []}

    def _initialize_device(self):
        if self.cfg.device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        log.info(f"Using device: {self.device}")

    def _initialize_pipeline_modules(self):
        log.info("Initializing pipeline modules...")
        # Stage 1: Player Detection
        self.player_preprocessor = PlayerPreprocessor()
        self.player_detector = PlayerDetector(self.cfg.player.checkpoint_path, self.device)
        self.player_postprocessor = PlayerPostprocessor(self.cfg.player.threshold)
        # Stage 2: Pose Estimation
        self.pose_preprocessor = PosePreprocessor()
        self.pose_estimator = PoseEstimator(self.device, model_name=self.cfg.pose.model_name)
        self.pose_postprocessor = PosePostprocessor()
        log.info("Pipeline modules initialized.")

    def _initialize_video_io(self):
        cap = cv2.VideoCapture(self.cfg.io.video)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {self.cfg.io.video}")
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
        """Worker 1: 動画読み込み & Player検出の前処理"""
        log.info("Worker 1 (I/O & Player Preprocess) started.")
        cap = cv2.VideoCapture(self.cfg.io.video)
        frames_batch, frame_indices = [], []
        total_frames = self.video_properties['total_frames']

        for frame_idx in range(total_frames):
            if not self.is_running.is_set(): break
            ret, frame = cap.read()
            if not ret: break
            frames_batch.append(frame)
            frame_indices.append(frame_idx)

            if len(frames_batch) == self.cfg.batch_size or (frame_idx == total_frames - 1 and frames_batch):
                start_time = time.perf_counter()
                player_inputs, player_meta = self.player_preprocessor.process_batch(frames_batch)
                self.preprocess_queue.put((player_inputs, player_meta, frames_batch, frame_indices))
                self.timings['io_preprocess'].append(time.perf_counter() - start_time)
                frames_batch, frame_indices = [], []
        
        self.preprocess_queue.put(None)
        cap.release()
        log.info("Worker 1 (I/O & Player Preprocess) finished.")

    def _worker_inference_pipeline(self):
        """Worker 2: Player推論 -> Player後処理 -> Pose前処理 -> Pose推論"""
        log.info("Worker 2 (Inference Pipeline) started.")
        while self.is_running.is_set():
            try: data = self.preprocess_queue.get(timeout=1)
            except queue.Empty: continue
            if data is None:
                self.inference_queue.put(None)
                break

            start_time = time.perf_counter()
            player_inputs, player_meta, frames_batch, frame_indices = data
            
            # --- Player Detection ---
            player_outputs = self.player_detector.predict(player_inputs)
            player_detections_batch = self.player_postprocessor.process_batch(player_outputs, player_meta)
            
            # --- Pose Estimation ---
            pose_results_batch = []
            for i, (frame, detections) in enumerate(zip(frames_batch, player_detections_batch)):
                if len(detections['boxes']) > 0:
                    pose_inputs, pose_meta = self.pose_preprocessor.process_frame(frame, detections)
                    if pose_inputs:
                        pose_outputs = self.pose_estimator.predict(pose_inputs)
                        pose_results = self.pose_postprocessor.process_frame(pose_outputs, pose_meta)
                        pose_results_batch.append(pose_results)
                    else:
                        pose_results_batch.append([]) # No detections
                else:
                    pose_results_batch.append([]) # No detections

            if self.device.type == 'cuda': torch.cuda.synchronize()
            self.timings['inference_pipeline'].append(time.perf_counter() - start_time)

            self.inference_queue.put((player_detections_batch, pose_results_batch, frames_batch, frame_indices))
        
        log.info("Worker 2 (Inference Pipeline) finished.")
    
    def _worker_postprocess_write(self):
        """Worker 3: 描画 & 動画/CSV書き込み"""
        log.info("Worker 3 (Postprocess & Write) started.")
        pbar = tqdm(total=self.video_properties['total_frames'], desc="Processing frames")
        while self.is_running.is_set():
            try: data = self.inference_queue.get(timeout=1)
            except queue.Empty: continue
            if data is None: break

            start_time = time.perf_counter()
            player_detections_batch, pose_results_batch, frames_batch, frame_indices = data

            for i, frame in enumerate(frames_batch):
                player_detections = player_detections_batch[i]
                pose_results = pose_results_batch[i]
                
                # 描画
                if self.cfg.visualization.enabled:
                    output_frame = draw_results_on_frame(frame, player_detections, pose_results, self.cfg.pose.keypoint_threshold)
                else:
                    output_frame = frame
                self.video_writer.write(output_frame)
                
                # CSV用データ保存
                self.all_results_for_csv.append({
                    "frame_idx": frame_indices[i],
                    "player_detections": player_detections,
                    "pose_results": pose_results
                })
                pbar.update(1)

            self.timings['postprocess_write'].append(time.perf_counter() - start_time)
        
        pbar.close()
        log.info("Worker 3 (Postprocess & Write) finished.")

    def _save_results_as_csv(self):
        """検出結果と姿勢推定結果をCSVファイルに保存する"""
        log.info(f"Saving results to {self.cfg.io.results_csv}...")
        
        # ヘッダー作成 (keypoint 0-16)
        header = ['frame', 'person_id', 'bbox_score', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max']
        for i in range(17): # 17 keypoints for COCO
            header.extend([f'kp_{i}_x', f'kp_{i}_y', f'kp_{i}_score'])
        
        try:
            with open(self.cfg.io.results_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                
                # フレームインデックスでソート
                sorted_results = sorted(self.all_results_for_csv, key=lambda x: x['frame_idx'])
                
                for data in sorted_results:
                    frame_idx = data['frame_idx']
                    # 1フレームに複数の人がいる場合
                    for person_id, (bbox_score, bbox) in enumerate(zip(data['player_detections']['scores'], data['player_detections']['boxes'])):
                        row = {
                            'frame': frame_idx,
                            'person_id': person_id,
                            'bbox_score': bbox_score,
                            'bbox_x_min': bbox[0], 'bbox_y_min': bbox[1],
                            'bbox_x_max': bbox[2], 'bbox_y_max': bbox[3],
                        }
                        if person_id < len(data['pose_results']):
                            pose = data['pose_results'][person_id]
                            for kp_idx in range(17):
                                row[f'kp_{kp_idx}_x'] = pose['keypoints'][kp_idx][0]
                                row[f'kp_{kp_idx}_y'] = pose['keypoints'][kp_idx][1]
                                row[f'kp_{kp_idx}_score'] = pose['scores'][kp_idx]
                        writer.writerow(row)
        except IOError as e:
            log.error(f"Failed to write CSV file: {e}")

    def _report_timings(self):
        log.info("--- Performance Report ---")
        for name, times in self.timings.items():
            if not times: continue
            total_time, num_batches = sum(times), len(times)
            avg_time = total_time / num_batches
            log.info(f"Stage '{name}': Total Batches: {num_batches}, Total Time: {total_time:.3f} s, Avg Time/Batch: {avg_time:.3f} s")
        log.info("--------------------------")
    
    def run(self):
        self._initialize_video_io()
        self.is_running.set()

        thread1 = threading.Thread(target=self._worker_io_preprocess)
        thread2 = threading.Thread(target=self._worker_inference_pipeline)
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

@hydra.main(config_path="../../../configs/infer/pose", config_name="pipeline_demo", version_base=None)
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
        pipeline = MultithreadedPosePipeline(cfg)
        pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()