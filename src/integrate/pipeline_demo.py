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
from collections import deque
import csv
import time
from typing import List, Dict, Any, Tuple

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
log = logging.getLogger(__name__)

# --- 各タスクのモジュールをインポート ---
# (パスは実行環境に合わせて調整してください)

# コート検出モジュール
from src.court.pipeline.pipeline_modules import CourtPreprocessor, CourtDetector, CourtPostprocessor
from src.court.pipeline.drawing_utils import draw_keypoints_on_frame as draw_court_keypoints
from src.court.pipeline.drawing_utils import draw_court_skeleton

# 選手・姿勢推定モジュール
from src.pose.pipeline.pipeline_module import (
    PlayerPreprocessor, PlayerDetector, PlayerPostprocessor,
    PosePreprocessor, PoseEstimator, PosePostprocessor
)
from src.pose.pipeline.drawing_utils import draw_results_on_frame as draw_pose_results

# ボール追跡モジュール
from src.ball.pipeline.wasb_modules import load_default_config, build_tracker
from src.ball.pipeline.wasb_modules.pipeline_modules import BallPreprocessor, BallDetector, DetectionPostprocessor as BallPostprocessor
from src.ball.pipeline.wasb_modules.drawing_utils import draw_on_frame as draw_ball_tracking

class IntegratedTennisAnalysisPipeline:
    """
    コート検出、選手姿勢推定、ボール追跡を統合したマルチスレッド解析パイプライン。
    3つのワーカースレッド（I/O & Preprocess, Unified Inference, Postprocess & Write）で処理を分担する。
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.is_running = threading.Event()
        self._initialize_device()

        # スレッド間通信用のキュー
        queue_size = cfg.threading.queue_size_multiplier * cfg.batch_size
        self.inference_queue = queue.Queue(maxsize=queue_size)
        self.postprocess_queue = queue.Queue(maxsize=queue_size)

        # パイプラインモジュールの初期化
        self._initialize_pipeline_modules()

        # I/O関連
        self.video_writer = None
        self.video_properties: Dict[str, Any] = {}
        
        # 結果保存用
        self.all_results_data: List[Dict] = []
        self.timings: Dict[str, List[float]] = {k: [] for k in ['io_preprocess', 'unified_inference', 'postprocess_write']}

    def _initialize_device(self):
        """デバイスを初期化"""
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        log.info(f"Using device: {self.device}")

    def _initialize_pipeline_modules(self):
        """各タスクの処理モジュールを初期化"""
        log.info("Initializing pipeline modules...")
        
        # --- コート検出モジュール ---
        if self.cfg.tasks.court:
            court_input_size = tuple(self.cfg.court.input_size)
            self.court_preprocessor = CourtPreprocessor(input_size=court_input_size)
            self.court_detector = CourtDetector(self.cfg.court.checkpoint, self.device)
            self.court_postprocessor = CourtPostprocessor(multi_channel=self.cfg.court.multi_channel)

        # --- 選手・姿勢推定モジュール ---
        if self.cfg.tasks.pose:
            self.player_preprocessor = PlayerPreprocessor()
            self.player_detector = PlayerDetector(self.cfg.player.checkpoint, self.device)
            self.player_postprocessor = PlayerPostprocessor(self.cfg.player.threshold)
            self.pose_preprocessor = PosePreprocessor()
            self.pose_estimator = PoseEstimator(self.device, model_name=self.cfg.pose.model_name)
            self.pose_postprocessor = PosePostprocessor()
            
        # --- ボール追跡モジュール ---
        if self.cfg.tasks.ball:
            self.ball_cfg = load_default_config()
            if self.cfg.ball.model_path:
                self.ball_cfg.detector.model_path = self.cfg.ball.model_path
            self.ball_preprocessor = BallPreprocessor(self.ball_cfg)
            self.ball_detector = BallDetector(self.ball_cfg, self.device)
            self.ball_postprocessor = BallPostprocessor(self.ball_cfg)
            self.ball_tracker = build_tracker(self.ball_cfg)
            
        log.info("Pipeline modules initialized.")

    def _initialize_video_io(self):
        """ビデオI/Oを初期化"""
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
        log.info(f"Video properties: {self.video_properties}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.cfg.io.output_video, fourcc, self.video_properties['fps'],
                                            (self.video_properties['width'], self.video_properties['height']))

    def run(self):
        """パイプライン全体を実行"""
        self._initialize_video_io()
        self.is_running.set()

        threads = [
            threading.Thread(target=self._worker_io_preprocess, name="Worker-IO-Preprocess"),
            threading.Thread(target=self._worker_unified_inference, name="Worker-Inference"),
            threading.Thread(target=self._worker_postprocess_write, name="Worker-Postprocess-Write")
        ]
        
        for t in threads: t.start()
        for t in threads: t.join()

        log.info("All workers have finished.")
        self._save_results_as_csv()
        self._report_timings()
        
        if self.video_writer: self.video_writer.release()
        log.info(f"Output video saved to: {self.cfg.io.output_video}")
        log.info(f"Analysis results saved to: {self.cfg.io.output_csv}")

    def _worker_io_preprocess(self):
        """Worker 1: ビデオ読み込みと全タスクの前処理"""
        log.info("Started.")
        cap = cv2.VideoCapture(self.cfg.io.video)
        total_frames = self.video_properties['total_frames']
        
        # ボール追跡用のフレームシーケンス管理
        ball_frame_history = deque(maxlen=self.ball_preprocessor.frames_in if self.cfg.tasks.ball else 1)
        
        frames_batch, ball_frames_batch, frame_indices_batch = [], [], []

        for frame_idx in range(total_frames):
            if not self.is_running.is_set(): break
            ret, frame = cap.read()
            if not ret: break
            
            frames_batch.append(frame)
            frame_indices_batch.append(frame_idx)

            if self.cfg.tasks.ball:
                ball_frame_history.append(frame)
                if len(ball_frame_history) == self.ball_preprocessor.frames_in:
                    ball_frames_batch.append(list(ball_frame_history))


            is_last_frame_in_video = (frame_idx == total_frames - 1)
            if len(frames_batch) == self.cfg.batch_size or (is_last_frame_in_video and frames_batch):
                start_time = time.perf_counter()
                
                # RGBに変換したフレームを各プリプロセッサに渡す
                frames_batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_batch]
                
                # 各タスクの前処理を実行
                data_for_inference = {'original_frames': frames_batch.copy(), 'frame_indices': frame_indices_batch.copy()}
                
                if self.cfg.tasks.court:
                    data_for_inference['court_input'], data_for_inference['court_meta'] = self.court_preprocessor.process_batch(frames_batch_rgb)

                if self.cfg.tasks.pose:
                    data_for_inference['player_input'], data_for_inference['player_meta'] = self.player_preprocessor.process_batch(frames_batch)

                if self.cfg.tasks.ball:
                    # ボールはシーケンス入力のため、バッチごとに処理
                    ball_sequences = []
                    # 最後のフレームを基準に過去のフレームを取得してシーケンスを作成
                    if len(ball_frame_history) >= self.ball_preprocessor.frames_in:
                         ball_sequences = [list(ball_frame_history)] * len(frames_batch) # 簡易的なバッチ化
                         data_for_inference['ball_input'], data_for_inference['ball_meta'] = self.ball_preprocessor.process_batch(ball_sequences)

                self.inference_queue.put(data_for_inference)
                self.timings['io_preprocess'].append(time.perf_counter() - start_time)
                
                frames_batch.clear()
                ball_frames_batch.clear()
                frame_indices_batch.clear()
        
        self.inference_queue.put(None) # 終了シグナル
        cap.release()
        log.info("Finished.")

    def _worker_unified_inference(self):
        """Worker 2: 全モデルの推論をシーケンシャルに実行"""
        log.info("Started.")
        while self.is_running.is_set():
            data = self.inference_queue.get()
            if data is None:
                self.postprocess_queue.put(None)
                break

            start_time = time.perf_counter()
            
            # --- 推論実行 ---
            # 1. コート検出
            if self.cfg.tasks.court:
                data['court_preds'] = self.court_detector.predict(data['court_input'])

            # 2. ボール検出
            if self.cfg.tasks.ball and 'ball_input' in data:
                data['ball_preds'] = self.ball_detector.predict_batch(data['ball_input'])

            # 3. 選手検出 -> 姿勢推定
            if self.cfg.tasks.pose:
                player_outputs = self.player_detector.predict(data['player_input'])
                player_detections_batch = self.player_postprocessor.process_batch(player_outputs, data['player_meta'])
                
                pose_inputs_batch, pose_meta_batch = [], []
                for i, (frame, detections) in enumerate(zip(data['original_frames'], player_detections_batch)):
                    if len(detections['boxes']) > 0:
                        pose_inputs, pose_meta = self.pose_preprocessor.process_frame(frame, detections)
                        if pose_inputs is not None:
                           pose_outputs = self.pose_estimator.predict(pose_inputs)
                           data.setdefault('pose_preds', {})[i] = pose_outputs
                           data.setdefault('pose_meta', {})[i] = pose_meta
                           data.setdefault('player_detections', {})[i] = detections

            if self.device.type == 'cuda': torch.cuda.synchronize()
            self.timings['unified_inference'].append(time.perf_counter() - start_time)
            
            self.postprocess_queue.put(data)
        
        log.info("Finished.")

    def _worker_postprocess_write(self):
        """Worker 3: 後処理、トラッキング、描画、書き込み"""
        log.info("Started.")
        pbar = tqdm(total=self.video_properties['total_frames'], desc="Processing Video")
        
        # ボールトラッカーの初期化
        if self.cfg.tasks.ball:
            self.ball_tracker.refresh()
            if hasattr(self.ball_preprocessor, 'frames_in') and self.ball_preprocessor.frames_in > 1:
                for _ in range(self.ball_preprocessor.frames_in - 1):
                    self.ball_tracker.update([])
        
        while self.is_running.is_set():
            data = self.postprocess_queue.get()
            if data is None: break

            start_time = time.perf_counter()
            original_frames = data['original_frames']

            # --- 後処理 ---
            # 1. コート
            court_results_batch = self.court_postprocessor.process_batch(data['court_preds'], data['court_meta']) if self.cfg.tasks.court and 'court_preds' in data else [None] * len(original_frames)
            
            # 2. ボール (最終修正)
            if self.cfg.tasks.ball and 'ball_preds' in data:
                # ball_detectorが出力した辞書を、CPUに移動させずにそのまま後処理モジュールに渡す
                # 後処理モジュールが内部でGPUデータと辞書形式を正しく処理する
                predictions_dict_gpu = data['ball_preds']
                ball_detections_batch = self.ball_postprocessor.process_batch(predictions_dict_gpu, data['ball_meta'], self.device)
            else:
                ball_detections_batch = [[]] * len(original_frames)

            # 3. 姿勢
            pose_results_batch = [self.pose_postprocessor.process_frame(data['pose_preds'][i], data['pose_meta'][i]) if self.cfg.tasks.pose and i in data.get('pose_preds', {}) else [] for i in range(len(original_frames))]

            # --- フレームごとの処理ループ ---
            for i, frame in enumerate(original_frames):
                frame_idx = data['frame_indices'][i]
                output_frame = frame.copy()
                frame_results = {"frame_idx": frame_idx}

                # コート描画
                if self.cfg.tasks.court and court_results_batch[i] and self.cfg.visualization.enabled:
                    court_res = court_results_batch[i]
                    frame_results["court"] = court_res
                    draw_court_keypoints(output_frame, court_res['keypoints'], court_res['scores'], self.cfg.court.score_threshold)
                    draw_court_skeleton(output_frame, court_res['keypoints'], court_res['scores'], self.cfg.court.score_threshold)

                # 選手・姿勢描画
                if self.cfg.tasks.pose:
                    player_dets = data.get('player_detections', {}).get(i, {'scores': [], 'boxes': []})
                    pose_res = pose_results_batch[i]
                    frame_results["players"] = player_dets
                    frame_results["poses"] = pose_res
                    if self.cfg.visualization.enabled:
                        draw_pose_results(output_frame, player_dets, pose_res, self.cfg)

                # ボール追跡と描画
                if self.cfg.tasks.ball:
                    ball_tracking_output = self.ball_tracker.update(ball_detections_batch[i])
                    frame_results["ball"] = ball_tracking_output
                    if self.cfg.visualization.enabled:
                        draw_ball_tracking(output_frame, ball_tracking_output)

                self.video_writer.write(output_frame)
                self.all_results_data.append(frame_results)
                pbar.update(1)

            self.timings['postprocess_write'].append(time.perf_counter() - start_time)

        pbar.close()
        log.info("Finished.")

    def _save_results_as_csv(self):
        """全解析結果を1つのCSVファイルに保存"""
        log.info("Saving combined results to CSV...")
        if not self.all_results_data: return

        # ヘッダーを動的に作成
        header = ['frame_idx', 'ball_visible', 'ball_x', 'ball_y', 'ball_score']
        # コートキーポイントのヘッダー
        for i in range(14): # コートは14点
            header.extend([f'court_kp_{i}_x', f'court_kp_{i}_y', f'court_kp_{i}_score'])
        # 選手のヘッダー (最大N人分)
        max_players = max(len(d.get('players', {}).get('boxes', [])) for d in self.all_results_data) if self.all_results_data else 0
        for p_id in range(max_players):
            header.extend([f'p{p_id}_score', f'p{p_id}_x1', f'p{p_id}_y1', f'p{p_id}_x2', f'p{p_id}_y2'])
            for kp_id in range(17): # COCO 17点
                header.extend([f'p{p_id}_kp{kp_id}_x', f'p{p_id}_kp{kp_id}_y', f'p{p_id}_kp{kp_id}_s'])

        with open(self.cfg.io.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            writer.writeheader()

            sorted_results = sorted(self.all_results_data, key=lambda x: x['frame_idx'])
            for data in sorted_results:
                row = {'frame_idx': data['frame_idx']}
                # ボール
                ball_data = data.get('ball', {})
                row.update({
                    'ball_visible': 1 if ball_data.get('visi') else 0,
                    'ball_x': ball_data.get('x', -1),
                    'ball_y': ball_data.get('y', -1),
                    'ball_score': ball_data.get('score', 0.0),
                })
                # コート
                court_data = data.get('court')
                if court_data:
                    for i in range(14):
                        row[f'court_kp_{i}_x'] = court_data['keypoints'][i][0]
                        row[f'court_kp_{i}_y'] = court_data['keypoints'][i][1]
                        row[f'court_kp_{i}_score'] = court_data['scores'][i]
                # 選手と姿勢
                players = data.get('players', {})
                poses = data.get('poses', [])
                for p_id in range(len(players.get('boxes', []))):
                    row[f'p{p_id}_score'] = players['scores'][p_id]
                    row[f'p{p_id}_x1'], row[f'p{p_id}_y1'], row[f'p{p_id}_x2'], row[f'p{p_id}_y2'] = players['boxes'][p_id]
                    if p_id < len(poses):
                        pose = poses[p_id]
                        for kp_id in range(17):
                           row[f'p{p_id}_kp{kp_id}_x'] = pose['keypoints'][kp_id][0]
                           row[f'p{p_id}_kp{kp_id}_y'] = pose['keypoints'][kp_id][1]
                           row[f'p{p_id}_kp{kp_id}_s'] = pose['scores'][kp_id]
                writer.writerow(row)

    def _report_timings(self):
        """パフォーマンスレポートを出力"""
        log.info("--- Performance Report ---")
        for name, times in self.timings.items():
            if not times: continue
            total_time = sum(times)
            num_calls = len(times)
            avg_time = total_time / num_calls if num_calls > 0 else 0
            log.info(f"Stage '{name}': Total Calls: {num_calls}, Total Time: {total_time:.3f} s, Avg Time/Call: {avg_time:.3f} s")
        log.info("--------------------------")

@hydra.main(config_path="../../configs/infer/integrate", config_name="pipeline_demo", version_base=None)
def main(cfg: DictConfig) -> None:
    # Validate required config
    if cfg.io.video is None:
        raise ValueError("Video path is required. Please set io.video in config or via command line.")
    
    # Validate task-specific requirements
    if cfg.tasks.court and cfg.court.checkpoint is None:
        raise ValueError("Court checkpoint is required when court task is enabled. Please set court.checkpoint.")
    if cfg.tasks.pose and cfg.player.checkpoint is None:
        raise ValueError("Player checkpoint is required when pose task is enabled. Please set player.checkpoint.")
    
    # Set up logging from config
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format=cfg.logging.format
    )
    
    try:
        pipeline = IntegratedTennisAnalysisPipeline(cfg)
        pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()