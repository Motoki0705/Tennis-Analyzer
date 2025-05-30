import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm


class MultiPredictor:
    def __init__(
        self,
        ball_predictor,
        court_predictor,
        pose_predictor,
        ball_interval: int = 1,
        court_interval: int = 30,
        pose_interval: int = 5,
        ball_batch_size: int = 1,  # Default to 1 if not specified, matching old behavior
        court_batch_size: int = 1,
        pose_batch_size: int = 1,
    ):
        # ロガーを直接初期化
        self.logger = logging.getLogger(self.__class__.__name__)

        self.ball_predictor = ball_predictor
        self.court_predictor = court_predictor
        self.pose_predictor = pose_predictor

        # 各タスク推論インターバル
        self.ball_interval = ball_interval
        self.court_interval = court_interval
        self.pose_interval = pose_interval

        # 各タスクバッチサイズ
        self.ball_batch_size = ball_batch_size
        self.court_batch_size = court_batch_size
        self.pose_batch_size = pose_batch_size

        # フレームバッファ
        self.ball_frame_buffer: List[np.ndarray] = (
            []
        )  # For ball predictor's sliding window
        self.court_batch_buffer: List[np.ndarray] = []
        self.pose_batch_buffer: List[np.ndarray] = []

        # オリジナルフレームのバッチ（オーバーレイ用）
        self.court_original_frames_batch: List[np.ndarray] = []
        self.pose_original_frames_batch: List[np.ndarray] = []

        # 直前結果キャッシュ
        self.last_ball_results: List[Dict] = (
            []
        )  # List of dicts, one per frame in the last ball batch
        self.last_court_results: List[List[Dict]] = (
            []
        )  # List of (List of keypoints per frame)
        self.last_pose_results: List[List[Dict]] = (
            []
        )  # List of (List of detections per frame)

        self.current_ball_result_idx = 0
        self.current_court_result_idx = 0
        self.current_pose_result_idx = 0

    def _process_ball_batch(self):
        """
        ボール検出バッチを処理します。
        ball_frame_buffer に十分なフレームが蓄積されたら、ball_predictor を使用して予測を実行します。
        """
        if not self.ball_frame_buffer:
            return
            
        num_frames = self.ball_predictor.num_frames
        
        if len(self.ball_frame_buffer) >= num_frames:
            clips = []
            
            for i in range(0, len(self.ball_frame_buffer) - num_frames + 1, self.ball_batch_size):
                if i + num_frames <= len(self.ball_frame_buffer):
                    clips.append(self.ball_frame_buffer[i:i+num_frames])
            
            if clips:
                self.last_ball_results = self.ball_predictor.predict(clips)
                self.ball_frame_buffer.clear()
                self.current_ball_result_idx = 0

    def _process_court_batch(self):
        if self.court_batch_buffer:
            self.last_court_results = self.court_predictor.predict(
                self.court_batch_buffer
            )[
                0
            ]  # predict returns (kps_list, hm_list)
            self.court_batch_buffer.clear()
            self.court_original_frames_batch.clear()
            self.current_court_result_idx = 0

    def _process_pose_batch(self):
        if self.pose_batch_buffer:
            self.last_pose_results = self.pose_predictor.predict(self.pose_batch_buffer)
            self.pose_batch_buffer.clear()
            self.pose_original_frames_batch.clear()
            self.current_pose_result_idx = 0

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error(f"動画を開けませんでした: {input_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"読み込み完了 → フレーム数: {total}, FPS: {fps:.2f}, 解像度: {width}×{height}"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Buffers for batching predictions
        ball_sliding_window: List[np.ndarray] = []  # For T-frame input to ball model

        # Cache for last known predictions to use between intervals
        # These store results for a *single* frame or sequence
        cached_ball_pred: Optional[Dict] = None
        cached_court_pred: Optional[List[Dict]] = None
        cached_pose_pred: Optional[List[Dict]] = None

        # Store BATCHED predictions and original frames
        ball_clips_for_batch: List[List[np.ndarray]] = []
        court_frames_for_batch: List[np.ndarray] = []
        pose_frames_for_batch: List[np.ndarray] = []

        # Store BATCHED prediction results
        # These will hold lists of predictions, one for each item in the processed batch
        ball_batch_predictions: List[Dict] = []
        court_batch_predictions: List[List[Dict]] = (
            []
        )  # List of (kps_list for each frame)
        pose_batch_predictions: List[List[Dict]] = (
            []
        )  # List of (detections_list for each frame)

        # Index to iterate through batched predictions
        ball_pred_idx = 0
        court_pred_idx = 0
        pose_pred_idx = 0

        frame_idx = 0
        with tqdm(total=total, desc="Multi 推論処理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- Ball Prediction ---
                ball_sliding_window.append(frame.copy())  # Use copy for safety
                if len(ball_sliding_window) > self.ball_predictor.num_frames:
                    ball_sliding_window.pop(0)

                if (
                    frame_idx % self.ball_interval == 0
                    and len(ball_sliding_window) >= self.ball_predictor.num_frames
                ):
                    ball_clips_for_batch.append(
                        list(ball_sliding_window)
                    )  # Add a copy of the current window

                if len(ball_clips_for_batch) >= self.ball_batch_size:
                    ball_batch_predictions = self.ball_predictor.predict(
                        ball_clips_for_batch
                    )
                    ball_clips_for_batch.clear()
                    ball_pred_idx = 0  # Reset index for new batch results

                if ball_batch_predictions and ball_pred_idx < len(
                    ball_batch_predictions
                ):
                    cached_ball_pred = ball_batch_predictions[ball_pred_idx]
                    # Only increment if we are "consuming" a prediction for the current frame_idx
                    # This alignment is tricky if ball_interval > 1
                    if (
                        frame_idx % self.ball_interval == 0
                    ):  # TODO: and this frame corresponds to the end of a processed clip
                        ball_pred_idx += 1

                # --- Court Prediction ---
                if frame_idx % self.court_interval == 0:
                    court_frames_for_batch.append(frame.copy())

                if len(court_frames_for_batch) >= self.court_batch_size:
                    court_batch_predictions = self.court_predictor.predict(
                        court_frames_for_batch
                    )[
                        0
                    ]  # kps_list
                    court_frames_for_batch.clear()
                    court_pred_idx = 0

                if court_batch_predictions and court_pred_idx < len(
                    court_batch_predictions
                ):
                    if (
                        frame_idx % self.court_interval == 0
                    ):  # Prediction was for this frame
                        cached_court_pred = court_batch_predictions[court_pred_idx]
                        court_pred_idx += 1

                # --- Pose Prediction ---
                if frame_idx % self.pose_interval == 0:
                    pose_frames_for_batch.append(frame.copy())

                if len(pose_frames_for_batch) >= self.pose_batch_size:
                    pose_batch_predictions = self.pose_predictor.predict(
                        pose_frames_for_batch
                    )
                    pose_frames_for_batch.clear()
                    pose_pred_idx = 0

                if pose_batch_predictions and pose_pred_idx < len(
                    pose_batch_predictions
                ):
                    if (
                        frame_idx % self.pose_interval == 0
                    ):  # Prediction was for this frame
                        cached_pose_pred = pose_batch_predictions[pose_pred_idx]
                        pose_pred_idx += 1

                # --- Overlay Drawing ---
                annotated_frame = frame.copy()
                if (
                    cached_ball_pred
                    and cached_ball_pred.get("confidence", 0)
                    >= self.ball_predictor.threshold
                ):
                    annotated_frame = self.ball_predictor.overlay(
                        annotated_frame, cached_ball_pred
                    )
                if cached_court_pred:
                    annotated_frame = self.court_predictor.overlay(
                        annotated_frame, cached_court_pred
                    )
                if cached_pose_pred:
                    annotated_frame = self.pose_predictor.overlay(
                        annotated_frame, cached_pose_pred
                    )

                writer.write(annotated_frame)
                frame_idx += 1
                pbar.update(1)

            # Process any remaining frames in buffers
            if (
                ball_clips_for_batch
            ):  # Should be empty if logic is correct, or handle remaining
                ball_batch_predictions = self.ball_predictor.predict(
                    ball_clips_for_batch
                )
                # Need to write these out, but the main loop is finished. This indicates a slight mismatch in batching logic for final frames.
                # For simplicity, we'll assume the main loop tries to align predictions with frames to be written.
            if court_frames_for_batch:
                court_batch_predictions = self.court_predictor.predict(
                    court_frames_for_batch
                )[0]
                # Similar issue for writing out these last predictions.
            if pose_frames_for_batch:
                pose_batch_predictions = self.pose_predictor.predict(
                    pose_frames_for_batch
                )
                # Similar issue.

            # A more robust way for leftover processing:
            # After the main loop, iterate through any remaining frames that were part of partial batches
            # and ensure their predictions (if made) are overlaid and written.
            # This part is complex and depends on how strictly aligned predictions must be.
            # The current overlay logic uses the *latest cached* prediction, which is reasonable.

        cap.release()
        writer.release()
        self.logger.info(f"処理完了 → 出力動画: {output_path}")
