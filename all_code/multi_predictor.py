import cv2
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Union

class MultiPredictor:
    def __init__(
        self,
        ball_predictor,
        court_predictor,
        pose_predictor,
        ball_interval: int = 1,
        court_interval: int = 30,
        pose_interval: int = 5,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

        self.ball_predictor  = ball_predictor
        self.court_predictor = court_predictor
        self.pose_predictor  = pose_predictor

        # 各タスク推論インターバル
        self.ball_interval  = ball_interval
        self.court_interval = court_interval
        self.pose_interval  = pose_interval

        # 直前結果キャッシュ
        self.last_ball  = None
        self.last_court = None
        self.last_pose  = None

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        input_path  = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error(f"動画を開けませんでした: {input_path}")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"読み込み完了 → フレーム数: {total}, FPS: {fps:.2f}, 解像度: {width}×{height}"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        buffer_frames: List[np.ndarray] = []
        frame_idx = 0

        with tqdm(total=total, desc="Multi 推論処理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Ball 用のスライディングウィンドウ
                buffer_frames.append(frame)
                if len(buffer_frames) > self.ball_predictor.num_frames:
                    buffer_frames.pop(0)

                # ── Ball 推論 ──
                if frame_idx % self.ball_interval == 0 and len(buffer_frames) >= self.ball_predictor.num_frames:
                    clip = [buffer_frames[-self.ball_predictor.num_frames:]]
                    self.last_ball = self.ball_predictor.predict(clip)[0]

                # ── Court 推論 ──
                if frame_idx % self.court_interval == 0:
                    self.last_court = self.court_predictor.predict([frame])[0]

                # ── Pose 推論 ──
                if frame_idx % self.pose_interval == 0:
                    self.last_pose = self.pose_predictor.predict([frame])[0]

                # ── オーバーレイ 描画 ──
                annotated = frame.copy()
                # Ball
                if self.last_ball and self.last_ball["confidence"] >= self.ball_predictor.threshold:
                    annotated = self.ball_predictor.overlay(annotated, self.last_ball)
                # Court
                if self.last_court:
                    annotated = self.court_predictor.overlay(annotated, self.last_court)
                # Pose (bbox + keypoints)
                if self.last_pose:
                    annotated = self.pose_predictor.overlay(annotated, self.last_pose)

                writer.write(annotated)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        writer.release()
        self.logger.info(f"処理完了 → 出力動画: {output_path}")
