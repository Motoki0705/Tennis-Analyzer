import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Union

class FrameAnnotator:
    """
    動画をフレームごとに JPG 保存しつつ、Ball/Court/Pose の推論結果を JSONL に書き出す。
    """
    def __init__(
        self,
        ball_predictor,
        court_predictor,
        pose_predictor,
        intervals: dict = None,
        frame_fmt: str = "frame_{:06d}.jpg"
    ):
        self.ball = ball_predictor
        self.court = court_predictor
        self.pose = pose_predictor

        # 各タスク推論間隔（フレーム単位）
        self.intv = intervals or {"ball":1, "court":1, "pose":1}
        # sliding window for ball
        self.buf = []

        self.frame_fmt = frame_fmt

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_json: Union[str, Path]
    ):
        input_path  = Path(input_path)
        output_dir  = Path(output_dir)
        output_json = Path(output_json)

        os.makedirs(output_dir, exist_ok=True)
        # JSONL で一行ずつ書き出し
        fout = open(output_json, "w", encoding="utf-8")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        with tqdm(total=total, desc="Frame & Predict") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- 1) フレーム保存 ---
                fname = self.frame_fmt.format(frame_idx)
                save_path = output_dir / fname
                cv2.imwrite(str(save_path), frame)

                # --- 2) 推論準備 ---
                # Ball 用スライディングウィンドウ
                self.buf.append(frame)
                if len(self.buf) > self.ball.num_frames:
                    self.buf.pop(0)

                record = {"frame": frame_idx, "file_name": fname}

                # Ball 推論
                if frame_idx % self.intv["ball"] == 0 and len(self.buf) >= self.ball.num_frames:
                    ball_res = self.ball.predict([self.buf])[0]
                else:
                    ball_res = {"x": None, "y": None, "confidence": None}
                record["ball"]  = ball_res

                # Court 推論
                if frame_idx % self.intv["court"] == 0:
                    court_res = self.court.predict([frame])[0]
                else:
                    court_res = [{"x": None, "y": None, "confidence": None}]
                record["court_kps"] = court_res

                # Pose 推論
                if frame_idx % self.intv["pose"] == 0:
                    pose_res = self.pose.predict([frame])[0]
                else:
                    pose_res = []
                record["pose"] = pose_res

                # --- 3) JSONL 出力 ---
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                frame_idx += 1
                pbar.update(1)

        cap.release()
        fout.close()
        print(f"✅ Done! Frames in `{output_dir}`, annotations in `{output_json}`")
