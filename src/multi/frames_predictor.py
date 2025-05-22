import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

# Assuming PLAYER_CATEGORY, COURT_CATEGORY, BALL_CATEGORY are defined
try:
    from src.annotation.const import PLAYER_CATEGORY
except ImportError:
    PLAYER_CATEGORY = {
        "id": 2,
        "name": "player",
        "supercategory": "person",
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        "skeleton": [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [7, 9],
            [6, 8],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ],
    }

COURT_CATEGORY = {
    "id": 3,
    "name": "court",
    "supercategory": "field",
    "keypoints": [f"pt{i}" for i in range(15)],
    "skeleton": [],
}

BALL_CATEGORY = {
    "id": 1,
    "name": "ball",
    "supercategory": "sports",
    "keypoints": ["center"],
    "skeleton": [],
}


class FrameAnnotator:
    """
    動画をフレームごとに JPG 保存しつつ、Ball/Court/Pose の推論結果を
    単一の COCO 形式 JSON ファイルに書き出す。各タスクでバッチ推論をサポート。
    """

    def __init__(
        self,
        ball_predictor,
        court_predictor,
        pose_predictor,
        intervals: dict = None,
        batch_sizes: dict = None,
        frame_fmt: str = "frame_{:06d}.jpg",
        ball_vis_thresh: float = 0.5,
        court_vis_thresh: float = 0.5,
        pose_vis_thresh: float = 0.5,
    ):
        self.ball_predictor = ball_predictor
        self.court_predictor = court_predictor
        self.pose_predictor = pose_predictor

        self.intervals = intervals or {"ball": 1, "court": 1, "pose": 1}
        self.batch_sizes = batch_sizes or {"ball": 1, "court": 1, "pose": 1}

        self.ball_sliding_window: List[np.ndarray] = []

        self.frame_fmt = frame_fmt
        self.ball_vis_thresh = ball_vis_thresh
        self.court_vis_thresh = court_vis_thresh
        self.pose_vis_thresh = pose_vis_thresh

        self.coco_output: Dict[str, Any] = {
            "info": {
                "description": "Annotated Frames from Video with Batched Predictions",
                "version": "1.1",
                "year": datetime.now().year,
                "contributor": "FrameAnnotator",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "Placeholder License", "url": ""}],
            "categories": [BALL_CATEGORY, PLAYER_CATEGORY, COURT_CATEGORY],
            "images": [],
            "annotations": [],
        }

        self.annotation_id_counter = 1
        self.image_id_counter = 1

    def _add_image_entry(
        self,
        frame_idx: int,
        file_name: str,
        height: int,
        width: int,
        video_path_str: str = "",
    ) -> int:
        entry = {
            "id": self.image_id_counter,
            "file_name": file_name,
            "original_path": file_name,
            "height": height,
            "width": width,
            "license": 1,
            "frame_idx_in_video": frame_idx,
            "source_video": video_path_str,
        }
        self.coco_output["images"].append(entry)
        img_id = self.image_id_counter
        self.image_id_counter += 1
        return img_id

    def _add_ball_annotation(self, image_id: int, ball_res: Dict):
        if not ball_res:
            return
        x, y, conf = (
            ball_res.get("x"),
            ball_res.get("y"),
            ball_res.get("confidence", 0.0),
        )
        if x is None or y is None:
            return
        visibility = 2 if conf >= self.ball_vis_thresh else 1
        ann = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": BALL_CATEGORY["id"],
            "keypoints": [float(x), float(y), visibility],
            "num_keypoints": 1 if visibility > 0 else 0,
            "iscrowd": 0,
            "score": float(conf),
        }
        self.coco_output["annotations"].append(ann)
        self.annotation_id_counter += 1

    def _add_court_annotation(self, image_id: int, court_kps: List[Dict]):
        if not court_kps:
            return
        num_expected = len(COURT_CATEGORY["keypoints"])
        keypoints_flat, keypoints_scores = [], []
        num_visible = 0

        for i in range(num_expected):
            if i < len(court_kps):
                kp = court_kps[i]
                x, y, conf = (
                    kp.get("x", 0.0),
                    kp.get("y", 0.0),
                    kp.get("confidence", 0.0),
                )
            else:
                x, y, conf = 0.0, 0.0, 0.0

            if conf >= self.court_vis_thresh:
                v = 2
            elif conf > 0.01:
                v = 1
            else:
                v = 0

            keypoints_flat.extend([float(x), float(y), v])
            keypoints_scores.append(float(conf))
            if v > 0:
                num_visible += 1

        ann = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": COURT_CATEGORY["id"],
            "keypoints": keypoints_flat,
            "num_keypoints": num_visible,
            "keypoints_scores": keypoints_scores,
            "iscrowd": 0,
        }
        self.coco_output["annotations"].append(ann)
        self.annotation_id_counter += 1

    def _add_pose_annotations(self, image_id: int, pose_results: List[Dict]):
        if not pose_results:
            return
        for res in pose_results:
            bbox = res.get("bbox")
            kps = res.get("keypoints")
            scores = res.get("scores")
            det_score = res.get("det_score", 0.0)
            if not bbox or not kps or not scores:
                continue
            flat, num_vis = [], 0
            for (x, y), s in zip(kps, scores, strict=False):
                if s >= self.pose_vis_thresh:
                    v = 2
                    num_vis += 1
                elif s > 0.01:
                    v = 1
                else:
                    v = 0
                flat.extend([float(x), float(y), v])

            ann = {
                "id": self.annotation_id_counter,
                "image_id": image_id,
                "category_id": PLAYER_CATEGORY["id"],
                "bbox": [float(b) for b in bbox],
                "area": float(bbox[2] * bbox[3]),
                "keypoints": flat,  # [x0, y0, v0, x1, y1, v1, ...]
                "keypoints_scores": [float(s) for s in scores],  # ← 新しく追加
                "num_keypoints": num_vis,
                "iscrowd": 0,
                "score": float(det_score),
            }
            self.coco_output["annotations"].append(ann)
            self.annotation_id_counter += 1

    def _process_batch(
        self,
        predictor,
        frames_buffer: List,
        meta_buffer: List[Tuple[int, int]],
        cache: Dict[int, Any],
    ):
        if not frames_buffer:
            return

        # モデル予測の呼び出し
        if predictor == self.ball_predictor:
            preds = predictor.predict(frames_buffer)
        else:
            out = predictor.predict(frames_buffer)
            preds = out[0] if predictor == self.court_predictor else out

        # キャッシュ登録とアノテーション追加
        for (img_id, _), pred in zip(meta_buffer, preds, strict=False):
            cache[img_id] = pred
            if predictor == self.ball_predictor:
                self._add_ball_annotation(img_id, pred)
            elif predictor == self.court_predictor:
                self._add_court_annotation(img_id, pred)
            else:
                self._add_pose_annotations(img_id, pred)

        frames_buffer.clear()
        meta_buffer.clear()

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_json: Union[str, Path],
    ):
        input_path, output_dir, output_json = map(
            Path, (input_path, output_dir, output_json)
        )
        os.makedirs(output_dir, exist_ok=True)

        # 初期化
        self.coco_output["images"].clear()
        self.coco_output["annotations"].clear()
        self.annotation_id_counter = 1
        self.image_id_counter = 1

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # バッファとキャッシュ
        ball_buf, court_buf, pose_buf = [], [], []
        ball_meta, court_meta, pose_meta = [], [], []
        ball_cache, court_cache, pose_cache = {}, {}, {}

        with tqdm(total=total, desc="Frame Annotation & Batched Predict") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # フレーム保存 & image entry
                fname = self.frame_fmt.format(frame_idx)
                cv2.imwrite(str(output_dir / fname), frame)
                img_id = self._add_image_entry(
                    frame_idx, fname, frame.shape[0], frame.shape[1], str(input_path)
                )

                # Ball 用クリップ管理
                self.ball_sliding_window.append(frame.copy())
                if len(self.ball_sliding_window) > self.ball_predictor.num_frames:
                    self.ball_sliding_window.pop(0)
                if (
                    frame_idx % self.intervals["ball"] == 0
                    and len(self.ball_sliding_window) >= self.ball_predictor.num_frames
                ):
                    ball_buf.append(list(self.ball_sliding_window))
                    ball_meta.append((img_id, frame_idx))

                # Court
                if frame_idx % self.intervals["court"] == 0:
                    court_buf.append(frame.copy())
                    court_meta.append((img_id, frame_idx))

                # Pose
                if frame_idx % self.intervals["pose"] == 0:
                    pose_buf.append(frame.copy())
                    pose_meta.append((img_id, frame_idx))

                # バッチ処理
                if len(ball_buf) >= self.batch_sizes["ball"]:
                    self._process_batch(
                        self.ball_predictor, ball_buf, ball_meta, ball_cache
                    )
                if len(court_buf) >= self.batch_sizes["court"]:
                    self._process_batch(
                        self.court_predictor, court_buf, court_meta, court_cache
                    )
                if len(pose_buf) >= self.batch_sizes["pose"]:
                    self._process_batch(
                        self.pose_predictor, pose_buf, pose_meta, pose_cache
                    )

                frame_idx += 1
                pbar.update(1)

            # 残りバッチの処理
            if ball_buf:
                self._process_batch(
                    self.ball_predictor, ball_buf, ball_meta, ball_cache
                )
            if court_buf:
                self._process_batch(
                    self.court_predictor, court_buf, court_meta, court_cache
                )
            if pose_buf:
                self._process_batch(
                    self.pose_predictor, pose_buf, pose_meta, pose_cache
                )

        cap.release()

        # JSON 出力
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(self.coco_output, f, ensure_ascii=False, indent=2)

        print(
            f"✅ Done! Frames saved in `{output_dir}`, COCO annotations in `{output_json}`"
        )
        print(f"Total images: {len(self.coco_output['images'])}")
        print(f"Total annotations: {len(self.coco_output['annotations'])}")
