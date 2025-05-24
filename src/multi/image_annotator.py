import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

# FrameAnnotatorと同様のカテゴリ定義
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


class ImageAnnotator:
    """
    画像ディレクトリを入力として、Ball/Court/Poseの推論結果を
    単一のCOCO形式JSONファイルに書き出す。各タスクでバッチ推論をサポート。
    """

    def __init__(
        self,
        ball_predictor,
        court_predictor,
        pose_predictor,
        batch_sizes: dict = None,
        ball_vis_thresh: float = 0.5,
        court_vis_thresh: float = 0.5,
        pose_vis_thresh: float = 0.5,
    ):
        self.ball_predictor = ball_predictor
        self.court_predictor = court_predictor
        self.pose_predictor = pose_predictor

        self.batch_sizes = batch_sizes or {"ball": 1, "court": 1, "pose": 1}

        # ボールの時系列予測に必要なスライディングウィンドウ
        self.ball_sliding_window: List[np.ndarray] = []

        self.ball_vis_thresh = ball_vis_thresh
        self.court_vis_thresh = court_vis_thresh
        self.pose_vis_thresh = pose_vis_thresh

        self.coco_output: Dict[str, Any] = {
            "info": {
                "description": "Annotated Images with Batched Predictions",
                "version": "1.1",
                "year": datetime.now().year,
                "contributor": "ImageAnnotator",
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
        file_path: Path,
        height: int,
        width: int,
    ) -> int:
        entry = {
            "id": self.image_id_counter,
            "file_name": file_path.name,
            "original_path": str(file_path),
            "height": height,
            "width": width,
            "license": 1,
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
                "keypoints_scores": [float(s) for s in scores],
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
        meta_buffer: List[Tuple[int, Path]],
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
        input_dir: Union[str, Path],
        output_json: Union[str, Path],
        image_extensions: List[str] = None,
    ):
        """
        画像ディレクトリ内の画像に対して推論を実行し、結果をCOCO形式のJSONファイルに出力する

        Args:
            input_dir: 入力画像が格納されているディレクトリのパス
            output_json: 出力JSONファイルのパス
            image_extensions: 処理対象の画像拡張子リスト（デフォルト: ['.jpg', '.jpeg', '.png']）
        """
        input_dir, output_json = Path(input_dir), Path(output_json)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"入力ディレクトリが存在しないか、ディレクトリではありません: {input_dir}")

        # 出力JSONファイルのディレクトリが存在しない場合は作成
        output_json.parent.mkdir(parents=True, exist_ok=True)

        # 初期化
        self.coco_output["images"].clear()
        self.coco_output["annotations"].clear()
        self.annotation_id_counter = 1
        self.image_id_counter = 1

        # 処理対象の画像拡張子
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png']
        
        # 画像ファイルのリストを取得
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_dir.glob(f"**/*{ext}")))
        
        if not image_files:
            raise ValueError(f"指定されたディレクトリに画像ファイルが見つかりませんでした: {input_dir}")
        
        # 画像のソート（ファイル名順）
        image_files.sort()

        # バッファとキャッシュ
        ball_buf, court_buf, pose_buf = [], [], []
        ball_meta, court_meta, pose_meta = [], [], []
        ball_cache, court_cache, pose_cache = {}, {}, {}

        with tqdm(total=len(image_files), desc="画像アノテーション処理中") as pbar:
            for img_path in image_files:
                # 画像読み込み
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"警告: 画像の読み込みに失敗しました: {img_path}")
                    continue

                # 画像エントリの追加
                img_id = self._add_image_entry(
                    img_path, frame.shape[0], frame.shape[1]
                )

                # Ball 用クリップ管理（時系列モデルの場合）
                self.ball_sliding_window.append(frame.copy())
                if len(self.ball_sliding_window) > self.ball_predictor.num_frames:
                    self.ball_sliding_window.pop(0)
                if len(self.ball_sliding_window) >= self.ball_predictor.num_frames:
                    ball_buf.append(list(self.ball_sliding_window))
                    ball_meta.append((img_id, img_path))

                # Court
                court_buf.append(frame.copy())
                court_meta.append((img_id, img_path))

                # Pose
                pose_buf.append(frame.copy())
                pose_meta.append((img_id, img_path))

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

        # JSON 出力
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(self.coco_output, f, ensure_ascii=False, indent=2)

        print(f"✅ 完了！COCO形式アノテーションを保存しました: {output_json}")
        print(f"総画像数: {len(self.coco_output['images'])}")
        print(f"総アノテーション数: {len(self.coco_output['annotations'])}") 