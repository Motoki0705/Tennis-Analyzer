# multi_flow_annotator/coco_utils.py

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .definitions import BALL_CATEGORY, PLAYER_CATEGORY, COURT_CATEGORY

class CocoManager:
    """COCO形式のアノテーションデータを管理、生成、保存するクラス。

    このクラスはスレッドセーフです。

    Attributes:
        coco_output (Dict[str, Any]): COCO形式のデータ全体を保持する辞書。
        annotation_lock (threading.Lock): アノテーションデータへのアクセスを保護するロック。
        annotation_id_counter (int): アノテーションIDを生成するためのカウンター。
        image_id_counter (int): 画像IDを生成するためのカウンター。
    """
    def __init__(self):
        """CocoManagerのインスタンスを初期化します。"""
        self.coco_output: Dict[str, Any] = self._initialize_coco_data()
        self.annotation_lock = threading.Lock()
        self.annotation_id_counter = 1
        self.image_id_counter = 1

    def _initialize_coco_data(self) -> Dict[str, Any]:
        """COCOデータ構造の骨格を初期化します。

        Returns:
            Dict[str, Any]: 初期化されたCOCO形式の辞書。
        """
        return {
            "info": {
                "description": "Annotated Images with Multi-Flow Predictions",
                "version": "2.0", "year": datetime.now().year,
                "contributor": "MultiFlowAnnotator",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "Placeholder License", "url": ""}],
            "categories": [BALL_CATEGORY, PLAYER_CATEGORY, COURT_CATEGORY],
            "images": [],
            "annotations": [],
        }

    def add_image_entry(
        self, file_path: Path, height: int, width: int,
        game_id: int, clip_id: int, base_dir: Path
    ) -> int:
        """COCO形式の画像エントリを追加します。

        Args:
            file_path (Path): 画像ファイルのパス。
            height (int): 画像の高さ。
            width (int): 画像の幅。
            game_id (int): ゲームID。
            clip_id (int): クリップID。
            base_dir (Path): 相対パスを計算するための基準ディレクトリ。

        Returns:
            int: 追加された画像のID。
        """
        rel_path = str(file_path.relative_to(base_dir))
        entry = {
            "id": self.image_id_counter,
            "file_name": file_path.name,
            "original_path": rel_path,
            "height": height,
            "width": width,
            "license": 1,
            "game_id": game_id,
            "clip_id": clip_id
        }
        with self.annotation_lock:
            self.coco_output["images"].append(entry)
            img_id = self.image_id_counter
            self.image_id_counter += 1
        return img_id

    def add_ball_annotation(self, image_id: int, ball_res: Dict, vis_thresh: float):
        """ボールのアノテーションをCOCOデータに追加します。

        Args:
            image_id (int): 対応する画像のID。
            ball_res (Dict): ボールの検出結果 (x, y, confidenceを含む辞書)。
            vis_thresh (float): 可視と判断するための信頼度の閾値。
        """
        if not ball_res: return
        x, y, conf = ball_res.get("x"), ball_res.get("y"), ball_res.get("confidence", 0.0)
        if x is None or y is None: return

        visibility = 2 if conf >= vis_thresh else 1
        ann = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": BALL_CATEGORY["id"],
            "keypoints": [float(x), float(y), visibility],
            "num_keypoints": 1, "iscrowd": 0,
            "score": float(conf),
        }
        with self.annotation_lock:
            self.coco_output["annotations"].append(ann)
            self.annotation_id_counter += 1

    def add_court_annotation(self, image_id: int, court_kps: List[Dict], vis_thresh: float):
        """コートのアノテーションをCOCOデータに追加します。

        Args:
            image_id (int): 対応する画像のID。
            court_kps (List[Dict]): コートのキーポイント検出結果のリスト。
            vis_thresh (float): 可視と判断するための信頼度の閾値。
        """
        if not court_kps: return
        num_expected = len(COURT_CATEGORY["keypoints"])
        keypoints_flat, keypoints_scores = [], []
        num_visible = 0

        for i in range(num_expected):
            if i < len(court_kps):
                kp = court_kps[i]
                x, y, conf = kp.get("x", 0.0), kp.get("y", 0.0), kp.get("confidence", 0.0)
            else:
                x, y, conf = 0.0, 0.0, 0.0
            
            v = 2 if conf >= vis_thresh else (1 if conf > 0.01 else 0)
            keypoints_flat.extend([float(x), float(y), v])
            keypoints_scores.append(float(conf))
            if v > 0: num_visible += 1

        ann = {
            "id": self.annotation_id_counter, "image_id": image_id,
            "category_id": COURT_CATEGORY["id"],
            "keypoints": keypoints_flat, "num_keypoints": num_visible,
            "keypoints_scores": keypoints_scores, "iscrowd": 0,
        }
        with self.annotation_lock:
            self.coco_output["annotations"].append(ann)
            self.annotation_id_counter += 1

    def add_pose_annotations(self, image_id: int, pose_results: List[Dict], vis_thresh: float):
        """複数のポーズアノテーションをCOCOデータに追加します。

        Args:
            image_id (int): 対応する画像のID。
            pose_results (List[Dict]): ポーズ推定結果のリスト。
            vis_thresh (float): 可視と判断するための信頼度の閾値。
        """
        if not pose_results: return
        for res in pose_results:
            bbox, kps, scores, det_score = (
                res.get("bbox"), res.get("keypoints"), res.get("scores"), res.get("det_score", 0.0)
            )
            if not all([bbox, kps, scores]): continue
            
            flat_kps, num_vis = [], 0
            for (x, y), s in zip(kps, scores, strict=False):
                v = 2 if s >= vis_thresh else (1 if s > 0.01 else 0)
                if v > 0: num_vis += 1
                flat_kps.extend([float(x), float(y), v])

            ann = {
                "id": self.annotation_id_counter, "image_id": image_id,
                "category_id": PLAYER_CATEGORY["id"],
                "bbox": [float(b) for b in bbox],
                "area": float(bbox[2] * bbox[3]),
                "keypoints": flat_kps,
                "keypoints_scores": [float(s) for s in scores],
                "num_keypoints": num_vis, "iscrowd": 0,
                "score": float(det_score),
            }
            with self.annotation_lock:
                self.coco_output["annotations"].append(ann)
                self.annotation_id_counter += 1

    def save_to_json(self, output_path: Path):
        """生成されたCOCOアノテーションをJSONファイルに保存します。

        Args:
            output_path (Path): 出力するJSONファイルのパス。
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.coco_output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 完了！COCO形式アノテーションを保存しました: {output_path}")
        print(f"総画像数: {len(self.coco_output['images'])}")
        print(f"総アノテーション数: {len(self.coco_output['annotations'])}")
        
        category_counts = {cat["id"]: 0 for cat in self.coco_output["categories"]}
        for ann in self.coco_output["annotations"]:
            category_counts[ann["category_id"]] += 1
        
        for cat in self.coco_output["categories"]:
            print(f"  - {cat['name']} アノテーション数: {category_counts[cat['id']]}")
