#!/usr/bin/env python
"""
Hydra-driven Auto Court Annotator (per-image COCO-style)
=========================================================
- 全画像に対して CourtDetectorFPN による 15点キーポイント推論を行い、
- 各 image_id に対応する annotation を COCO アノテーションに追加する。
"""

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.court.predictor import CourtPredictor


# ──────────────────────────────────────────────
# court annotations 追加（image_idごとに1つ）
# ──────────────────────────────────────────────
def flush_batch(
    imgs: List,
    metas: List[Tuple[int, Tuple[int, int]]],
    predictor: CourtPredictor,
    annotations: List[dict],
    starting_id: int,
) -> int:
    kps_batches = predictor.predict(imgs)
    next_id = starting_id

    for (image_id, _), kps in zip(metas, kps_batches, strict=False):
        if not kps or len(kps) < 15:
            continue

        keypoints = []
        scores = []
        for kp in kps:
            x, y = int(kp["x"]), int(kp["y"])
            score = float(kp.get("confidence", 0.0))
            keypoints.extend([x, y, 2])  # visibility=2
            scores.append(score)

        annotations.append(
            {
                "id": next_id,
                "image_id": image_id,
                "category_id": 3,  # court
                "iscrowd": 0,
                "keypoints": keypoints,
                "keypoints_scores": scores,
                "num_keypoints": 15,
            }
        )
        next_id += 1

    return next_id


# ──────────────────────────────────────────────
# メインエントリポイント
# ──────────────────────────────────────────────
@hydra.main(
    config_path="../../configs/annotations",
    config_name="auto_court",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    coco_path = Path(to_absolute_path(cfg.coco_json))
    image_root = Path(to_absolute_path(cfg.image_root))
    output_path = Path(to_absolute_path(cfg.output_json))

    # --- COCO 読み込み ---
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco["images"]
    annotations = coco.get("annotations", [])
    existing_ids = {
        ann["image_id"] for ann in annotations if ann.get("category_id") == 3
    }

    todo = [img for img in images if img["id"] not in existing_ids]
    print(f"未処理 {len(todo)} / 総 {len(images)} フレームを推論します。")

    # --- Court カテゴリが無ければ追加 ---
    if not any(cat["name"] == "court" for cat in coco["categories"]):
        coco["categories"].append(
            {
                "id": 3,
                "name": "court",
                "supercategory": "field",
                "keypoints": [f"pt{i}" for i in range(15)],
                "skeleton": [],
            }
        )

    # --- 推論器セットアップ ---
    predictor = CourtPredictor(
        model_path=cfg.model.path,
        device=cfg.device,
        input_size=tuple(cfg.input_size),
        threshold=cfg.threshold,
        min_distance=cfg.min_distance,
    )

    # --- annotation ID 初期化 ---
    max_ann_id = max((a["id"] for a in annotations), default=0)
    next_ann_id = max_ann_id + 1

    # --- バッチ推論ループ ---
    batch_imgs, batch_meta = [], []
    for img_info in tqdm(todo, desc="Court 推論中"):
        image_id = img_info["id"]
        img_path = image_root / img_info.get("original_path", img_info["file_name"])
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読込失敗: {img_path}")
            continue

        batch_imgs.append(img)
        batch_meta.append((image_id, img.shape[:2]))

        if len(batch_imgs) == cfg.batch_size:
            next_ann_id = flush_batch(
                batch_imgs, batch_meta, predictor, annotations, next_ann_id
            )
            batch_imgs, batch_meta = [], []

    if batch_imgs:
        next_ann_id = flush_batch(
            batch_imgs, batch_meta, predictor, annotations, next_ann_id
        )

    # --- 保存 ---
    coco["annotations"] = annotations
    output_path.write_text(
        json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[完了] 出力ファイル: {output_path.resolve()}")


if __name__ == "__main__":
    main()
