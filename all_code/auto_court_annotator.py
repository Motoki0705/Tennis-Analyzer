#!/usr/bin/env python
"""
Hydra-driven Auto Court Annotator (per-frame version)
=====================================================
(1) COCO から全画像を取得
(2) CourtDetectorFPN でキーポイント推論
(3) 各 image_id ごとに court_shapes を保存
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.court.predictor import CourtPredictor


# ──────────────────────────────────────────────
# バッチ推論して `court_shapes` に追記する関数
# ──────────────────────────────────────────────
def flush_batch(
    imgs: List, metas: List[Tuple[int, Tuple[int, int]]],
    predictor: CourtPredictor, cache: Dict[int, dict]
) -> None:
    kps_batches = predictor.predict(imgs)
    for (image_id, _), kps in zip(metas, kps_batches):
        if not kps:
            continue
        triplets = []
        for kp in kps:
            triplets.extend([int(kp["x"]), int(kp["y"]), 2])  # visibility=2
        cache[image_id] = {
            "image_id": image_id,
            "keypoints": triplets
        }


# ──────────────────────────────────────────────
# メインエントリポイント
# ──────────────────────────────────────────────
@hydra.main(config_path="../../configs/annotations", config_name="auto_court", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    coco_path = Path(to_absolute_path(cfg.coco_json))
    image_root = Path(to_absolute_path(cfg.image_root))
    output_path = Path(to_absolute_path(cfg.output_json))

    # --- COCO 読み込み ---
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    existing = {c["image_id"]: c for c in coco.get("court_shapes", [])}
    images = coco["images"]
    todo = [img for img in images if img["id"] not in existing]

    if not todo:
        print("すべての image_id に court_shapes が存在します。処理不要で終了。")
        return
    print(f"未処理 {len(todo)} / 総 {len(images)} フレームを推論します。")

    # --- 推論器セットアップ ---
    predictor = CourtPredictor(
        model_path=cfg.model.path,
        device=cfg.device,
        input_size=tuple(cfg.input_size),
        threshold=cfg.threshold,
        min_distance=cfg.min_distance,
    )

    # --- バッチ推論ループ ---
    batch_imgs, batch_meta = [], []
    for img_info in tqdm(todo, desc="フレーム推論中"):
        image_id = img_info["id"]
        img_path = image_root / img_info.get("original_path", img_info["file_name"])
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読込失敗: {img_path}")
            continue

        batch_imgs.append(img)
        batch_meta.append((image_id, img.shape[:2]))

        if len(batch_imgs) == cfg.batch_size:
            flush_batch(batch_imgs, batch_meta, predictor, existing)
            batch_imgs, batch_meta = [], []

    # --- 残り端数の処理 ---
    if batch_imgs:
        flush_batch(batch_imgs, batch_meta, predictor, existing)

    # --- COCO 形式に保存 ---
    coco["court_shapes"] = list(existing.values())
    output_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[完了] court_shapes 書き込み先: {output_path.resolve()}")


if __name__ == "__main__":
    main()
