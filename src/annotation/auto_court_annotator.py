#!/usr/bin/env python
"""
Hydra-driven Auto Court Annotator
=================================
(1) COCO から game_id ごとの代表画像を取得
(2) CourtDetectorFPN でキーポイント推論
(3) court_shapes を生成／マージして保存
"""

from pathlib import Path
import json, cv2, torch, hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List, Tuple

from src.court.predictor import CourtPredictor   # 既存実装

# ──────────────────────────────────────────────────────────────
# 内部関数
# ──────────────────────────────────────────────────────────────
def load_game_reps(coco: dict) -> dict[int, dict]:
    reps = {}
    for img in coco["images"]:
        reps.setdefault(img["game_id"], img)  # 1 枚目のみ保持
    return reps


def flush_batch(
    imgs: List, metas: List[Tuple[int, Tuple[int,int]]],
    predictor: CourtPredictor, cache: dict
) -> None:
    """バッチ推論して cache (=court_shapes dict) を更新"""
    kps_batches = predictor.predict(imgs)
    for (gid, _), kps in zip(metas, kps_batches):
        if not kps:           # 検出失敗
            continue
        triplets = []
        for kp in kps:
            triplets.extend([int(kp["x"]), int(kp["y"]), 2])
        cache[gid] = {"game_id": gid, "keypoints": triplets}


# ──────────────────────────────────────────────────────────────
# Hydra エントリポイント
# ──────────────────────────────────────────────────────────────
@hydra.main(config_path="../../configs/annotations", config_name="auto_court", version_base="1.2")
def main(cfg: DictConfig) -> None:           # cfg はネスト可
    print(OmegaConf.to_yaml(cfg, resolve=True))

    coco_path = Path(to_absolute_path(cfg.coco_json))
    image_root = Path(to_absolute_path(cfg.image_root))
    output_path = Path(to_absolute_path(cfg.output_json))

    # --- COCO 読み込み ----------------------------------------------------
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    existing = {c["game_id"]: c for c in coco.get("court_shapes", [])}

    # --- 推論器 -----------------------------------------------------------
    predictor = CourtPredictor(
        model_path=    cfg.model.path,
        device=        cfg.device,
        input_size=    tuple(cfg.input_size),
        threshold=     cfg.threshold,
        min_distance=  cfg.min_distance,
    )

    # --- 推論対象 game_id --------------------------------------------------
    reps = load_game_reps(coco)
    todo = [gid for gid in reps if gid not in existing]
    if not todo:
        print("全 game_id の court_shapes が既に存在します。処理不要で終了。")
        return
    print(f"未処理 {len(todo)} / 総 {len(reps)} ゲームを推論します。")

    # --- バッチ推論ループ --------------------------------------------------
    batch_imgs, batch_meta = [], []
    for gid in tqdm(todo, desc="推論"):
        meta      = reps[gid]
        img_path  = image_root / meta.get("original_path", meta["file_name"])
        img       = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読込失敗: {img_path}")
            continue

        batch_imgs.append(img)
        batch_meta.append((gid, img.shape[:2]))

        if len(batch_imgs) == cfg.batch_size:
            flush_batch(batch_imgs, batch_meta, predictor, existing)
            batch_imgs, batch_meta = [], []

    # 残り端数
    if batch_imgs:
        flush_batch(batch_imgs, batch_meta, predictor, existing)

    # --- JSON 保存 ---------------------------------------------------------
    coco["court_shapes"] = list(existing.values())
    output_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"書き込み完了: {output_path.resolve()}")


if __name__ == "__main__":
    main()
