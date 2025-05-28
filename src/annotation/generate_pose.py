#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.annotation.const import PLAYER_CATEGORY
from src.pose.predictor import PosePredictor
from src.utils.model_utils import load_model_weights


# ─────────────── Dataset 定義 ─────────────────
class ImageDataset(Dataset):
    def __init__(self, images: List[Dict[str, Any]], img_root: Path):
        self.images = images
        self.img_root = img_root

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_info = self.images[idx]
        path = self.img_root / img_info["original_path"]
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"failed loading {path}")
        return frame, img_info["id"]


def flatten_keypoints(
    kps_xy: List[Tuple[int, int]], scores: List[float], vis_thresh: float
) -> Tuple[List[float], int]:
    flat = []
    cnt = 0
    for (x, y), s in zip(kps_xy, scores, strict=False):
        v = 2 if s >= vis_thresh else 1
        cnt += v > 0
        flat.extend([float(x), float(y), int(v)])
    return flat, cnt


# ─────────────── Hydra エントリ ─────────────────
@hydra.main(config_path="../../configs/annotations", config_name="generate_pose")
def main(cfg: DictConfig):
    # 絶対パス化
    json_in = Path(to_absolute_path(cfg.json_in))
    json_out = Path(to_absolute_path(cfg.json_out))
    img_root = Path(to_absolute_path(cfg.img_root))

    # JSON 読み込み
    data = json.loads(json_in.read_text(encoding="utf-8"))

    # categories 更新
    categories = [c for c in data["categories"] if c["id"] not in (2, 3)] + [
        PLAYER_CATEGORY
    ]

    # ball アノテーションはそのまま
    ball_anns = [a for a in data["annotations"] if a["category_id"] == 1]

    # ── モデル準備 ───────────────────────────────────
    det_processor = hydra.utils.instantiate(cfg.det_processor)
    det_model = hydra.utils.instantiate(cfg.det_model)
    det_model = load_model_weights(det_model, to_absolute_path(cfg.det_checkpoint))
    pose_processor = hydra.utils.instantiate(cfg.pose_processor)
    pose_model = hydra.utils.instantiate(cfg.pose_model)

    predictor = PosePredictor(
        det_model=det_model,
        det_processor=det_processor,
        pose_model=pose_model,
        pose_processor=pose_processor,
        device=cfg.device,
        player_label_id=cfg.player_label_id,
        det_score_thresh=cfg.det_score_thresh,
        pose_score_thresh=cfg.pose_score_thresh,
    )

    # ── DataLoader によるバッチ処理 ─────────────────────
    images = data["images"]
    dataset = ImageDataset(images, img_root)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch, strict=False)),
    )

    new_anns: List[Dict[str, Any]] = []
    next_id = max(a["id"] for a in data["annotations"]) + 1

    pbar = tqdm(total=len(dataset), desc="Pose 推論中")
    for frames, ids in loader:
        # frames: tuple of np.ndarray → list に
        frames = list(frames)
        results = predictor.predict(frames)

        for img_id, dets in zip(ids, results, strict=False):
            for d in dets:
                bbox = d["bbox"]
                area = bbox[2] * bbox[3]
                kps_xy = d["keypoints"]
                scores = d["scores"]
                kp_flat, num_vis = flatten_keypoints(
                    kps_xy, scores, cfg.pose_score_thresh
                )
                new_anns.append(
                    {
                        "id": next_id,
                        "image_id": img_id,
                        "category_id": 2,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "keypoints": kp_flat,
                        "num_keypoints": num_vis,
                    }
                )
                next_id += 1

        pbar.update(len(ids))
    pbar.close()

    # ── 出力 JSON 作成 ────────────────────────────────
    out = {
        "info": data["info"],
        "licenses": data["licenses"],
        "images": images,
        "categories": categories,
        "annotations": ball_anns + new_anns,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ JSON を出力しました → {json_out}")


if __name__ == "__main__":
    main()
