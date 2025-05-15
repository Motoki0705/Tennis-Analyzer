import os, random, json
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A

from src.ball.utils.visualize import play_overlay_sequence_xy

class SequenceKeypointDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        N: int,
        input_size: List[int],
        transform: A.ReplayCompose,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        seed: int = 42,
        skip_frames_range: Tuple[int, int] = (1, 5)
    ):
        # 基本設定
        self.N = N
        self.in_h, self.in_w = input_size   # 例 [512,512]
        self.image_root = Path(image_root)
        self.transform = transform
        self.split = split
        self.skip_min, self.skip_max = skip_frames_range

        # JSON 読み込み
        data = json.load(open(annotation_file, "r", encoding="utf-8"))
        self.images = {img["id"]: img for img in data["images"]}
        self.anns_by_image = {
            ann["image_id"]: ann
            for ann in data["annotations"]
            if ann["category_id"] == 1
        }

        # クリップ → フレーム ID グループ化
        clip_groups = defaultdict(list)
        for img in data["images"]:
            clip_groups[(img["game_id"], img["clip_id"])].append(img["id"])

        # クリップ単位 split
        random.seed(seed)
        clip_keys = list(clip_groups.keys())
        random.shuffle(clip_keys)
        n = len(clip_keys)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        split_map = {
            "train": clip_keys[:n_train],
            "val":   clip_keys[n_train:n_train+n_val],
            "test":  clip_keys[n_train+n_val:],
        }
        target_clip_keys = set(split_map[split])

        # スライディングウィンドウ列挙
        self.windows = []
        for key in target_clip_keys:
            ids_sorted = sorted(clip_groups[key])
            for start in range(0, len(ids_sorted) - N + 1):
                self.windows.append((key, start))
        self.clip_groups = clip_groups
        print(f"[{split.upper()}] clips: {len(target_clip_keys)}, windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        clip_key, start = self.windows[idx]
        ids_sorted = sorted(self.clip_groups[clip_key])
        L = len(ids_sorted)

        # ランダム skip（学習時のみ）
        if self.split == "train" and (self.skip_min, self.skip_max) != (1, 1):
            max_skip = (L - 1 - start) // (self.N - 1)
            skip_upper = max(1, min(self.skip_max, max_skip))
            skip = random.randint(self.skip_min, skip_upper)
            frame_ids = [ids_sorted[start + i * skip] for i in range(self.N)]
        else:
            frame_ids = ids_sorted[start:start + self.N]

        frames, coords, visibilities, replay = [], [], [], None
        for i, fid in enumerate(frame_ids):
            info = self.images[fid]
            img_np = np.array(Image.open(self.image_root / info["original_path"]).convert("RGB"))

            ann = self.anns_by_image.get(fid)
            if ann is not None:
                x, y, v = ann["keypoints"][:3]
            else:
                x, y, v = 0.0, 0.0, 0        # アノテ無し

            # Albumentations 変換
            if i == 0:
                res = self.transform(image=img_np, keypoints=[(x, y)] if v else [])
                replay = res["replay"]
            else:
                res = A.ReplayCompose.replay(replay, image=img_np, keypoints=[(x, y)] if v else [])

            img_aug = res["image"]
            kp_aug  = res["keypoints"]

            # 正規化座標
            if kp_aug:
                xn = kp_aug[0][0] / self.in_w
                yn = kp_aug[0][1] / self.in_h
            else:
                xn, yn = 0.0, 0.0
                v = 0   # visibility も 0 扱い

            frames.append(img_aug)               # Tensor [3,H,W]
            coords.append([xn, yn])
            visibilities.append(v)

        frames_tensor = torch.stack(frames, dim=0).permute(1,0,2,3)  # [3,N,H,W]
        coords_tensor = torch.tensor(coords, dtype=torch.float32)    # [N,2]
        vis_tensor    = torch.tensor(visibilities, dtype=torch.float32)  # [N]

        return frames_tensor, coords_tensor, vis_tensor


if __name__ == '__main__':
    input_size = [512, 512]
    heatmap_size = [512, 512]

    transform = A.ReplayCompose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),

        # --- 以下が追加分 ---
        A.ColorJitter(),

        # 正規化＆Tensor変換（最後）
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))


    dataset = SequenceKeypointDataset(
        annotation_file=r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\annotation_jsons\coco_annotations_globally_tracked.json",
        image_root=r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\images",
        N=20,
        input_size=input_size,
        transform=transform,
        split="train",
        skip_frames_range=[1, 5]
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for frames, coords, visibilty in dataloader:
        print(frames.shape)
        print(coords.shape)
        print(visibilty.shape)
        break

    play_overlay_sequence_xy(frames=frames[0], coords=coords[0], vis=visibilty[0])