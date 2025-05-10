import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from torch.utils.data import DataLoader

from src.ball.utils.heatmap import generate_gaussian_heatmap
from src.ball.utils.visualize import play_overlay_sequence

class SequenceKeypointDataset(Dataset):
    """
    SequenceKeypointDataset:
    ------------------------
    動画シーケンスからNフレームを読み込み、時系列キーポイントヒートマップを生成するデータセット。

    Args:
        annotation_file (str): COCO形式アノテーションファイル（JSON）。
        image_root (str): 画像ファイルのルートディレクトリ。
        T (int): 連続するフレーム数。
        input_size (List[int]): 入力画像サイズ (H, W)。
        heatmap_size (List[int]): 出力ヒートマップサイズ (H, W)。
        transform (A.ReplayCompose): Albumentations変換。
        split (str): "train" / "val" / "test"。
        use_group_split (bool): クリップ単位でデータ分割するかどうか。
        train_ratio (float): トレーニングセット比率。
        val_ratio (float): 検証セット比率。
        seed (int): 乱数シード。
        input_type (str): 
            入力のフォーマット:
            - "cat": [C×T, H, W]
            - "stack": [N, C, H, W]
        output_type (str): 
            出力ヒートマップのフォーマット:
            - "all": [N, H, W]
            - "last": [H, W]
        skip_frames_range (Tuple[int, int]): スキップするフレーム範囲（訓練時のみ）。
    """
    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        T: int,
        input_size: List[int],
        heatmap_size: List[int],
        transform: A.ReplayCompose,
        split: str = "train",
        use_group_split: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        seed: int = 42,
        input_type: str = "stack",
        output_type: str = "all",
        skip_frames_range: Tuple[int, int] = (1, 5)
    ):
        assert input_type in {"cat", "stack"}, "input_type must be 'cat' or 'stack'"
        assert output_type in {"all", "last"}, "output_type must be 'all' or 'last'"

        self.T = T
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.image_root = Path(image_root)
        self.transform = transform
        self.split = split
        self.skip_min, self.skip_max = skip_frames_range
        self.input_type = input_type
        self.output_type = output_type

        # JSON読み込み
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self.images = {img["id"]: img for img in data["images"]}
        self.anns_by_image = {
            ann["image_id"]: ann
            for ann in data["annotations"]
            if ann["category_id"] == 1
        }

        # クリップ単位でフレームをグループ化
        clip_groups = defaultdict(list)
        for img in data["images"]:
            key = (img["game_id"], img["clip_id"])
            clip_groups[key].append(img["id"])

        # Split
        clip_keys = list(clip_groups.keys())
        if use_group_split:
            train_keys, val_keys, test_keys = self.group_split_clip(
                clip_keys, train_ratio, val_ratio, seed
            )
        else:
            random.seed(seed)
            random.shuffle(clip_keys)
            n_train = int(len(clip_keys) * train_ratio)
            n_val = int(len(clip_keys) * val_ratio)
            train_keys = clip_keys[:n_train]
            val_keys = clip_keys[n_train : n_train + n_val]
            test_keys = clip_keys[n_train + n_val :]

        split2keys = {"train": train_keys, "val": val_keys, "test": test_keys}
        target_clip_keys = set(split2keys[split])

        # Nフレーム単位のウィンドウを列挙
        self.windows: List[Tuple[Tuple[int, int], int]] = []
        for key in target_clip_keys:
            ids_sorted = sorted(clip_groups[key])
            if len(ids_sorted) < T:
                continue
            for start in range(0, len(ids_sorted) - T + 1):
                self.windows.append((key, start))

        self.clip_groups = clip_groups
        print(f"[{split.upper()}] clips: {len(target_clip_keys)}, windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        clip_key, start = self.windows[idx]
        ids_sorted = sorted(self.clip_groups[clip_key])
        L = len(ids_sorted)

        # スキップ適用（訓練時のみ）
        if self.T > 1 and self.split == "train" and (self.skip_min, self.skip_max) != (1, 1):
            max_allowed = (L - 1 - start) // (self.T - 1)
            skip_upper = min(self.skip_max, max_allowed) if max_allowed > 0 else 1
            skip = random.randint(self.skip_min, skip_upper)
            frame_ids = [ids_sorted[start + i * skip] for i in range(self.T)]
        else:
            frame_ids = ids_sorted[start:start + self.T]

        frames, heatmaps, visibilities, replay = [], [], [], None
        for i, img_id in enumerate(frame_ids):
            info = self.images[img_id]
            img_path = self.image_root / info["original_path"]
            img_np = np.array(Image.open(img_path).convert("RGB"))
            h_img, w_img = img_np.shape[:2]

            ann = self.anns_by_image.get(img_id)
            if ann is not None:
                x, y, v = ann["keypoints"][:3]
                if x >= w_img or y >= h_img or x < 0 or y < 0:
                    x, y = self._clip_keypoints((x, y), (h_img, w_img))
                keypoints = [(x, y)]
                visibilities.append(v)
            else:
                keypoints = []
                visibilities.append(0)  # アノテーションがない場合は不可視扱い


            if i == 0:
                out = self.transform(image=img_np, keypoints=keypoints)
                replay = out["replay"]
            else:
                out = A.ReplayCompose.replay(replay, image=img_np, keypoints=keypoints)

            img_aug = out["image"]
            kp_aug  = out["keypoints"]

            if kp_aug:
                hm = generate_gaussian_heatmap(
                    raw_label={"keypoints": [*kp_aug[0], 2]},
                    input_size=self.input_size,
                    output_size=self.heatmap_size,
                ).squeeze(0)  # [H, W]
            else:
                hm = torch.zeros(self.heatmap_size, dtype=torch.float32)

            frames.append(img_aug)
            heatmaps.append(hm)

        # 入力フォーマット
        if self.input_type == "cat":
            input_tensor = torch.cat(frames, dim=0)  # [C*N, H, W]
        else:
            input_tensor = torch.stack(frames, dim=0)  # [N, C, H, W]

        # 出力フォーマット
        if self.output_type == "all":
            heatmap_tensor = torch.stack(heatmaps, dim=0)  # [N, H, W]
            visibility_tensor = torch.tensor(visibilities, dtype=torch.float32)  # [N]
        else:
            heatmap_tensor = heatmaps[-1]  # [H, W]
            visibility_tensor = torch.tensor(visibilities[-1:], dtype=torch.float32)  # [1]


        return input_tensor, heatmap_tensor, visibility_tensor

    @staticmethod
    def _clip_keypoints(kp_xy: Tuple[float, float], img_hw: Tuple[int, int]) -> Tuple[float, float]:
        h, w = img_hw
        x, y = kp_xy
        x = min(max(x, 0.0), w - 1e-3)
        y = min(max(y, 0.0), h - 1e-3)
        return x, y

    @staticmethod
    def group_split_clip(
        clip_keys: List[Tuple[int, int]],
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ):
        random.seed(seed)
        random.shuffle(clip_keys)
        n = len(clip_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_keys = clip_keys[:n_train]
        val_keys = clip_keys[n_train : n_train + n_val]
        test_keys = clip_keys[n_train + n_val :]
        return train_keys, val_keys, test_keys


if __name__ == '__main__':
    input_size = [512, 512]
    heatmap_size = [512, 512]

    transform = A.ReplayCompose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.ColorJitter(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    dataset = SequenceKeypointDataset(
        annotation_file=r"data\ball\coco_annotations_globally_tracked.json",
        image_root=r"data\ball\images",
        T=4,
        input_size=input_size,
        heatmap_size=heatmap_size,
        transform=transform,
        split="train",
        use_group_split=True,
        input_type="stack",    # cat or stack
        output_type="all",   # all or last
        skip_frames_range=(1, 5)
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for frames, heatmaps, visibility in dataloader:
        print(f"frames shape: {frames.shape}")
        print(f"heatmaps shape: {heatmaps.shape}")
        print(f"visibility shape: {visibility.shape}")
        break

