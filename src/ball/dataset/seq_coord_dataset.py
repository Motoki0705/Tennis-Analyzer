import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SequenceCoordDataset(Dataset):
    """
    SequenceCoordDataset
    --------------------
    連続 T フレームを読み込み、各フレームの
        (x, y) ∈ [0,1]×[0,1]  正規化座標
    をターゲットとして返すデータセット。
    それ以外の引数・機能は元クラスと同じ。
    """

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        T: int,
        input_size: List[int],
        transform: A.ReplayCompose,
        split: str = "train",
        use_group_split: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        input_type: str = "stack",  # "cat" or "stack"
        output_type: str = "all",  # "all" or "last"
        skip_frames_range: Tuple[int, int] = (1, 5),
    ):
        assert input_type in {"cat", "stack"}
        assert output_type in {"all", "last"}

        self.T = T
        self.input_size = tuple(input_size)  # ★ ここが正規化の基準
        self.image_root = Path(image_root)
        self.transform = transform
        self.split = split
        self.skip_min, self.skip_max = skip_frames_range
        self.input_type = input_type
        self.output_type = output_type

        # --- JSON 読み込み ---
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self.images = {img["id"]: img for img in data["images"]}
        self.anns_by_image = {
            ann["image_id"]: ann
            for ann in data["annotations"]
            if ann["category_id"] == 1
        }

        # クリップ単位のグループ化
        clip_groups = defaultdict(list)
        for img in data["images"]:
            key = (img["game_id"], img["clip_id"])
            clip_groups[key].append(img["id"])

        # クリップ単位 Split
        clip_keys = list(clip_groups.keys())
        train_keys, val_keys, test_keys = (
            self.group_split_clip(clip_keys, train_ratio, val_ratio, seed)
            if use_group_split
            else self.group_split_clip(clip_keys, train_ratio, val_ratio, seed)
        )
        target_keys = {"train": train_keys, "val": val_keys, "test": test_keys}[split]

        # スライディングウィンドウ
        self.windows = []
        for key in target_keys:
            ids_sorted = sorted(clip_groups[key])
            if len(ids_sorted) < T:
                continue
            for start in range(0, len(ids_sorted) - T + 1):
                self.windows.append((key, start))

        self.clip_groups = clip_groups
        print(
            f"[{split.upper()}] clips: {len(target_keys)}, windows: {len(self.windows)}"
        )

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        clip_key, start = self.windows[idx]
        ids_sorted = sorted(self.clip_groups[clip_key])
        L = len(ids_sorted)

        # --- フレーム間 skip（訓練時のみ） ---
        if (
            self.T > 1
            and self.split == "train"
            and (self.skip_min, self.skip_max) != (1, 1)
        ):
            max_allowed = (L - 1 - start) // (self.T - 1)
            skip_upper = min(self.skip_max, max_allowed) if max_allowed > 0 else 1
            skip = random.randint(self.skip_min, skip_upper)
            frame_ids = [ids_sorted[start + i * skip] for i in range(self.T)]
        else:
            frame_ids = ids_sorted[start : start + self.T]

        frames, coords, visibilities, replay = [], [], [], None
        H_in, W_in = self.input_size  # ★ 正規化基準

        for i, img_id in enumerate(frame_ids):
            info = self.images[img_id]
            img_path = self.image_root / info["original_path"]
            img_np = np.array(Image.open(img_path).convert("RGB"))
            h_img, w_img = img_np.shape[:2]

            # --- キーポイント読込 ---
            ann = self.anns_by_image.get(img_id)
            if ann is not None:
                x, y, v = ann["keypoints"][:3]
                if not (0 <= x < w_img and 0 <= y < h_img):
                    x, y = self._clip_keypoints((x, y), (h_img, w_img))
                keypoints = [(x, y)]
                visibilities.append(v)
            else:
                keypoints = []
                visibilities.append(0)

            # --- Albumentations 変換 & 同一リプレイ ---
            if i == 0:
                out = self.transform(image=img_np, keypoints=keypoints)
                replay = out["replay"]
            else:
                out = A.ReplayCompose.replay(replay, image=img_np, keypoints=keypoints)

            img_aug = out["image"]  # tensor(C,H,W)
            kp_aug = out["keypoints"]

            # ★ ヒートマップ生成 → 座標正規化に変更
            if kp_aug:
                x_aug, y_aug = kp_aug[0]
                coords.append(
                    torch.tensor([x_aug / W_in, y_aug / H_in], dtype=torch.float32)
                )
            else:
                coords.append(torch.tensor([0.0, 0.0], dtype=torch.float32))

            frames.append(img_aug)

        # --- 入力フォーマット ---
        if self.input_type == "cat":
            input_tensor = torch.cat(frames, dim=0)  # [C*T, H, W]
        else:
            input_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]

        # --- 出力フォーマット ---
        if self.output_type == "all":
            coord_tensor = torch.stack(coords, dim=0)  # [T, 2]
            visibility_tensor = torch.tensor(visibilities, dtype=torch.float32)  # [T]
        else:  # "last"
            coord_tensor = coords[-1]  # [2]
            visibility_tensor = torch.tensor(
                visibilities[-1:], dtype=torch.float32
            )  # [1]

        return input_tensor, coord_tensor, visibility_tensor

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clip_keypoints(kp_xy: Tuple[float, float], img_hw: Tuple[int, int]):
        h, w = img_hw
        x, y = kp_xy
        return min(max(x, 0.0), w - 1e-3), min(max(y, 0.0), h - 1e-3)

    @staticmethod
    def group_split_clip(clip_keys, train_ratio, val_ratio, seed):
        random.seed(seed)
        random.shuffle(clip_keys)
        n = len(clip_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            clip_keys[:n_train],
            clip_keys[n_train : n_train + n_val],
            clip_keys[n_train + n_val :],
        )


import matplotlib.pyplot as plt


def visualize_coords_on_frames(
    frames: torch.Tensor, coords: torch.Tensor, input_type: str = "cat"
):
    """
    frames: Tensor of shape
        - [B, C*T, H, W] if input_type="cat"
        - [B, T, C, H, W] if input_type="stack"
    coords: Tensor of shape
        - [B, 2] for output_type="last"
        - [B, T, 2] for output_type="all"
    """
    frames = frames.cpu()
    coords = coords.cpu().numpy()
    B = frames.shape[0]

    # Restore individual frames
    if input_type == "cat":
        C = frames.shape[1] // 3  # C*T = channels, assume original C=3
        T = 3  # or infer T = frames.shape[1] // 3 if dynamic
        frames = frames.view(B, T, 3, frames.shape[2], frames.shape[3])
    else:
        # frames already [B, T, C, H, W]
        frames = frames

    for i in range(B):
        # Choose last frame for visualization
        img_tensor = frames[i, -1]  # shape [3, H, W]
        img = img_tensor.permute(1, 2, 0).numpy()

        # Get the coordinate for the last frame
        if coords.ndim == 2:
            x_norm, y_norm = coords[i]
        else:
            x_norm, y_norm = coords[i, -1]

        H, W, _ = img.shape
        x = x_norm * W
        y = y_norm * H

        plt.figure()
        plt.imshow(img)
        plt.scatter(x, y)
        plt.title(f"Sample {i} - Predicted Coord")
        plt.axis("off")
        plt.show()


# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    input_size = [360, 640]

    transform = A.ReplayCompose(
        [
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )

    dataset = SequenceCoordDataset(
        annotation_file=r"data\ball\coco_annotations_globally_tracked.json",
        image_root=r"data\ball\images",
        T=3,
        input_size=input_size,
        transform=transform,
        split="train",
        input_type="cat",  # cat / stack
        output_type="last",  # all / last
        skip_frames_range=(1, 5),
    )

    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    frames, coords, vis = next(iter(dl))
    print("frames:", frames.shape)  # cat: [B, C*T, H, W]
    print("coords:", coords.shape)  # last: [B, 2]
    print("vis   :", vis.shape)  # [B, 1]
    visualize_coords_on_frames(frames, coords, input_type="cat")
