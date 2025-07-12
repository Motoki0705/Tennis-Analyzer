# court_dataset.py

import json
import os
from typing import Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

# 適切なパスに修正してください
from src.utils.heatmap.gaussian import draw_gaussian, draw_negative_gaussian
# from src.utils.visualization.court import visualize_court_overlay # visualize_court_overlayはメインの可視化スクリプトで使われるので、ここでは不要かもしれません

class CourtDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        image_root,
        input_size=(360, 640),
        heatmap_size=(360, 640),
        default_num_keypoints=15,
        transform=None,
        sigma=3,
        is_each_keypoint=True,
        use_peak_valley_heatmaps: bool = False,
    ):
        self.input_size = input_size
        self.image_root = image_root
        self.heatmap_size = heatmap_size
        self.default_num_keypoints = default_num_keypoints
        self.sigma = sigma
        self.is_each_keypoint = is_each_keypoint
        # --- ここからが追加箇所 ---
        self.use_peak_valley_heatmaps = use_peak_valley_heatmaps # ★引数をインスタンス変数に保存★
        # --- ここまでが追加箇所 ---

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transform(input_size)

        with open(annotation_path, "r") as f:
            self.data = json.load(f)

    def default_transform(self, input_size):
        return A.Compose(
            [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ... (前半の画像・キーポイント読み込み部分は変更なし) ...
        example = self.data[idx]
        file_name = example["file_name"]
        image_path = os.path.join(self.image_root, file_name)
        
        try:
            image = self.load_image(image_path)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Warning: Failed to load {image_path}. Reason: {e}. Returning dummy data.")
            image = None

        if image is None:
            # Fallback for missing or corrupted images
            dummy_image = torch.zeros((3, *self.input_size))
            num_kps = example.get("num_keypoints", self.default_num_keypoints)
            
            if self.is_each_keypoint:
                dummy_heatmaps = torch.zeros((num_kps, *self.heatmap_size))
            else:
                dummy_heatmaps = torch.zeros((1, *self.heatmap_size))
                
            dummy_keypoints = torch.zeros((num_kps, 3))
            return dummy_image, dummy_heatmaps, dummy_keypoints

        keypoints = torch.tensor(example["keypoints"], dtype=torch.float32).view(-1, 3)
        num_keypoints = example.get("num_keypoints", self.default_num_keypoints)

        image = np.array(image)
        augmented = self.transform(image=image, keypoints=keypoints)
        aug_image = augmented["image"]
        aug_keypoints = torch.tensor(augmented["keypoints"])

        self.filtering_outside_screen_keypoints(aug_keypoints, self.input_size)
        scaled_keypoints = self.scaling_to_heatmap(
            aug_keypoints, self.input_size, self.heatmap_size
        )

        if self.is_each_keypoint:
            heatmaps = torch.zeros(
                (num_keypoints, *self.heatmap_size), dtype=torch.float32
            )
            
            # --- ここからが修正箇所 ---
            # use_peak_valley_heatmaps が True の場合のみ、谷を生成する
            if self.use_peak_valley_heatmaps:
                for i in range(num_keypoints):
                    peak_x, peak_y, peak_v = scaled_keypoints[i]
                    if peak_v > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], (peak_x.item(), peak_y.item()), sigma=self.sigma
                        )
                    # 他のキーポイントを谷にする
                    for j in range(num_keypoints):
                        if i == j:
                            continue
                        valley_x, valley_y, valley_v = scaled_keypoints[j]
                        if valley_v > 0:
                            heatmaps[i] = draw_negative_gaussian(
                                heatmaps[i], (valley_x.item(), valley_y.item()), sigma=self.sigma
                            )
            else:
                # 従来通りのロジック（各ヒートマップに1つのピークのみ）
                for i, (x, y, v) in enumerate(scaled_keypoints):
                    if v > 0:
                        heatmaps[i] = draw_gaussian(heatmaps[i], (x.item(), y.item()), sigma=self.sigma)
            # --- ここまでが修正箇所 ---
            
            return aug_image, heatmaps, scaled_keypoints

        else: # is_each_keypoint が False の場合
            heatmap = torch.zeros((1, *self.heatmap_size), dtype=torch.float32)
            for i, (x, y, v) in enumerate(scaled_keypoints):
                if v > 0:
                    heatmap[0] = draw_gaussian(heatmap[0], (x.item(), y.item()), sigma=self.sigma)
            return aug_image, heatmap, scaled_keypoints
    
    # ... (load_image, filtering_outside_screen_keypoints, scaling_to_heatmap は変更なし) ...
    def load_image(self, image_path: str) -> Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except UnidentifiedImageError as e:
            print(f"Warning: Could not open image {image_path}. It might be corrupted. Error: {e}")
            return None

    def filtering_outside_screen_keypoints(self, keypoints: torch.Tensor, input_size: Tuple[int, int]) -> None:
        screen_h, screen_w = input_size
        for idx, keypoint in enumerate(keypoints):
            if not (0 <= keypoint[0] < screen_w and 0 <= keypoint[1] < screen_h):
                keypoint[2] = 0

    def scaling_to_heatmap(self, keypoints: torch.Tensor, input_size: Tuple[int, int], heatmap_size: Tuple[int, int]) -> torch.Tensor:
        scale_x = heatmap_size[1] / input_size[1]
        scale_y = heatmap_size[0] / input_size[0]
        scaled_keypoints = keypoints.clone()
        scaled_keypoints[:, 0] *= scale_x
        scaled_keypoints[:, 1] *= scale_y
        return scaled_keypoints