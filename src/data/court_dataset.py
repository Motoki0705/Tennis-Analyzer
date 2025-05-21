from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
import json # For loading annotations
import os

from .base_dataset import BaseDataset
# It seems CourtDataset had its own heatmap generation logic, so we might need to import/reimplement that if not moving to a common util
# from src.court.utils.visualize_dataset import visualize_overlay # This was for visualization, maybe not needed directly in dataset

class CourtDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        # Extract court-specific parameters before calling super().__init__
        # These will be used in _load_annotations or __getitem__
        self.input_size_court = tuple(kwargs.pop('input_size_court', (360, 640))) # Example default
        self.heatmap_size_court = tuple(kwargs.pop('heatmap_size_court', (360, 640)))
        self.default_num_keypoints = kwargs.pop('default_num_keypoints', 15)
        self.sigma = kwargs.pop('sigma', 2)
        self.is_each_keypoint_heatmap = kwargs.pop('is_each_keypoint_heatmap', True)
        
        super().__init__(*args, **kwargs)
        # Post-super init if needed, e.g., validating specific kwargs

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        COCO形式のcourtアノテーションファイルを読み込み、画像ごとにアノテーションをまとめたリストを返す。
        """
        print(f"Loading court annotations from: {self.annotation_file}")
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
        images = coco["images"]
        anns = coco["annotations"]
        # 画像IDごとにアノテーションをまとめる
        anns_by_image = {}
        for ann in anns:
            img_id = ann["image_id"]
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)
        # 画像ごとにまとめたリストを作成
        all_samples = []
        for img in images:
            img_id = img["id"]
            sample = {
                "image_id": img_id,
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "annotations": anns_by_image.get(img_id, [])
            }
            all_samples.append(sample)
        print(f"CourtDataset: Found {len(all_samples)} images with annotations.")
        return all_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        Get a single court sample (image and corresponding keypoint heatmaps).
        """
        sample_data = self.samples[idx]
        anns = sample_data.get("annotations", [])
        if not anns or "keypoints" not in anns[0]:
            print(f"[court] No keypoints found for idx={idx}, file={sample_data.get('file_name')}")
            return torch.empty(0), torch.empty(0)
        keypoints = anns[0]["keypoints"]
        keypoints_raw = torch.tensor(keypoints, dtype=torch.float32).view(-1, 3)
        num_kps_in_sample = keypoints_raw.size(0)

        # 画像読み込み
        image_path = os.path.join(self.image_root, sample_data["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if num_kps_in_sample != self.default_num_keypoints:
            # This check was in the original, decide if it's still needed or how to handle
            # It might indicate an issue with the annotation file or expected format
            print(f"Warning: Keypoint count mismatch for sample {idx}. Expected {self.default_num_keypoints}, got {num_kps_in_sample}")
            # Decide on error handling: skip sample, pad keypoints, or raise error
            # For now, let's try to proceed if possible, or return a dummy if severe
            # return torch.empty(0), torch.empty(0) # Example of skipping

        # Prepare target for _apply_transform (it expects a dict with 'keypoints')
        target_for_transform = {"keypoints": keypoints_raw.tolist()} # Convert to list of lists for albumentations
        
        # Apply transformations
        # self.transform should be an Albumentations Compose object
        # The _apply_transform in BaseDataset handles the basic call
        # but keypoint specific processing like visibility might need care
        augmented = self._apply_transform(image_np, target=target_for_transform)
        
        image_tensor = augmented["image"] # This should be a tensor if ToTensorV2 is in transform
        # Albumentations returns keypoints as a list of [x,y] lists.
        # We need to re-associate visibility if it was stripped or handle it.
        # For simplicity, assuming visibility is maintained or handled by transform or later steps.
        # If transform changed number of keypoints (e.g. remove_invisible=True), this needs care.
        augmented_keypoints_list = augmented.get("keypoints", []) 
        
        # Convert augmented keypoints back to tensor, ensure correct shape
        if not augmented_keypoints_list or len(augmented_keypoints_list) != self.default_num_keypoints:
            # Handle cases where keypoints might have been filtered out by transforms
            # This might involve padding or different heatmap generation strategy
            # print(f"Warning: Keypoint count mismatch post-transform for sample {idx}. Expected {self.default_num_keypoints}, got {len(augmented_keypoints_list)}")
            # For now, create dummy keypoints if count is wrong to avoid crashing heatmap gen
            # This needs robust handling.
            arged_keypoints = torch.zeros((self.default_num_keypoints, 3), dtype=torch.float32)
            # Heuristically assign visibility based on whether any kps returned, crude.
            if augmented_keypoints_list: # if some kps, assume they are visible for now
                 for i, kp_xy in enumerate(augmented_keypoints_list):
                    if i < self.default_num_keypoints:
                        arged_keypoints[i, :2] = torch.tensor(kp_xy, dtype=torch.float32)
                        arged_keypoints[i, 2] = 1 # Assume visible if returned by aug
            # If NO keypoints returned by aug, all will be 0,0,0 (invisible)
        else:
            # Assuming keypoints from aug are just [x,y]. Need to add back visibility.
            # The original CourtDataset preserved visibility through transforms.
            # We'll assume for now that visibility from keypoints_raw can be used, 
            # or that transforms are configured not to drop keypoints unexpectedly.
            arged_keypoints = torch.zeros((self.default_num_keypoints, 3), dtype=torch.float32)
            for i in range(self.default_num_keypoints):
                if i < len(augmented_keypoints_list):
                    arged_keypoints[i, :2] = torch.tensor(augmented_keypoints_list[i], dtype=torch.float32)
                    arged_keypoints[i, 2] = keypoints_raw[i, 2] # Carry over original visibility for now
                else:
                    arged_keypoints[i, 2] = 0 # Padded keypoints are not visible

        # Original CourtDataset had filtering for outside screen and scaling
        # These should ideally be part of the albumentations pipeline if possible (e.g. custom transform or post-proc)
        # For now, reimplementing them here based on original logic.
        self._filter_outside_screen_keypoints(arged_keypoints, self.input_size_court)
        scaled_keypoints = self._scale_to_heatmap(arged_keypoints, self.input_size_court, self.heatmap_size_court)

        # Generate heatmaps
        if self.is_each_keypoint_heatmap:
            heatmaps = torch.zeros((self.default_num_keypoints, *self.heatmap_size_court), dtype=torch.float32)
            for i, (x, y, v) in enumerate(scaled_keypoints):
                if v > 0: # Only draw visible keypoints
                    self._draw_gaussian_on_heatmap(heatmaps[i], (x.item(), y.item()), sigma=self.sigma)
            target_heatmaps = heatmaps
        else:
            heatmap = torch.zeros((1, *self.heatmap_size_court), dtype=torch.float32)
            for i, (x, y, v) in enumerate(scaled_keypoints):
                if v > 0:
                    self._draw_gaussian_on_heatmap(heatmap[0], (x.item(), y.item()), sigma=self.sigma)
            target_heatmaps = heatmap

        return image_tensor, target_heatmaps

    # Helper methods from original CourtDataset (could be moved to a util if shared)
    def _filter_outside_screen_keypoints(self, keypoints: torch.Tensor, current_input_size: Tuple[int, int]):
        screen_h, screen_w = current_input_size
        for kp in keypoints:
            if not (0 <= kp[0] < screen_w and 0 <= kp[1] < screen_h):
                kp[2] = 0 # Set visibility to 0 if outside

    def _scale_to_heatmap(self, keypoints: torch.Tensor, current_input_size: Tuple[int, int], target_heatmap_size: Tuple[int, int]) -> torch.Tensor:
        scaled_kps = keypoints.clone()
        scale_x = target_heatmap_size[1] / current_input_size[1]
        scale_y = target_heatmap_size[0] / current_input_size[0]
        scaled_kps[:, 0] *= scale_x
        scaled_kps[:, 1] *= scale_y
        return scaled_kps

    def _draw_gaussian_on_heatmap(self, heatmap: torch.Tensor, center: Tuple[float, float], sigma: float):
        tmp_size = sigma * 3
        mu_x, mu_y = center
        # Determine dimensions, handling both [H, W] and [C, H, W] (assuming C=1 for 2D case)
        if heatmap.ndim == 2: # [H, W]
            h, w = heatmap.shape
            # Unsqueeze to [1, H, W] for consistent processing if needed by Gaussian generation
            # heatmap_proc = heatmap.unsqueeze(0) 
        elif heatmap.ndim == 3 and heatmap.shape[0] == 1: # [1, H, W]
            _, h, w = heatmap.shape
            # heatmap_proc = heatmap
        else:
            raise ValueError(f"Heatmap must be [H,W] or [1,H,W], got {heatmap.shape}")

        x = torch.arange(0, w, 1, dtype=torch.float32, device=heatmap.device)
        y = torch.arange(0, h, 1, dtype=torch.float32, device=heatmap.device)
        y = y.unsqueeze(1)

        if not (0 <= mu_x < w and 0 <= mu_y < h):
            return 

        g = torch.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))
        
        if heatmap.ndim == 2:
            heatmap_to_update = heatmap
        else: # ndim == 3, shape [1,H,W]
            heatmap_to_update = heatmap[0]

        heatmap_to_update[:] = torch.max(heatmap_to_update, g)
        # Modification is in-place on the passed tensor view.
        pass
