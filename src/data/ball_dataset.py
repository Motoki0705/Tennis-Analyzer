from typing import Any, Dict, List, Tuple, Optional
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import albumentations as A
# Ensure this import path is correct based on your project structure
# If utils is moved or heatmap.py is moved, adjust accordingly.
# Assuming src.ball.utils.heatmap is still the correct path relative to the project root
# For a cleaner structure, utils could be top-level src/utils/ or src/common/utils/
from src.ball.utils.heatmap import generate_gaussian_heatmap

from .base_dataset import BaseDataset


class BallDataset(BaseDataset):
    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        transform: Optional[A.ReplayCompose] = None, # Should be ReplayCompose for sequences
        split: str = "train",
        seed: int = 42,
        train_ratio: float = 0.7, # Default from BaseDataset
        val_ratio: float = 0.2,   # Default from BaseDataset
        # Ball-specific parameters
        T: int = 3,
        input_size_ball: Tuple[int, int] = (256, 256),
        heatmap_size_ball: Tuple[int, int] = (64, 64), # Example, adjust as needed
        input_type: str = "stack", # "cat" or "stack"
        output_type: str = "all",  # "all" or "last"
        skip_frames_range: Tuple[int, int] = (1, 1), # (min_skip, max_skip)
        use_group_split: bool = True, # Added from original for consistent splitting if desired
        **kwargs # To pass any other base class args or future args
    ):
        # Pop ball-specific args before calling super, so they don't go to BaseDataset's kwargs if not defined there
        self.T = T
        self.input_size_ball = input_size_ball
        self.heatmap_size_ball = heatmap_size_ball
        self.input_type = input_type
        assert input_type in {"cat", "stack"}, "input_type must be 'cat' or 'stack'"
        self.output_type = output_type
        assert output_type in {"all", "last"}, "output_type must be 'all' or 'last'"
        self.skip_min, self.skip_max = skip_frames_range
        self.use_group_split = use_group_split # Store this for _load_annotations

        # Call super().__init__ last, after setting ball-specific attributes
        # Pass through relevant args. If BaseDataset also handles train/val ratio, seed for splitting, they are passed.
        super().__init__(annotation_file, image_root, transform, split, seed, train_ratio, val_ratio, **kwargs)
        # self.samples will be populated by BaseDataset's _load_and_split_annotations


    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load ball annotations from COCO-like JSON, group by clips, and create T-frame windows.
        This method prepares a list of all possible samples (windows) BEFORE splitting.
        The actual train/val/test split of these samples is handled by BaseDataset.
        """
        print(f"Loading ball annotations from: {self.annotation_file} for BallDataset")
        with open(self.annotation_file, "r") as f:
            data = json.load(f)
        
        images_dict = {img["id"]: img for img in data["images"]}
        # Filter for ball annotations (category_id == 1, assuming from original)
        anns_by_image_id = {
            ann["image_id"]: ann
            for ann in data["annotations"]
            if ann["category_id"] == 1 
        }

        clip_groups = defaultdict(list)
        for img_id, img_data in images_dict.items():
            # Ensure 'game_id' and 'clip_id' are present in your COCO image entries
            game_id = img_data.get("game_id", "default_game") 
            clip_id = img_data.get("clip_id", "default_clip")
            # We need to store image_id for sorting and frame retrieval
            clip_groups[(game_id, clip_id)].append(img_id) 

        all_windows = []
        for clip_key, img_ids_in_clip in clip_groups.items():
            # Sort image_ids within each clip to ensure temporal order
            # This assumes image_ids or an associated frame number can be used for sorting
            # If file_names dictate order, that needs to be handled (e.g. sort by file_name)
            # For now, assume img_ids themselves can be sorted if they are sequential,
            # or that they are already in order from the JSON.
            # A robust way is to sort by a 'frame_number' field if present in img_data.
            # Example: ids_sorted = sorted(img_ids_in_clip, key=lambda img_id: images_dict[img_id].get('frame_number', img_id))
            ids_sorted = sorted(img_ids_in_clip) # Simplistic sort by id, ensure this is correct for your data

            if len(ids_sorted) < self.T:
                continue
            
            for start_idx in range(len(ids_sorted) - self.T + 1):
                # Each window is a dictionary that BaseDataset can split
                window_img_ids = ids_sorted[start_idx : start_idx + self.T]
                # Ensure all frames in window have annotations, or handle missing ones in __getitem__
                # For simplicity, assume __getitem__ will handle missing annotations for a frame.
                all_windows.append({
                    "clip_key": clip_key, # For potential group-based splitting if BaseDataset is adapted
                    "image_ids_in_window": window_img_ids,
                    "image_root": self.image_root, # For convenience in __getitem__
                    "images_meta": images_dict, # Pass all image metadata
                    "annotations_meta": anns_by_image_id # Pass all ball annotations
                })
        
        print(f"BallDataset: Found {len(all_windows)} total T-frame windows from {self.annotation_file}")
        return all_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single ball sample (sequence of frames and corresponding heatmaps).
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for BallDataset with {len(self.samples)} samples.")
        
        window_info = self.samples[idx]
        image_ids = window_info["image_ids_in_window"]
        images_meta = window_info["images_meta"]
        annotations_meta = window_info["annotations_meta"]

        # --- Debug: Print window info ---
        print(f"\n[BallDataset Debug __getitem__] Sample Index: {idx}")
        print(f"[BallDataset Debug __getitem__] Window Image IDs: {image_ids}")

        actual_frame_ids_to_load = image_ids
        if self.T > 1 and self.split == "train" and (self.skip_min != 1 or self.skip_max != 1):
            max_allowed = (len(image_ids) - 1) // (self.T - 1) if (self.T - 1) > 0 else 1
            skip_upper = min(self.skip_max, max_allowed) if max_allowed > 0 else 1
            current_skip = random.randint(self.skip_min, skip_upper)
            base_idx_in_original_clip = 0 # Assuming image_ids are indices from a larger clip, or just use them directly
            actual_frame_ids_to_load = []
            for i in range(self.T):
                # This logic assumes image_ids themselves are sequential or can be indexed directly if they are part of a pre-filtered list.
                # If image_ids are arbitrary and not sorted, this skip logic needs re-evaluation based on sorted frame numbers.
                # For now, assuming image_ids is already a sequence we can index into for skipping.
                idx_to_load = base_idx_in_original_clip + i * current_skip
                if idx_to_load < len(image_ids):
                    actual_frame_ids_to_load.append(image_ids[idx_to_load])
                else:
                    actual_frame_ids_to_load.append(image_ids[-1]) 

        print(f"[BallDataset Debug __getitem__] Actual Frame IDs to load: {actual_frame_ids_to_load}")

        frames_np = []
        keypoints_for_transform = [] 

        for i, img_id in enumerate(actual_frame_ids_to_load):
            print(f"[BallDataset Debug __getitem__] Processing frame {i+1}/{self.T}, Image ID: {img_id}")
            img_meta = images_meta.get(img_id)
            if not img_meta:
                print(f"Warning: Image metadata for ID {img_id} not found. Skipping frame.")
                continue 
            
            original_path_key = "original_path" if "original_path" in img_meta else "file_name"
            image_pil = self._load_image(img_meta[original_path_key])
            img_np = np.array(image_pil)
            h_img, w_img = img_np.shape[:2]

            ann = annotations_meta.get(img_id)
            current_frame_kps = []
            # --- Debug: Print original annotation ---
            if ann and "keypoints" in ann:
                print(f"[BallDataset Debug __getitem__] Frame {img_id} - Original Annotation Keypoints: {ann['keypoints']}")
                x, y, v = ann["keypoints"][:3]
                if not (0 <= x < w_img and 0 <= y < h_img):
                    x_orig, y_orig = x, y
                    x = min(max(x, 0.0), w_img - 1e-3)
                    y = min(max(y, 0.0), h_img - 1e-3)
                    print(f"[BallDataset Debug __getitem__] Frame {img_id} - Clipped keypoint ({x_orig},{y_orig}) to ({x},{y})")
                if v > 0: 
                    current_frame_kps.append([x, y])
            else:
                print(f"[BallDataset Debug __getitem__] Frame {img_id} - No annotation or no keypoints found.")

            keypoints_for_transform.append(current_frame_kps)
            frames_np.append(img_np)

        if len(frames_np) != self.T:
            if not frames_np:
                 print(f"[BallDataset Debug __getitem__] No frames loaded for sample {idx}. Returning empty.")
                 return torch.empty(0), {"heatmaps": torch.empty(0), "visibility": torch.empty(0)}
            print(f"Warning: Expected {self.T} frames, but loaded {len(frames_np)} for sample {idx}")
            # Pad if necessary, or handle error
            while len(frames_np) < self.T:
                frames_np.append(frames_np[-1].copy()) # Repeat last frame
                keypoints_for_transform.append(keypoints_for_transform[-1]) # Repeat last keypoints
                print(f"[BallDataset Debug __getitem__] Padded frame {len(frames_np)}/{self.T}")


        augmented_frames = []
        augmented_keypoints_seq = [] 
        replay_data = None

        for i in range(len(frames_np)):
            current_img_np = frames_np[i]
            current_kps_for_img = keypoints_for_transform[i]

            if i == 0:
                if self.transform:
                    out = self.transform(image=current_img_np, keypoints=current_kps_for_img)
                    replay_data = out.get("replay")
                else:
                    out = {"image": torch.from_numpy(current_img_np).permute(2,0,1).float() / 255.0, "keypoints": current_kps_for_img}
            else:
                if self.transform and replay_data:
                    out = A.ReplayCompose.replay(replay_data, image=current_img_np, keypoints=current_kps_for_img)
                else: 
                     out = {"image": torch.from_numpy(current_img_np).permute(2,0,1).float() / 255.0, "keypoints": current_kps_for_img}
            
            augmented_frames.append(out["image"]) 
            augmented_keypoints_seq.append(out.get("keypoints", [])) 
            # --- Debug: Print augmented keypoints ---
            print(f"[BallDataset Debug __getitem__] Frame {i+1} (ID: {actual_frame_ids_to_load[i] if i < len(actual_frame_ids_to_load) else 'Padded'}) - Augmented Keypoints: {out.get('keypoints', [])}")


        heatmaps_list = []
        visibility_list = []

        for i in range(len(augmented_frames)):
            kps_after_aug = augmented_keypoints_seq[i]
            hm = torch.zeros(self.heatmap_size_ball, dtype=torch.float32)
            vis = 0
            current_frame_id_for_log = actual_frame_ids_to_load[i] if i < len(actual_frame_ids_to_load) else 'Padded'
            if kps_after_aug: 
                ball_kp_xy = kps_after_aug[0] 
                raw_label_for_hm = {"keypoints": [ball_kp_xy[0], ball_kp_xy[1], 1]}
                current_hm = generate_gaussian_heatmap(
                    raw_label=raw_label_for_hm,
                    input_size=self.input_size_ball, 
                    output_size=self.heatmap_size_ball
                ) 
                if current_hm.ndim == 3 and current_hm.shape[0] == 1:
                    hm = current_hm.squeeze(0)
                elif current_hm.ndim == 2:
                    hm = current_hm
                else: 
                    print(f"Warning: Unexpected heatmap dimension {current_hm.shape}")
                vis = 1 
            
            # --- Debug: Print heatmap info ---
            print(f"[BallDataset Debug __getitem__] Frame {i+1} (ID: {current_frame_id_for_log}) - Heatmap Max: {hm.max().item()}, Visibility: {vis}")
            heatmaps_list.append(hm)
            visibility_list.append(vis)

        num_processed_frames = len(augmented_frames) # Should be self.T due to padding

        if self.input_type == "cat":
            if num_processed_frames == 0:
                 return torch.empty(0), {"heatmaps": torch.empty(0), "visibility": torch.empty(0)}
            input_tensor = torch.cat(augmented_frames, dim=0) 
        else: # "stack"
            if num_processed_frames == 0:
                 return torch.empty(0), {"heatmaps": torch.empty(0), "visibility": torch.empty(0)}
            input_tensor = torch.stack(augmented_frames, dim=0)


        if self.output_type == "all":
            if num_processed_frames == 0:
                 target_heatmaps = torch.empty(0)
                 target_visibility = torch.empty(0)
                 return input_tensor, {"heatmaps": target_heatmaps, "visibility": target_visibility}

            target_heatmaps = torch.stack(heatmaps_list, dim=0) 
            target_visibility = torch.tensor(visibility_list, dtype=torch.float32) 
        else: # "last"
            if num_processed_frames > 0:
                target_heatmaps = heatmaps_list[-1] 
                target_visibility = torch.tensor([visibility_list[-1]], dtype=torch.float32)
            else: 
                target_heatmaps = torch.zeros(self.heatmap_size_ball, dtype=torch.float32)
                target_visibility = torch.tensor([0], dtype=torch.float32)

        return input_tensor, {"heatmaps": target_heatmaps, "visibility": target_visibility}

    # _clip_keypoints and group_split_clip from original SequenceKeypointDataset are not
    # directly needed here if BaseDataset handles splitting and keypoint clipping is part of transform
    # or handled in __getitem__. The _clip_keypoints logic was simple and is now in __getitem__.
    # Group splitting logic from original is more complex with BaseDataset's current simple random split.
    # If group splitting is critical, BaseDataset._split_samples would need to be overridden or made more flexible.


    # Add any additional methods or overrides from BaseDataset as needed
    # For example, _load_image, _apply_transform, etc.
    # These should be implemented in the BaseDataset class or its subclasses
    # and called from here as needed.

    # _clip_keypoints and group_split_clip from original SequenceKeypointDataset are not
    # directly needed here if BaseDataset handles splitting and keypoint clipping is part of transform
    # or handled in __getitem__. The _clip_keypoints logic was simple and is now in __getitem__.
    # Group splitting logic from original is more complex with BaseDataset's current simple random split.
    # If group splitting is critical, BaseDataset._split_samples would need to be overridden or made more flexible. 