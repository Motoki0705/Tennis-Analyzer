from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        transform: Optional[A.BasicTransform] = None,
        split: str = "train",
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        # Add other common parameters like input_size, heatmap_size if truly universal
    ):
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.transform = transform
        self.split = split
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image root directory not found: {self.image_root}")

        self.samples = self._load_and_split_annotations()

    @abstractmethod
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load annotations from the annotation file.
        This method should be implemented by subclasses.
        It should return a list of all samples (e.g., dictionaries),
        before splitting into train/val/test.
        """
        pass

    def _split_samples(self, all_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits the loaded samples into train, validation, or test sets.
        This can be overridden by subclasses if more complex splitting logic
        (e.g., group-based splitting) is required.
        """
        n_total = len(all_samples)
        indices = list(range(n_total))
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        if self.split == "train":
            split_indices = indices[:n_train]
        elif self.split == "val":
            split_indices = indices[n_train : n_train + n_val]
        elif self.split == "test":
            split_indices = indices[n_train + n_val :]
        else:
            raise ValueError(f"Invalid split name: {self.split}. Must be 'train', 'val', or 'test'.")
        
        return [all_samples[i] for i in split_indices]

    def _load_and_split_annotations(self) -> List[Dict[str, Any]]:
        """Helper method to load and then split annotations."""
        all_samples = self._load_annotations()
        if not all_samples:
            print(f"Warning: No samples loaded from {self.annotation_file} for split {self.split}.")
            return []
        
        # Filter samples or assign to splits based on pre-existing split information if available
        # For example, if your COCO json or dataset structure already defines splits.
        # This is a placeholder for more sophisticated split handling.
        # If samples do not have pre-defined split info, use _split_samples.
        
        # Example: if samples have a 'split' key
        # if all(isinstance(s, dict) and 'split' in s for s in all_samples):
        #     return [s for s in all_samples if s['split'] == self.split]
        # else:
        #     # Fallback to random split if no pre-defined split info
        return self._split_samples(all_samples)


    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        Get a single sample (image, target) from the dataset.
        This method should be implemented by subclasses.
        """
        pass

    def _load_image(self, image_filename: str) -> Image.Image:
        """Loads an image from the specified filename relative to the image_root."""
        try:
            image_path = self.image_root / image_filename
            if not image_path.is_file():
                # Try to find the image in subdirectories if original_path is used
                # This is a common pattern seen in the existing datasets
                for parent_dir in self.image_root.iterdir():
                    if parent_dir.is_dir():
                        potential_path = parent_dir / image_filename
                        if potential_path.is_file():
                            image_path = potential_path
                            break
                if not image_path.is_file(): # check again after searching subdirs
                     raise FileNotFoundError(f"Image file not found: {image_path} (also checked common subdirs)")
            
            img = Image.open(image_path).convert("RGB")
            return img
        except FileNotFoundError as e:
            print(f"Error loading image {image_filename}: {e}")
            # Return a placeholder or raise error, depending on desired handling
            # For now, re-raising to make it explicit.
            raise 
        except Exception as e:
            print(f"An unexpected error occurred while loading image {image_filename}: {e}")
            raise

    def _apply_transform(self, image: np.ndarray, target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Applies the albumentations transform."""
        if self.transform:
            if target and "keypoints" in target and "bboxes" in target: # Example for COCO-like targets
                keypoints_for_transform = [kp[:2] for kp_list in target["keypoints"] for kp in kp_list] # Assuming [x,y,v] format
                visibility_for_transform = [kp[2] for kp_list in target["keypoints"] for kp in kp_list]
                return self.transform(
                    image=image, 
                    bboxes=target.get("bboxes", []), 
                    category_id=target.get("category_id", []),
                    keypoints=keypoints_for_transform # Pass only x,y to albumentations
                    # You might need to reconstruct visibility or handle it separately post-transform
                )
            elif target and "keypoints" in target: # For keypoint-only tasks
                 # Adjust based on actual keypoint format
                keypoints_for_transform = target["keypoints"] # Assuming this is already list of [x,y] or [x,y,v]
                if keypoints_for_transform and isinstance(keypoints_for_transform[0], list) and len(keypoints_for_transform[0]) == 3:
                     # If [x,y,v], extract x,y for albumentations
                     processed_kps = [[kp[0], kp[1]] for kp in keypoints_for_transform]
                     return self.transform(image=image, keypoints=processed_kps)
                return self.transform(image=image, keypoints=keypoints_for_transform)
            elif target and "bboxes" in target: # For bbox-only tasks
                return self.transform(image=image, bboxes=target["bboxes"], category_id=target.get("category_id", []))
            else: # Image-only transform
                return self.transform(image=image)
        # If no transform or no specific target keys, return image as is (or handle as needed)
        # This part might need refinement based on how image-only cases are handled
        return {"image": image}

    def _base_getitem_logic(self, idx: int, load_image_filename_key: str = "file_name") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Provides a base logic for __getitem__ that loads an image and its corresponding sample data.
        Subclasses should call this and then process the image and sample_data further.
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.samples)} samples.")
        
        sample_data = self.samples[idx].copy() # Use a copy to avoid modifying the original data
        image_filename = sample_data.get(load_image_filename_key)

        if not image_filename:
            raise ValueError(
                f"Sample at index {idx} does not have the required image filename key '{load_image_filename_key}'. "
                f"Sample data: {sample_data}"
            )

        image_pil = self._load_image(str(image_filename))
        image_np = np.array(image_pil)
        
        return image_np, sample_data

    # Placeholder for collate_fn if needed, often useful for sequence data or custom batching
    # @staticmethod
    # def collate_fn(batch):
    #     pass
