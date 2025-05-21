from typing import Any, Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
# import cv2 # Not needed if not generating masks with it for now

# Attempt to import COCO tools, handle if not available during refactoring
try:
    from pycocotools.coco import COCO
    # from pycocotools import mask as coco_mask_util # Not used if not generating masks
except ImportError:
    COCO = None
    # coco_mask_util = None
    print("Warning: pycocotools not found. PlayerDataset functionality related to COCO will be limited.")

from .base_dataset import BaseDataset
import albumentations as A # Ensure albumentations is imported

class PlayerDataset(BaseDataset):
    def __init__(
        self, 
        annotation_file: str, 
        image_root: str, 
        transform: Optional[A.Compose] = None, # Standard Albumentations Compose
        split: str = "train", 
        seed: int = 42, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.2,
        # Player-specific parameters (DETR-focused)
        num_keypoints: int = 17, # Still useful for filtering annotations, but not for heatmap
        # input_size is implicitly handled by Albumentations transforms if resize is part of it
        # generate_mask: bool = False, # Mask generation is out of scope for DETR target
        filter_min_keypoints: int = 1, # Min keypoints for an annotation to be considered valid
        filter_min_area: float = 0.0,    # Min area for an annotation
        return_masks: bool = False, # For segmentation masks if DETR model supports it
        **kwargs
    ):
        super().__init__(annotation_file, image_root, transform, split, seed, train_ratio, val_ratio, **kwargs)
        
        if COCO is None:
            raise ImportError("pycocotools is required for PlayerDataset but not found. Please install it.")

        self.coco = COCO(self.annotation_file.as_posix())
        
        # Player-specific attributes
        self.num_keypoints_for_filter = num_keypoints # Used if filtering by num_keypoints
        self.filter_min_keypoints = filter_min_keypoints
        self.filter_min_area = filter_min_area
        self.return_masks = return_masks # For DETR segmentation variants

        self.person_cat_id = self.coco.getCatIds(catNms=['person'])[0]
        # For DETR, we usually map COCO category IDs to a contiguous range [0, num_classes-1]
        # Assuming 'person' is the only category of interest, its mapped ID would be 0.
        self.coco_id_to_contiguous_id = {self.person_cat_id: 0} # Example: person maps to class 0
        # If other classes were present, this map would be more extensive.

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load annotations from COCO. Each returned item will correspond to a single image 
        and will contain all valid person annotations for that image.
        This is closer to standard COCO detection dataset practices.
        """
        print(f"Loading player (COCO) annotations from: {self.annotation_file} for PlayerDataset (DETR-style)")
        img_ids = self.coco.getImgIds()
        all_image_samples = []

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id], iscrowd=False)
            raw_annotations_for_img = self.coco.loadAnns(ann_ids)

            valid_annotations_for_image = []
            for ann in raw_annotations_for_img:
                # Apply filters (e.g., min keypoints, min area)
                # Note: original CocoDetection filtered for exactly 2 players per image later in split_ids.
                # Here, we filter individual annotations first.
                if ann.get('num_keypoints', 0) >= self.filter_min_keypoints and ann.get('area', 0) > self.filter_min_area:
                    valid_annotations_for_image.append(ann)
            
            if valid_annotations_for_image: # Only include images that have at least one valid person
                all_image_samples.append({
                    "image_id": img_id,
                    "file_name": img_info['file_name'],
                    "width": img_info['width'],
                    "height": img_info['height'],
                    "annotations": valid_annotations_for_image # List of person annotations for this image
                })
        
        if not all_image_samples:
            print(f"Warning: No images with valid person instances found in {self.annotation_file}.")
        else:
            print(f"PlayerDataset: Found {len(all_image_samples)} images with person instances.")
        return all_image_samples # BaseDataset will split these image-level samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_sample_info = self.samples[idx] # From BaseDataset, one image with its annotations
        
        image_np, _ = self._base_getitem_logic(idx, load_image_filename_key="file_name")
        # `_` is sample_data from BaseDataset, which is image_sample_info itself.

        annotations_for_image = image_sample_info["annotations"]
        
        # Prepare for Albumentations
        bboxes_xywh = [ann["bbox"] for ann in annotations_for_image]
        # Map COCO category_id (e.g., 1 for person) to contiguous IDs (e.g., 0 for person)
        class_labels = [self.coco_id_to_contiguous_id[ann["category_id"]] for ann in annotations_for_image]

        # Segmentation masks (optional, for DETR segmentation variants)
        masks_for_transform = []
        if self.return_masks:
            # This requires pycocotools.mask and careful handling of mask formats
            # For simplicity, this part is highly conceptual and needs robust implementation
            # if coco_mask_util:
            #     for ann in annotations_for_image:
            #         rle = self.coco.annToRLE(ann)
            #         mask = coco_mask_util.decode([rle])[:,:,0] # Get binary mask
            #         masks_for_transform.append(mask)
            pass # Placeholder for actual mask loading and transformation

        transform_input = {
            "image": image_np.copy(),
            "bboxes": bboxes_xywh,
            "category_id": class_labels # Albumentations uses 'category_id' for class labels with bboxes
        }
        if self.return_masks and masks_for_transform: # If actual masks were prepared
             transform_input["masks"] = masks_for_transform

        augmented = self.transform(**transform_input) if self.transform else transform_input
        
        image_tensor = augmented["image"] # Should be a tensor [C,H,W]
        transformed_bboxes_xywh = augmented.get("bboxes", [])
        transformed_class_labels = augmented.get("category_id", [])
        # transformed_masks = augmented.get("masks", []) # If masks were transformed

        # Convert to DETR target format
        # Boxes: [num_objects, 4] in (cx, cy, w, h) normalized format
        # Labels: [num_objects] class indices
        target_boxes = []
        target_labels = []
        
        img_h, img_w = image_tensor.shape[-2:] # Get H,W from the transformed image tensor

        for bbox, label in zip(transformed_bboxes_xywh, transformed_class_labels):
            x, y, w, h = bbox
            # Convert xywh to cxcywh normalized
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            target_boxes.append([cx, cy, norm_w, norm_h])
            target_labels.append(label)

        targets = {}
        if target_boxes:
            targets["boxes"] = torch.as_tensor(target_boxes, dtype=torch.float32)
            targets["labels"] = torch.as_tensor(target_labels, dtype=torch.int64)
        else: # No objects after transform, DETR expects empty tensors in specific format
            targets["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            targets["labels"] = torch.zeros(0, dtype=torch.int64)

        # Add other fields DETR might need, like 'image_id', 'orig_size', 'size'
        targets["image_id"] = torch.as_tensor([image_sample_info["image_id"]], dtype=torch.int64)
        original_img_h = image_sample_info["height"]
        original_img_w = image_sample_info["width"]
        targets["orig_size"] = torch.as_tensor([int(original_img_h), int(original_img_w)], dtype=torch.int64)
        targets["size"] = torch.as_tensor([int(img_h), int(img_w)], dtype=torch.int64)

        # if self.return_masks and transformed_masks:
        #     targets['masks'] = torch.stack([torch.from_numpy(m) for m in transformed_masks])
        
        return image_tensor, targets

    # No _draw_gaussian_on_heatmap needed for DETR-style dataset 