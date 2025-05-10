import os
import yaml
from typing import Tuple, Dict, Optional, Callable
import torch
from torch.utils.data import DataLoader
import torchvision
import transformers
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import RTDetrImageProcessorFast, RTDetrImageProcessor

from src.player.dataset.coco_dataset import CocoDetection
from src.player.arguments.prepare_transform import prepare_transform
from src.player.utils.visualize import visualize_datamodule

class CocoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_folder: str = None,
        annotation_file: str = None,
        cat_id_map: Dict[int, int] = {2: 0, 3: 1},
        use_original_path: str = None,
        processor: Callable = None,
        batch_size: int = 2,
        num_workers: int = 0,
        train_transform: Optional[Callable] = None,
        val_test_transform: Optional[Callable] = None,
    ):
        super().__init__()
        # cocodetection
        self.img_folder = img_folder
        self.annotation_file = annotation_file
        self.cat_id_map = cat_id_map
        self.use_original_path = use_original_path

        self.processor = processor

        # dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = self.prepare_dataset(
                split="train",
                transform=self.train_transform
            )
            self.val_dataset = self.prepare_dataset(
                split="val",
                transform=self.val_test_transform
            )

        elif stage == "test":
            self.test_dataset = self.prepare_dataset(
                split="test",
                transform=self.val_test_transform
            )

    def prepare_dataset(self, split, transform):
        return CocoDetection(
            img_folder=self.img_folder,
            annotation_file=self.annotation_file,
            cat_id_map=self.cat_id_map,
            use_original_path=self.use_original_path,
            split=split,
            transform=transform
        )

    def collate_fn(self, batch):
        if batch is None:
            print('batch is none')
            return None
        
        # None を除外
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None  # または raise でもよい
        
        # batch: list of (nparray, {"image_id":…, "annotations":[…]})
        images, targets = zip(*batch)

        # DETR の processor にまとめて投げる
        encoding = self.processor(
            images=list(images),
            annotations=list(targets),
            return_tensors="pt",
        )
        # encoding は dict {
        #   pixel_values: Tensor[B, 3, H, W],
        #   pixel_mask:   Tensor[B, H, W],
        #   labels:       Dict[str, Tensor],
        # }
        return encoding
    
    def train_dataloader(self):
        return self.prepare_dataloader(
            dataset=self.train_dataset,
            shuffle=True
        )

    def val_dataloader(self):
        return self.prepare_dataloader(
            dataset=self.val_dataset,
            shuffle=False
        )

    def test_dataloader(self):
        return self.prepare_dataloader(
            dataset=self.test_dataset,
            shuffle=False
        )
    
    def prepare_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
if __name__ == '__main__':
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    transform = prepare_transform()
    data_module = CocoDataModule(
        img_folder=r"data\ball\images",
        annotation_file=r"data\ball\coco_annotations_ball_ranged.json",
        cat_id_map={2: 0, 3: 1},
        use_original_path=True,
        processor=processor,
        batch_size=1,
        num_workers=0,
        train_transform=transform,
        val_test_transform=None
    )
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    for encoding in train_dataloader:
        print(encoding.keys())
        pixel_values = encoding["pixel_values"].squeeze()    # → Tensor[3, H, W]
        print(pixel_values.shape)
        boxes        = encoding["labels"][0]["boxes"].numpy()          # → array[N, 4], normalized [cx, cy, w, h]
        labels       = encoding["labels"][0]["class_labels"].numpy()      # → array[N]
        visualize_datamodule(img_tensor=pixel_values, boxes=boxes, labels=labels)
        break