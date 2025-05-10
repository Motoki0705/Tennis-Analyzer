import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, Callable, Union, List
from pathlib import Path

from src.ball.dataset.seq_key_dataset import SequenceKeypointDataset
from src.ball.arguments.prepare_transform import prepare_transform

class TennisBallDataModule(pl.LightningDataModule):
    def __init__(
            self,
            annotation_file: Union[str, Path] = r"data\annotation_jsons\coco_annotations_globally_tracked.json",
            image_root: Union[str, Path] = r"data/images",
            T: int = 3,
            batch_size: int = 32,
            num_workers: int = 8,
            input_size: List[int] = [512, 512],
            heatmap_size: List[int] = [512, 512],
            skip_frames_range: List[int] = [1, 5],
            input_type: str = "cat",
            output_type: str = "all"
                    ):
        super().__init__()
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.skip_frames_range = skip_frames_range
        self.input_type = input_type
        self.output_type = output_type

    def setup(self, stage=None):
        # transformの初期化
        train_transform, val_test_transform = prepare_transform(self.input_size)

        # モードによってデータを切り替え
        if stage in (None, "fit"):        
            self.train_dataset = self._prepare_dataset(
                split="train",
                transform=train_transform
            )

            self.val_dataset = self._prepare_dataset(
                split="val",
                transform=val_test_transform
            )
        elif stage in (None, "test"):
            self.test_dataset = self._prepare_dataset(
                split="test",
                transform=val_test_transform
            )

    def _prepare_dataset(self, split, transform):
        return SequenceKeypointDataset(
            annotation_file=self.annotation_file,
            image_root=self.image_root,
            T=self.T,
            transform=transform,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            split=split,
            skip_frames_range=self.skip_frames_range,
            input_type=self.input_type,
            output_type=self.output_type
        )
    
    def train_dataloader(self):
        return self._prepare_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._prepare_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._prepare_dataloader(self.test_dataset, shuffle=False)
    
    def _prepare_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )