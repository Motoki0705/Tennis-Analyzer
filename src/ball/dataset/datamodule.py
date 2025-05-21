from pathlib import Path
from typing import List, Optional, Tuple, Union

import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.ball.arguments.prepare_transform import prepare_transform
from src.ball.dataset.seq_coord_dataset import (
    SequenceCoordDataset,  # コーディネート回帰版
)
from src.ball.dataset.seq_key_dataset import SequenceKeypointDataset


class TennisBallDataModule(pl.LightningDataModule):
    """
    DataModule: ヒートマップ型 or 回帰型 を選択できます。
    dataset_type="heatmap"  → SequenceKeypointDataset
    dataset_type="coord"    → SequenceCoordDataset
    """

    def __init__(
        self,
        annotation_file: Union[
            str, Path
        ] = r"data/annotation_jsons/coco_annotations_globally_tracked.json",
        image_root: Union[str, Path] = r"data/images",
        T: int = 3,
        batch_size: int = 32,
        num_workers: int = 8,
        input_size: List[int] = [512, 512],
        heatmap_size: List[int] = [512, 512],
        skip_frames_range: Tuple[int, int] = (1, 5),
        input_type: str = "cat",
        output_type: str = "all",
        dataset_type: str = "heatmap",  # "heatmap" or "coord"
    ):
        super().__init__()
        assert dataset_type in {
            "heatmap",
            "coord",
        }, "`dataset_type` must be 'heatmap' or 'coord'"
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
        self.dataset_type = dataset_type

    def setup(self, stage: Optional[str] = None):
        # transformの準備
        train_transform, val_test_transform = prepare_transform(self.input_size)

        # fitフェーズ
        if stage in (None, "fit"):
            self.train_dataset = self._prepare_dataset("train", train_transform)
            self.val_dataset = self._prepare_dataset("val", val_test_transform)

        # testフェーズ
        if stage in (None, "test"):
            self.test_dataset = self._prepare_dataset("test", val_test_transform)

    def _prepare_dataset(self, split: str, transform: A.ReplayCompose):
        if self.dataset_type == "heatmap":
            # ヒートマップ学習用データセット
            return SequenceKeypointDataset(
                annotation_file=self.annotation_file,
                image_root=self.image_root,
                T=self.T,
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
                transform=transform,
                split=split,
                skip_frames_range=self.skip_frames_range,
                input_type=self.input_type,
                output_type=self.output_type,
            )
        else:
            # 座標回帰用データセット
            return SequenceCoordDataset(
                annotation_file=self.annotation_file,
                image_root=self.image_root,
                T=self.T,
                input_size=self.input_size,
                transform=transform,
                split=split,
                skip_frames_range=self.skip_frames_range,
                input_type=self.input_type,
                output_type=self.output_type,
            )

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, shuffle=False)

    def _make_dataloader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
