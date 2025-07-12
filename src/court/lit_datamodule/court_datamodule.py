# court_datamodule.py

import os
from pathlib import Path
from typing import Optional, Tuple, Union, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.court.dataset.court_dataset import CourtDataset


class CourtDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Court Keypoint Detection.
    Handles loading and preprocessing of court keypoint detection datasets.
    """

    def __init__(
        self,
        annotation_root: Union[str, Path] = r"datasets/",
        image_root: Union[str, Path] = r"datasets/images",
        batch_size: int = 8,
        num_workers: int = 8,
        input_size: Tuple[int, int] = (224, 224),
        heatmap_size: Tuple[int, int] = (224, 224),
        default_keypoints: int = 15,
        is_each_keypoint: bool = True,
        sigma: float = 3.0,
        train_transform: Optional[Callable] = None,
        val_test_transform: Optional[Callable] = None,
        # --- ここからが追加箇所 ---
        use_peak_valley_heatmaps: bool = False, # ★新しい引数★
        # --- ここまでが追加箇所 ---
    ):
        """
        Initialize the CourtDataModule.
        """
        super().__init__()
        
        # self.save_hyperparameters() を使うと、全ての引数が自動的に保存・代入される
        self.save_hyperparameters()

    def prepare_data(self):
        """
        Prepare data. Called only once and on 1 GPU.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.
        """
        if stage in (None, "fit"):
            train_ann = os.path.join(self.hparams.annotation_root, "converted_train.json")
            val_ann = os.path.join(self.hparams.annotation_root, "converted_val.json")
            if not os.path.exists(train_ann):
                raise FileNotFoundError(f"File not found: {train_ann}")
            if not os.path.exists(val_ann):
                raise FileNotFoundError(f"File not found: {val_ann}")
            self.train_dataset = self._prepare_dataset(
                annotation_path=train_ann, transform=self.hparams.train_transform
            )
            self.val_dataset = self._prepare_dataset(
                annotation_path=val_ann, transform=self.hparams.val_test_transform
            )

        if stage in (None, "test", "predict"): # predictステージも追加
            test_ann = os.path.join(self.hparams.annotation_root, "test.json")
            if not os.path.exists(test_ann):
                raise FileNotFoundError(f"File not found: {test_ann}")
            self.test_dataset = self._prepare_dataset(
                annotation_path=test_ann, transform=self.hparams.val_test_transform
            )

    def _prepare_dataset(self, annotation_path, transform):
        """
        Create a dataset instance, passing all relevant hyperparameters.
        """
        return CourtDataset(
            annotation_path=annotation_path,
            image_root=self.hparams.image_root,
            input_size=self.hparams.input_size,
            heatmap_size=self.hparams.heatmap_size,
            default_num_keypoints=self.hparams.default_keypoints,
            transform=transform,
            sigma=self.hparams.sigma,
            is_each_keypoint=self.hparams.is_each_keypoint,
            # --- ここからが修正箇所 ---
            use_peak_valley_heatmaps=self.hparams.use_peak_valley_heatmaps, # ★引数を渡す★
            # --- ここまでが修正箇所 ---
        )

    def train_dataloader(self):
        """Create training dataloader."""
        return self._prepare_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """Create validation dataloader."""
        return self._prepare_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """Create test dataloader."""
        return self._prepare_dataloader(self.test_dataset, shuffle=False)

    def _prepare_dataloader(self, dataset, shuffle: bool):
        """Create a DataLoader for a dataset."""
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )