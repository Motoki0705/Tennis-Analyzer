import os
from pathlib import Path
from typing import Optional, Tuple, Union, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.court.dataset.court_dataset import CourtDataset


class CourtDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Court Keypoint Detection.
    
    This module handles loading and preprocessing of court keypoint detection datasets,
    including training, validation, and test data preparation.
    
    Attributes:
        annotation_root: Path to annotation files
        image_root: Path to image files
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        input_size: Input image size (H, W)
        heatmap_size: Output heatmap size (H, W)
        default_keypoints: Number of keypoints to detect
        is_each_keypoint: Whether to generate separate heatmap per keypoint
        sigma: Gaussian sigma for heatmap generation
        train_transform: Transforms for training data
        val_test_transform: Transforms for validation and test data
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
    ):
        """
        Initialize the CourtDataModule.
        
        Args:
            annotation_root: Path to annotation files (train/val/test)
            image_root: Path to images
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            input_size: Model input image size (H, W)
            heatmap_size: Output heatmap size (H, W)
            default_keypoints: Number of keypoints to detect
            is_each_keypoint: Whether to generate heatmap per keypoint
            sigma: Gaussian sigma for heatmap
            train_transform: Transforms for training data
            val_test_transform: Transforms for validation and test data
        """
        super().__init__()
        self.annotation_root = str(annotation_root)
        self.image_root = str(image_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.default_keypoints = default_keypoints
        self.is_each_keypoint = is_each_keypoint
        self.sigma = sigma
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        
        # Save hyperparameters for logging
        self.save_hyperparameters()

    def prepare_data(self):
        """
        Prepare data for training.
        
        This method is called only once and on 1 GPU.
        Download data, tokenize, etc...
        """
        # Nothing to do for now
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Prepare datasets as needed
        if stage in (None, "fit"):
            train_ann = os.path.join(self.annotation_root, "converted_train.json")
            val_ann = os.path.join(self.annotation_root, "converted_val.json")
            if not os.path.exists(train_ann):
                raise FileNotFoundError(f"File not found: {train_ann}")
            if not os.path.exists(val_ann):
                raise FileNotFoundError(f"File not found: {val_ann}")
            self.train_dataset = self._prepare_dataset(
                annotation_path=train_ann, transform=self.train_transform
            )
            self.val_dataset = self._prepare_dataset(
                annotation_path=val_ann, transform=self.val_test_transform
            )

        if stage in (None, "test"):
            test_ann = os.path.join(self.annotation_root, "test.json")
            if not os.path.exists(test_ann):
                raise FileNotFoundError(f"File not found: {test_ann}")
            self.test_dataset = self._prepare_dataset(
                annotation_path=test_ann, transform=self.val_test_transform
            )

    def _prepare_dataset(self, annotation_path, transform):
        """
        Create a dataset instance.
        
        Args:
            annotation_path: Path to annotation file
            transform: Transforms to apply to the data
            
        Returns:
            CourtDataset instance
        """
        return CourtDataset(
            annotation_path=annotation_path,
            image_root=self.image_root,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            default_num_keypoints=self.default_keypoints,
            transform=transform,
            sigma=self.sigma,
            is_each_keypoint=self.is_each_keypoint,
        )

    def train_dataloader(self):
        """
        Create training dataloader.
        
        Returns:
            Training DataLoader
        """
        return self._prepare_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """
        Create validation dataloader.
        
        Returns:
            Validation DataLoader
        """
        return self._prepare_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """
        Create test dataloader.
        
        Returns:
            Test DataLoader
        """
        return self._prepare_dataloader(self.test_dataset, shuffle=False)

    def _prepare_dataloader(self, dataset, shuffle: bool):
        """
        Create a DataLoader for a dataset.
        
        Args:
            dataset: Dataset to create loader for
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        ) 