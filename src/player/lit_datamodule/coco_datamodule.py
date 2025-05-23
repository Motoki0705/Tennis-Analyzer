from typing import Callable, Dict, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.player.dataset.coco_dataset import CocoDetection


class CocoDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for COCO-format object detection datasets.
    
    This module handles loading and preprocessing of COCO-format detection datasets,
    including training, validation, and test data preparation.
    
    Attributes:
        img_folder: Path to image folder
        annotation_file: Path to COCO annotation file
        cat_id_map: Mapping from COCO category IDs to model category IDs
        use_original_path: Whether to use original image paths from annotations
        processor: Image processor for DETR-based models
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train_transform: Transforms for training data
        val_test_transform: Transforms for validation and test data
    """

    def __init__(
        self,
        img_folder: str,
        annotation_file: str,
        processor: Callable,
        cat_id_map: Dict[int, int] = {2: 0},
        use_original_path: bool = False,
        batch_size: int = 2,
        num_workers: int = 0,
        train_transform: Optional[Callable] = None,
        val_test_transform: Optional[Callable] = None,
    ):
        """
        Initialize the CocoDataModule.
        
        Args:
            img_folder: Path to image folder
            annotation_file: Path to COCO annotation file
            processor: Image processor for DETR-based models
            cat_id_map: Mapping from COCO category IDs to model category IDs
            use_original_path: Whether to use original image paths from annotations
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            train_transform: Transforms for training data
            val_test_transform: Transforms for validation and test data
        """
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
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["processor", "train_transform", "val_test_transform"])

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
        if stage in (None, "fit"):
            self.train_dataset = self._prepare_dataset(
                split="train", transform=self.train_transform
            )
            self.val_dataset = self._prepare_dataset(
                split="val", transform=self.val_test_transform
            )

        if stage in (None, "test"):
            self.test_dataset = self._prepare_dataset(
                split="test", transform=self.val_test_transform
            )

    def _prepare_dataset(self, split: str, transform: Optional[Callable] = None):
        """
        Create a dataset instance.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            transform: Transforms to apply to the data
            
        Returns:
            CocoDetection instance
        """
        return CocoDetection(
            img_folder=self.img_folder,
            annotation_file=self.annotation_file,
            cat_id_map=self.cat_id_map,
            use_original_path=self.use_original_path,
            split=split,
            transform=transform,
        )

    def collate_fn(self, batch):
        """
        Collate function for batching.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Processed batch ready for model input
        """
        if batch is None:
            print("batch is none")
            return None

        # None を除外
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        # batch: list of (nparray, {"image_id":…, "annotations":[…]})
        images, targets = zip(*batch, strict=False)

        # DETR の processor にまとめて投げる
        encoding = self.processor(
            images=list(images),
            annotations=list(targets),
            return_tensors="pt",
        )
        return encoding

    def train_dataloader(self):
        """
        Create training dataloader.
        
        Returns:
            Training DataLoader
        """
        return self._prepare_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """
        Create validation dataloader.
        
        Returns:
            Validation DataLoader
        """
        return self._prepare_dataloader(dataset=self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """
        Create test dataloader.
        
        Returns:
            Test DataLoader
        """
        return self._prepare_dataloader(dataset=self.test_dataset, shuffle=False)

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
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        ) 