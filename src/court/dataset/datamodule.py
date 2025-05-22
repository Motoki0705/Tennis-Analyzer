import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.court.arguments.prerare_teansforms import prepare_transform
from src.court.dataset.court_dataset import CourtDataset


class CourtDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Court Keypoint Detection.
    """

    def __init__(
        self,
        annotation_root: Union[str, Path] = r"data/",
        image_root: Union[str, Path] = r"data/images",
        batch_size: int = 8,
        num_workers: int = 8,
        input_size: Tuple[int, int] = (224, 224),
        heatmap_size: Tuple[int, int] = (224, 224),
        default_keypoints: int = 15,
        is_each_keypoint: bool = True,
        sigma: float = 2.0,
    ):
        """
        Args:
            annotation_root: Path to annotation files (train/val/test).
            image_root: Path to images.
            batch_size: Batch size.
            num_workers: Number of workers for DataLoader.
            input_size: Model input image size (H, W).
            heatmap_size: Output heatmap size (H, W).
            num_annotated_keypoints: Keypoints in original annotation (14).
            num_model_keypoints: Keypoints used in model (typically 15, with center point).
            is_each_keypoint: Whether to generate heatmap per keypoint.
            sigma: Gaussian sigma for heatmap.
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
        print("✅ CourtDataModule initialized.")

    def setup(self, stage: Optional[str] = None):
        print(f"📦 setup() called with stage={stage}")

        train_transform, val_test_transform = prepare_transform(self.input_size)
        print("✅ Transforms prepared.")

        # Prepare datasets as needed
        if stage in (None, "fit"):
            train_ann = os.path.join(self.annotation_root, "converted_train.json")
            val_ann = os.path.join(self.annotation_root, "converted_val.json")
            if not os.path.exists(train_ann):
                raise FileNotFoundError(f"File not found: {train_ann}")
            if not os.path.exists(val_ann):
                raise FileNotFoundError(f"File not found: {val_ann}")
            self.train_dataset = self._prepare_dataset(
                annotation_path=train_ann, transform=train_transform
            )
            print("✅ Train dataset ready.")
            self.val_dataset = self._prepare_dataset(
                annotation_path=val_ann, transform=val_test_transform
            )
            print("✅ Validation dataset ready.")

        if stage in (None, "test"):
            test_ann = os.path.join(self.annotation_root, "test.json")
            if not os.path.exists(test_ann):
                raise FileNotFoundError(f"File not found: {test_ann}")
            self.test_dataset = self._prepare_dataset(
                annotation_path=test_ann, transform=val_test_transform
            )
            print("✅ Test dataset ready.")

    def _prepare_dataset(self, annotation_path, transform):
        print(f"🔧 Creating dataset from {annotation_path}")
        return CourtDataset(
            annotation_path=annotation_path,
            image_root=self.image_root,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            default_num_keypoints=self.default_keypoints,
            transform=transform,
            sigma=self.sigma,
            is_each_keypoint_heatmap=self.is_each_keypoint,
        )

    def train_dataloader(self):
        print("📤 Creating train DataLoader...")
        return self._prepare_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        print("📤 Creating validation DataLoader...")
        return self._prepare_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        print("📤 Creating test DataLoader...")
        return self._prepare_dataloader(self.test_dataset, shuffle=False)

    def _prepare_dataloader(self, dataset, shuffle: bool):
        print(
            f"🚚 DataLoader prepared with batch_size={self.batch_size}, shuffle={shuffle}"
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    dm = CourtDataModule(
        annotation_root="data/",
        image_root="data/images",
        batch_size=4,
        input_size=(256, 256),
        heatmap_size=(64, 64),
        num_annotated_keypoints=14,
        num_model_keypoints=15,
        is_each_keypoint=True,
        sigma=2.0,
    )
    dm.setup(stage="fit")
    for batch in dm.val_dataloader():
        # CourtDatasetの__getitem__が (image, heatmap, meta_info) を返すなら以下の通り
        image, heatmap, meta_info = batch
        print(image.shape, heatmap.shape)
        # 期待: torch.Size([4, 3, 256, 256]), torch.Size([4, 15, 64, 64])
        break
