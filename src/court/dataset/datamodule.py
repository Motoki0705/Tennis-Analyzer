import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, Union
from pathlib import Path

from src.court.dataset.court_dataset import CourtDataset
from src.court.arguments.prerare_teansforms import prepare_transform


class CourtDataModule(pl.LightningDataModule):
    def __init__(
        self,
        annotation_root: Union[str, Path] = r"data/",
        image_root: Union[str, Path] = r"data/images",
        batch_size: int = 8,
        num_workers: int = 8,
        input_size: Tuple[int, int] = (224, 224),
        heatmap_size: Tuple[int, int] = (224, 224),
        default_num_keypoints: int = 15,
        sigma: int = 2
    ):
        super().__init__()
        self.annotation_root = annotation_root
        self.image_root = image_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.default_num_keypoints = default_num_keypoints
        self.sigma = sigma
        print("✅ CourtDataModule initialized.")

    def setup(self, stage=None):
        print(f"📦 setup() called with stage={stage}")
        # Initialize transforms
        train_transform, val_test_transform = prepare_transform(self.input_size)
        print("✅ Transforms prepared.")

        # Setup datasets depending on the stage
        if stage in (None, "fit"):
            if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
                print("📂 Preparing train/val datasets...")
                train_annotation_path = os.path.join(self.annotation_root, "converted_train.json")
                val_annotation_path = os.path.join(self.annotation_root, "converted_val.json")
                if not os.path.exists(train_annotation_path):
                    raise FileNotFoundError(f"File not found: {train_annotation_path}")
                if not os.path.exists(val_annotation_path):
                    raise FileNotFoundError(f"File not found: {val_annotation_path}")
                
                self.train_dataset = self._prepare_dataset(
                    annotation_path=train_annotation_path,
                    transform=train_transform
                )
                print("✅ Train dataset ready.")

                self.val_dataset = self._prepare_dataset(
                    annotation_path=val_annotation_path,
                    transform=val_test_transform
                )
                print("✅ Validation dataset ready.")

        elif stage in (None, "test"):
            if not hasattr(self, "test_dataset"):
                print("📂 Preparing test dataset...")
                test_annotation_path = os.path.join(self.annotation_root, "test.json")
                if not os.path.exists(test_annotation_path):
                    raise FileNotFoundError(f"File not found: {test_annotation_path}")
                
                self.test_dataset = self._prepare_dataset(
                    annotation_path=test_annotation_path,
                    transform=val_test_transform
                )
                print("✅ Test dataset ready.")

    def _prepare_dataset(self, annotation_path, transform):
        print(f"🔧 Creating dataset from {annotation_path}")
        return CourtDataset(
            annotation_path=annotation_path,
            image_root=self.image_root,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            default_num_keypoints=self.default_num_keypoints,
            transform=transform,
            sigma=self.sigma,
            is_each_keypoint=False
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

    def _prepare_dataloader(self, dataset, shuffle):
        print(f"🚚 DataLoader prepared with batch_size={self.batch_size}, shuffle={shuffle}")
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False  # Prevent RAM buildup
        )

if __name__ == '__main__':
    dm = CourtDataModule(
        annotation_root="data/",
        batch_size=4,
        input_size=(256, 256),
        heatmap_size=(64, 64)
    )
    dm.setup(stage="fit")
    for image, heatmap in dm.val_dataloader():
        print(image.shape, heatmap.shape)  # 期待: torch.Size([4, 3, 256, 256]), torch.Size([4, 15, 64, 64])
