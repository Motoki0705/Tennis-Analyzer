import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Any, Dict

from src.data.ball_dataset import BallDataset
from src.data.court_dataset import CourtDataset
from src.data.player_dataset import PlayerDataset

class TennisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: str,  # "ball", "court", "player"
        dataset_kwargs: Optional[Dict[str, Any]] = None,  # 各Datasetの__init__に渡すパラメータ
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: Optional[Any] = None,
        val_test_transform: Optional[Any] = None,
        collate_fn: Optional[Any] = None,
    ):
        super().__init__()
        assert task in {"ball", "court", "player"}, f"Unknown task: {task}"
        self.task = task
        self.dataset_kwargs = dataset_kwargs or {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self._collate_fn = collate_fn

    def setup(self, stage: Optional[str] = None):
        if self.task == "ball":
            DatasetClass = BallDataset
        elif self.task == "court":
            DatasetClass = CourtDataset
        elif self.task == "player":
            DatasetClass = PlayerDataset
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if stage in (None, "fit"):
            self.train_dataset = DatasetClass(
                **self.dataset_kwargs,
                split="train",
                transform=self.train_transform,
            )
            self.val_dataset = DatasetClass(
                **self.dataset_kwargs,
                split="val",
                transform=self.val_test_transform,
            )
        if stage in (None, "test"):
            self.test_dataset = DatasetClass(
                **self.dataset_kwargs,
                split="test",
                transform=self.val_test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        ) 