from pathlib import Path
from typing import List, Optional, Tuple, Union

import albumentations as A
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.ball.dataset.seq_coord_dataset import (
    SequenceCoordDataset,  # コーディネート回帰版
)
from src.ball.dataset.seq_key_dataset import SequenceKeypointDataset


class BallDataModule(pl.LightningDataModule):
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
        heatmap_size: List[int] = [64, 64],
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
        
        # save_hyperparameters() は __init__ の最後に呼び出すのが一般的
        self.save_hyperparameters()

    def prepare_data(self):
        """データの準備処理（ダウンロードなど）を行います。"""
        # テスト用なので何もしない
        pass

    def setup(self, stage: Optional[str] = None):
        """データセットの設定を行います。"""
        # テスト用の簡易実装
        if stage in (None, "fit"):
            self.train_dataset = self._create_dummy_dataset(10)
            self.val_dataset = self._create_dummy_dataset(5)
        if stage in (None, "test"):
            self.test_dataset = self._create_dummy_dataset(5)

    def _create_dummy_dataset(self, size: int = 10):
        """テスト用のダミーデータセットを作成します。"""
        # 入力フレーム
        if self.input_type == "cat":
            # [B, C*T, H, W]
            frames = torch.rand(size, 3 * self.T, self.input_size[0], self.input_size[1])
        else:  # stack
            # [B, T, C, H, W]
            frames = torch.rand(size, self.T, 3, self.input_size[0], self.input_size[1])
        
        # 出力ターゲット
        target_t = 1 if self.output_type == "last" else self.T
        
        if self.dataset_type == "heatmap":
            # [B, T_out, H_map, W_map]
            targets = torch.rand(size, target_t, self.heatmap_size[0], self.heatmap_size[1])
        else:  # coord
            # [B, T_out, 2]
            targets = torch.rand(size, target_t, 2)
        
        # 可視性フラグ
        visibility = torch.ones(size, target_t)
        
        # ダミーデータセット
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, frames, targets, visibility):
                self.frames = frames
                self.targets = targets
                self.visibility = visibility
                self.length = len(frames)
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                return self.frames[idx], self.targets[idx], self.visibility[idx]
        
        return DummyDataset(frames, targets, visibility)

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