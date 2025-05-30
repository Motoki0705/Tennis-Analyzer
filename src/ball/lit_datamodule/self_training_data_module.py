import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.ball.dataset.pseudo_labeled_seq_dataset import PseudoLabeledSequenceDataset
from src.ball.dataset.seq_key_dataset import SequenceKeypointDataset
from src.ball.dataset.seq_coord_dataset import SequenceCoordDataset


class SelfTrainingBallDataModule(pl.LightningDataModule):
    """
    自己学習用のボール検出データモジュール。
    ラベル付きデータと未ラベルデータを組み合わせて自己学習を行います。
    """

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        unlabeled_annotation_file: str,
        unlabeled_image_root: str,
        pseudo_label_dir: str,
        T: int = 3,
        input_size: List[int] = [360, 640],
        heatmap_size: List[int] = [360, 640],
        batch_size: int = 32,
        num_workers: int = 8,
        skip_frames_range: List[int] = [1, 5],
        input_type: str = "cat",
        output_type: str = "last",
        dataset_type: str = "coord",
        pseudo_label_weight: float = 0.5,
        train_transform=None,
        val_test_transform=None,
    ):
        """
        初期化

        Parameters
        ----------
        annotation_file : ラベル付きデータのアノテーションファイル
        image_root : ラベル付き画像のルートディレクトリ
        unlabeled_annotation_file : 未ラベルデータのアノテーションファイル
        unlabeled_image_root : 未ラベル画像のルートディレクトリ
        pseudo_label_dir : 擬似ラベルの保存ディレクトリ
        T : 時系列フレーム数
        input_size : 入力画像サイズ [H, W]
        heatmap_size : ヒートマップサイズ [H, W]
        batch_size : バッチサイズ
        num_workers : データローダーのワーカー数
        skip_frames_range : スキップするフレーム範囲
        input_type : 入力タイプ ("cat" または "stack")
        output_type : 出力タイプ ("all" または "last")
        dataset_type : データセットタイプ ("heatmap" または "coord")
        pseudo_label_weight : 擬似ラベルの重み
        train_transform : 訓練用変換
        val_test_transform : 検証・テスト用変換
        """
        super().__init__()
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.unlabeled_annotation_file = unlabeled_annotation_file
        self.unlabeled_image_root = unlabeled_image_root
        self.pseudo_label_dir = pseudo_label_dir
        self.T = T
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skip_frames_range = skip_frames_range
        self.input_type = input_type
        self.output_type = output_type
        self.dataset_type = dataset_type
        self.pseudo_label_weight = pseudo_label_weight
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform

        # 擬似ラベルのパス
        self.pseudo_label_path = None

    def setup(self, stage: Optional[str] = None):
        """
        データセットの準備

        Parameters
        ----------
        stage : "fit", "validate", "test", "predict" のいずれか
        """
        if self.train_transform is None:
            self.train_transform = A.ReplayCompose(
                [
                    A.Resize(height=self.input_size[0], width=self.input_size[1]),
                    A.Normalize(),
                    A.pytorch.ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

        if self.val_test_transform is None:
            self.val_test_transform = A.ReplayCompose(
                [
                    A.Resize(height=self.input_size[0], width=self.input_size[1]),
                    A.Normalize(),
                    A.pytorch.ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

        # 最新の擬似ラベルファイルを検索
        pseudo_label_dir = Path(self.pseudo_label_dir)
        if pseudo_label_dir.exists():
            pseudo_label_files = list(pseudo_label_dir.glob("pseudo_labels_cycle_*.json"))
            if pseudo_label_files:
                # 最新のサイクルのファイルを使用
                self.pseudo_label_path = str(sorted(pseudo_label_files)[-1])
                print(f"最新の擬似ラベルファイルを使用: {self.pseudo_label_path}")

        # データセットの初期化
        if stage == "fit" or stage is None:
            # 訓練データセット
            if self.dataset_type == "heatmap":
                self.train_dataset = PseudoLabeledSequenceDataset(
                    annotation_file=self.annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.train_transform,
                    split="train",
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=tuple(self.skip_frames_range),
                )
            else:  # coord
                # 座標回帰用のデータセットを使用（擬似ラベル対応版を実装する必要がある）
                self.train_dataset = PseudoLabeledSequenceDataset(
                    annotation_file=self.annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.train_transform,
                    split="train",
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=tuple(self.skip_frames_range),
                )

            # 擬似ラベルがあれば追加
            if self.pseudo_label_path:
                self.train_dataset.add_pseudo_labels(
                    self.pseudo_label_path, weight=self.pseudo_label_weight
                )

            # 検証データセット
            if self.dataset_type == "heatmap":
                self.val_dataset = SequenceKeypointDataset(
                    annotation_file=self.annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.val_test_transform,
                    split="val",
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=(1, 1),  # 検証時はスキップなし
                )
            else:  # coord
                self.val_dataset = SequenceCoordDataset(
                    annotation_file=self.annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    transform=self.val_test_transform,
                    split="val",
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=(1, 1),  # 検証時はスキップなし
                )

        if stage == "test" or stage is None:
            # テストデータセット
            if self.dataset_type == "heatmap":
                self.test_dataset = SequenceKeypointDataset(
                    annotation_file=self.annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.val_test_transform,
                    split="test",
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=(1, 1),  # テスト時はスキップなし
                )
            else:  # coord
                self.test_dataset = SequenceCoordDataset(
                    annotation_file=self.annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    transform=self.val_test_transform,
                    split="test",
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=(1, 1),  # テスト時はスキップなし
                )
            
        # 未ラベルデータセット（予測用）
        if self.dataset_type == "heatmap":
            self.unlabeled_dataset = SequenceKeypointDataset(
                annotation_file=self.unlabeled_annotation_file,
                image_root=self.unlabeled_image_root,
                T=self.T,
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
                transform=self.val_test_transform,
                split="train",  # すべてのデータを使用
                use_group_split=False,  # 分割しない
                input_type=self.input_type,
                output_type=self.output_type,
                skip_frames_range=(1, 1),
            )
        else:  # coord
            self.unlabeled_dataset = SequenceCoordDataset(
                annotation_file=self.unlabeled_annotation_file,
                image_root=self.unlabeled_image_root,
                T=self.T,
                input_size=self.input_size,
                transform=self.val_test_transform,
                split="train",  # すべてのデータを使用
                use_group_split=False,  # 分割しない
                input_type=self.input_type,
                output_type=self.output_type,
                skip_frames_range=(1, 1),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def unlabeled_dataloader(self):
        return DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
    def get_unlabeled_dataset(self):
        """
        未ラベルデータセットを取得

        Returns
        -------
        unlabeled_dataset : 未ラベルデータセット
        """
        return self.unlabeled_dataset 