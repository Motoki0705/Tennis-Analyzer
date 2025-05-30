import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.court.dataset.court_dataset import CourtDataset
from src.court.dataset.pseudo_labeled_court_dataset import PseudoLabeledCourtDataset


class SelfTrainingCourtDataModule(pl.LightningDataModule):
    """
    自己学習用のコート検出データモジュール。
    ラベル付きデータと未ラベルデータを組み合わせて自己学習を行います。
    """

    def __init__(
        self,
        annotation_root: str,
        image_root: str,
        unlabeled_annotation_file: str,
        unlabeled_image_root: str,
        pseudo_label_dir: str,
        input_size: Tuple[int, int] = (224, 224),
        heatmap_size: Tuple[int, int] = (224, 224),
        batch_size: int = 8,
        num_workers: int = 8,
        default_keypoints: int = 15,
        is_each_keypoint: bool = False,
        sigma: float = 2.0,
        pseudo_label_weight: float = 0.5,
        train_transform=None,
        val_test_transform=None,
    ):
        """
        初期化

        Parameters
        ----------
        annotation_root : ラベル付きデータのアノテーションルートディレクトリ
        image_root : ラベル付き画像のルートディレクトリ
        unlabeled_annotation_file : 未ラベルデータのアノテーションファイル
        unlabeled_image_root : 未ラベル画像のルートディレクトリ
        pseudo_label_dir : 擬似ラベルの保存ディレクトリ
        input_size : 入力画像サイズ (H, W)
        heatmap_size : ヒートマップサイズ (H, W)
        batch_size : バッチサイズ
        num_workers : データローダーのワーカー数
        default_keypoints : デフォルトのキーポイント数
        is_each_keypoint : 各キーポイントに個別のヒートマップを生成するかどうか
        sigma : ガウスカーネルの標準偏差
        pseudo_label_weight : 擬似ラベルの重み
        train_transform : 訓練用変換
        val_test_transform : 検証・テスト用変換
        """
        super().__init__()
        self.annotation_root = annotation_root
        self.image_root = image_root
        self.unlabeled_annotation_file = unlabeled_annotation_file
        self.unlabeled_image_root = unlabeled_image_root
        self.pseudo_label_dir = pseudo_label_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.default_keypoints = default_keypoints
        self.is_each_keypoint = is_each_keypoint
        self.sigma = sigma
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
        # トレーニング用アノテーションファイル
        train_annotation_path = os.path.join(self.annotation_root, "train.json")
        val_annotation_path = os.path.join(self.annotation_root, "val.json")
        test_annotation_path = os.path.join(self.annotation_root, "test.json")

        if self.train_transform is None:
            self.train_transform = A.Compose(
                [
                    A.Resize(height=self.input_size[0], width=self.input_size[1]),
                    A.Normalize(),
                    A.pytorch.ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

        if self.val_test_transform is None:
            self.val_test_transform = A.Compose(
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
            # 訓練データセット（擬似ラベル対応版）
            self.train_dataset = PseudoLabeledCourtDataset(
                annotation_path=train_annotation_path,
                image_root=self.image_root,
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
                default_num_keypoints=self.default_keypoints,
                transform=self.train_transform,
                sigma=self.sigma,
                is_each_keypoint=self.is_each_keypoint,
            )

            # 擬似ラベルがあれば追加
            if self.pseudo_label_path:
                self.train_dataset.add_pseudo_labels(
                    self.pseudo_label_path, weight=self.pseudo_label_weight
                )

            # 検証データセット（通常版）
            self.val_dataset = CourtDataset(
                annotation_path=val_annotation_path,
                image_root=self.image_root,
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
                default_num_keypoints=self.default_keypoints,
                transform=self.val_test_transform,
                sigma=self.sigma,
                is_each_keypoint=self.is_each_keypoint,
            )

        if stage == "test" or stage is None:
            # テストデータセット（通常版）
            self.test_dataset = CourtDataset(
                annotation_path=test_annotation_path,
                image_root=self.image_root,
                input_size=self.input_size,
                heatmap_size=self.heatmap_size,
                default_num_keypoints=self.default_keypoints,
                transform=self.val_test_transform,
                sigma=self.sigma,
                is_each_keypoint=self.is_each_keypoint,
            )
            
        # 未ラベルデータセット（予測用）
        self.unlabeled_dataset = CourtDataset(
            annotation_path=self.unlabeled_annotation_file,
            image_root=self.unlabeled_image_root,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            default_num_keypoints=self.default_keypoints,
            transform=self.val_test_transform,
            sigma=self.sigma,
            is_each_keypoint=self.is_each_keypoint,
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