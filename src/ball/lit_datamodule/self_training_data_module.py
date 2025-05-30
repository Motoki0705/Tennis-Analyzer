"""
Self-training用のLightningDataModule実装
"""
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

import albumentations as A
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.ball.dataset.pseudo_labeled_seq_dataset import PseudoLabeledSequenceDataset
from src.ball.dataset.seq_key_dataset import SequenceKeypointDataset
from src.ball.dataset.seq_coord_dataset import SequenceCoordDataset

# ロガー設定
logger = logging.getLogger(__name__)


class SelfTrainingBallDataModule(pl.LightningDataModule):
    """
    Self-training用のボール検出データモジュール。
    
    ラベル付きデータとラベルなしデータを管理し、
    擬似ラベルの追加をサポートします。
    
    Attributes
    ----------
    labeled_annotation_file : str
        ラベル付きデータのアノテーションファイル
    unlabeled_annotation_file : str
        ラベルなしデータのアノテーションファイル
    image_root : str
        画像ファイルのルートディレクトリ
    T : int
        シーケンス長
    input_size : List[int]
        入力画像サイズ [H, W]
    heatmap_size : List[int]
        ヒートマップサイズ [H, W]
    batch_size : int
        バッチサイズ
    num_workers : int
        データローダーのワーカー数
    """
    
    def __init__(
        self,
        labeled_annotation_file: str,
        unlabeled_annotation_file: str,
        image_root: str,
        T: int,
        input_size: List[int],
        heatmap_size: List[int],
        train_transform: Optional[Callable] = None,
        val_test_transform: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        skip_frames_range: tuple = (1, 5),
        input_type: str = "cat",
        output_type: str = "last",
        dataset_type: str = "heatmap",
        use_pseudo_labels: bool = True,
        pseudo_label_path: Optional[str] = None,
        pseudo_label_weight: float = 0.5,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        初期化
        
        Parameters
        ----------
        labeled_annotation_file : str
            ラベル付きデータのCOCOフォーマットアノテーションファイル
        unlabeled_annotation_file : str
            ラベルなしデータのCOCOフォーマットアノテーションファイル
        image_root : str
            画像ファイルのルートディレクトリ
        T : int
            シーケンス長
        input_size : List[int]
            入力画像サイズ [H, W]
        heatmap_size : List[int]
            ヒートマップサイズ [H, W]
        train_transform : Callable, optional
            トレーニング用の画像変換
        val_test_transform : Callable, optional
            検証・テスト用の画像変換
        batch_size : int, optional
            バッチサイズ（デフォルト: 32）
        num_workers : int, optional
            データローダーのワーカー数（デフォルト: 4）
        skip_frames_range : tuple, optional
            スキップするフレーム範囲（デフォルト: (1, 5)）
        input_type : str, optional
            入力フォーマット "cat" / "stack"（デフォルト: "cat"）
        output_type : str, optional
            出力フォーマット "all" / "last"（デフォルト: "last"）
        dataset_type : str, optional
            データセットタイプ "heatmap" / "coord"（デフォルト: "heatmap"）
        use_pseudo_labels : bool, optional
            擬似ラベルを使用するか（デフォルト: True）
        pseudo_label_path : str, optional
            擬似ラベルファイルのパス
        pseudo_label_weight : float, optional
            擬似ラベルの重み（デフォルト: 0.5）
        train_ratio : float, optional
            トレーニングセット比率（デフォルト: 0.8）
        val_ratio : float, optional
            検証セット比率（デフォルト: 0.1）
        seed : int, optional
            乱数シード（デフォルト: 42）
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.labeled_annotation_file = labeled_annotation_file
        self.unlabeled_annotation_file = unlabeled_annotation_file
        self.image_root = image_root
        self.T = T
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skip_frames_range = skip_frames_range
        self.input_type = input_type
        self.output_type = output_type
        self.dataset_type = dataset_type
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_label_path = pseudo_label_path
        self.pseudo_label_weight = pseudo_label_weight
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
        # データセット
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.unlabeled_dataset = None
        
        # 検証用にパスが存在するかチェック
        self._validate_paths()

    def _validate_paths(self) -> None:
        """
        ファイルパスの妥当性を検証
        """
        if not Path(self.labeled_annotation_file).exists():
            logger.warning(f"Labeled annotation file not found: {self.labeled_annotation_file}")
        
        if not Path(self.unlabeled_annotation_file).exists():
            logger.warning(f"Unlabeled annotation file not found: {self.unlabeled_annotation_file}")
        
        if not Path(self.image_root).exists():
            logger.warning(f"Image root directory not found: {self.image_root}")

    def _create_default_transforms(self) -> None:
        """
        デフォルトの変換を作成
        """
        if self.train_transform is None:
            self.train_transform = A.ReplayCompose(
                [
                    A.Resize(height=self.input_size[0], width=self.input_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.1),
                    A.Normalize(),
                    A.pytorch.ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            logger.info("Created default training transforms")

        if self.val_test_transform is None:
            self.val_test_transform = A.ReplayCompose(
                [
                    A.Resize(height=self.input_size[0], width=self.input_size[1]),
                    A.Normalize(),
                    A.pytorch.ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
            logger.info("Created default validation/test transforms")

    def prepare_data(self) -> None:
        """
        データの準備（ダウンロードなど）
        """
        # 必要に応じてデータのダウンロードなどを実装
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        データセットのセットアップ
        
        Parameters
        ----------
        stage : str, optional
            ステージ（"fit", "validate", "test", "predict"）
        """
        # デフォルト変換の作成
        self._create_default_transforms()
        
        if stage in ("fit", "validate", None):
            try:
                # データセットクラスの選択
                if self.dataset_type == "heatmap":
                    dataset_class = PseudoLabeledSequenceDataset
                else:  # coord
                    # TODO: PseudoLabeledSequenceCoordDatasetを実装
                    logger.warning("Using PseudoLabeledSequenceDataset for coord type. Consider implementing PseudoLabeledSequenceCoordDataset.")
                    dataset_class = PseudoLabeledSequenceDataset
                
                # トレーニングデータセット
                self.train_dataset = dataset_class(
                    annotation_file=self.labeled_annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.train_transform,
                    split="train",
                    use_group_split=True,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                    seed=self.seed,
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=self.skip_frames_range,
                )
                
                # 擬似ラベルを追加
                if self.use_pseudo_labels and self.pseudo_label_path:
                    try:
                        self.train_dataset.add_pseudo_labels(
                            self.pseudo_label_path, 
                            weight=self.pseudo_label_weight
                        )
                        logger.info(f"Added pseudo labels from {self.pseudo_label_path}")
                    except Exception as e:
                        logger.error(f"Failed to add pseudo labels: {e}")
                
                # 検証データセット（擬似ラベルなし）
                if self.dataset_type == "heatmap":
                    val_dataset_class = SequenceKeypointDataset
                else:  # coord
                    val_dataset_class = SequenceCoordDataset
                
                self.val_dataset = val_dataset_class(
                    annotation_file=self.labeled_annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.val_test_transform,
                    split="val",
                    use_group_split=True,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                    seed=self.seed,
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=(1, 1),  # 検証時はスキップなし
                )
                
                # ラベルなしデータセット（推論用）
                self.unlabeled_dataset = val_dataset_class(
                    annotation_file=self.unlabeled_annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.val_test_transform,
                    split="train",  # 全データを使用
                    use_group_split=False,
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=(1, 1),  # 推論時はスキップなし
                )
                
                logger.info(f"Setup datasets - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Unlabeled: {len(self.unlabeled_dataset)}")
                
            except Exception as e:
                logger.error(f"Failed to setup datasets: {e}")
                raise
        
        if stage in ("test", None):
            try:
                # テストデータセット
                if self.dataset_type == "heatmap":
                    test_dataset_class = SequenceKeypointDataset
                else:  # coord
                    test_dataset_class = SequenceCoordDataset
                
                self.test_dataset = test_dataset_class(
                    annotation_file=self.labeled_annotation_file,
                    image_root=self.image_root,
                    T=self.T,
                    input_size=self.input_size,
                    heatmap_size=self.heatmap_size,
                    transform=self.val_test_transform,
                    split="test",
                    use_group_split=True,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                    seed=self.seed,
                    input_type=self.input_type,
                    output_type=self.output_type,
                    skip_frames_range=(1, 1),  # テスト時はスキップなし
                )
                
                logger.info(f"Setup test dataset - Test: {len(self.test_dataset)}")
                
            except Exception as e:
                logger.error(f"Failed to setup test dataset: {e}")
                raise

    def train_dataloader(self) -> DataLoader:
        """
        トレーニング用DataLoaderを返す
        
        Returns
        -------
        dataloader : DataLoader
            トレーニング用DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """
        検証用DataLoaderを返す
        
        Returns
        -------
        dataloader : DataLoader
            検証用DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """
        テスト用DataLoaderを返す
        
        Returns
        -------
        dataloader : DataLoader
            テスト用DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        推論用DataLoader（ラベルなしデータ）を返す
        
        Returns
        -------
        dataloader : DataLoader
            推論用DataLoader
        """
        return DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def update_pseudo_labels(self, pseudo_label_path: Union[str, Path]) -> None:
        """
        擬似ラベルを更新する
        
        Parameters
        ----------
        pseudo_label_path : Union[str, Path]
            新しい擬似ラベルファイルのパス
        """
        self.pseudo_label_path = str(pseudo_label_path)
        
        # トレーニングデータセットが既に作成されている場合は更新
        if self.train_dataset is not None and hasattr(self.train_dataset, 'add_pseudo_labels'):
            try:
                self.train_dataset.add_pseudo_labels(
                    self.pseudo_label_path,
                    weight=self.pseudo_label_weight
                )
                logger.info(f"Updated pseudo labels from {self.pseudo_label_path}")
            except Exception as e:
                logger.error(f"Failed to update pseudo labels: {e}")
                raise

    def get_dataset_for_self_training(self) -> Dict[str, any]:
        """
        Self-training用のデータセットを取得
        
        Returns
        -------
        datasets : Dict[str, any]
            ラベル付き、ラベルなし、検証用データセットの辞書
        """
        return {
            "labeled": self.train_dataset,
            "unlabeled": self.unlabeled_dataset,
            "val": self.val_dataset,
        }

    def get_unlabeled_dataset(self):
        """
        未ラベルデータセットを取得

        Returns
        -------
        unlabeled_dataset : Dataset
            未ラベルデータセット
        """
        return self.unlabeled_dataset 