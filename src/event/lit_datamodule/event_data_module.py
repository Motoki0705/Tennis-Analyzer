from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.event.dataset.event_dataset import EventDataset


class EventDataModule(pl.LightningDataModule):
    """
    イベント検出用のDataModule。
    時系列でのイベントステータス予測を行うためのデータ管理モジュール。
    """

    def __init__(
        self,
        annotation_file: Union[
            str, Path
        ] = r"datasets/event/coco_annotations_ball_pose_court_event_status.json",
        T: int = 300,  # 時系列の長さ（フレーム数）
        batch_size: int = 32,
        num_workers: int = 8,
        skip_frames_range: Tuple[int, int] = (1, 5),
        output_type: str = "all",  # "all" or "last"
    ):
        super().__init__()
        
        self.annotation_file = annotation_file
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skip_frames_range = skip_frames_range
        self.output_type = output_type
        
        # データセット分析後に設定される値
        self.feature_dims = {
            "ball": 3,
            "player_bbox": 5,
            "player_pose": None,  # セットアップ時に計算
            "court": 31,
        }
        self.max_players = None  # セットアップ時に計算
        
        # save_hyperparameters() は __init__ の最後に呼び出す
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # fitフェーズ
        if stage in (None, "fit"):
            self.train_dataset = self._prepare_dataset("train")
            self.val_dataset = self._prepare_dataset("val")
            
            # 特徴量の次元を計算（最初のバッチから）
            batch = next(iter(self._make_dataloader(self.train_dataset, shuffle=False)))
            ball_features, player_bbox_features, player_pose_features, court_features, _, _ = batch
            
            # 各特徴の次元を保存
            self.feature_dims["ball"] = ball_features.shape[-1]
            self.feature_dims["player_bbox"] = player_bbox_features.shape[-1]
            self.feature_dims["player_pose"] = player_pose_features.shape[-1]
            self.feature_dims["court"] = court_features.shape[-1]
            
            # 最大プレイヤー数を取得
            self.max_players = player_bbox_features.shape[-2]
            
            print(f"特徴次元: {self.feature_dims}")
            print(f"最大プレイヤー数: {self.max_players}")

        # testフェーズ
        if stage in (None, "test"):
            self.test_dataset = self._prepare_dataset("test")
            
            # 特徴量の次元と最大プレイヤー数がまだ計算されていない場合
            if self.max_players is None:
                batch = next(iter(self._make_dataloader(self.test_dataset, shuffle=False)))
                ball_features, player_bbox_features, player_pose_features, court_features, _, _ = batch
                
                self.feature_dims["ball"] = ball_features.shape[-1]
                self.feature_dims["player_bbox"] = player_bbox_features.shape[-1]
                self.feature_dims["player_pose"] = player_pose_features.shape[-1]
                self.feature_dims["court"] = court_features.shape[-1]
                
                self.max_players = player_bbox_features.shape[-2]
                
                print(f"特徴次元: {self.feature_dims}")
                print(f"最大プレイヤー数: {self.max_players}")

    def _prepare_dataset(self, split: str):
        # イベント検出用データセット
        return EventDataset(
            annotation_file=self.annotation_file,
            T=self.T,
            split=split,
            skip_frames_range=self.skip_frames_range,
            output_type=self.output_type,
        )

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
            collate_fn=self.collate_fn,
        )
    
    def collate_fn(self, batch):
        """
        異なるプレイヤー数のサンプルをバッチ処理するためのcollate関数
        バッチ内の最大プレイヤー数に合わせてパディングを行います
        """
        ball_features = []
        player_bbox_features = []
        player_pose_features = []
        court_features = []
        targets = []
        image_infos = []
        
        # 各サンプルのデータを取得
        for sample in batch:
            ball_feat, player_bbox_feat, player_pose_feat, court_feat, target, image_info = sample
            
            ball_features.append(ball_feat)
            court_features.append(court_feat)
            targets.append(target)
            image_infos.append(image_info)
            
            # プレイヤー特徴のサイズを保存
            player_bbox_features.append(player_bbox_feat)
            player_pose_features.append(player_pose_feat)
        
        # バッチ内の最大プレイヤー数を計算
        max_players_in_batch = max([feat.shape[1] for feat in player_bbox_features])
        
        # プレイヤー特徴を最大プレイヤー数にパディング
        padded_bbox_features = []
        padded_pose_features = []
        
        for bbox_feat, pose_feat in zip(player_bbox_features, player_pose_features):
            T, curr_players, bbox_dim = bbox_feat.shape
            _, _, pose_dim = pose_feat.shape
            
            if curr_players < max_players_in_batch:
                # BBox特徴のパディング
                bbox_padding = torch.zeros((T, max_players_in_batch - curr_players, bbox_dim), 
                                          dtype=bbox_feat.dtype, device=bbox_feat.device)
                padded_bbox = torch.cat([bbox_feat, bbox_padding], dim=1)
                
                # ポーズ特徴のパディング
                pose_padding = torch.zeros((T, max_players_in_batch - curr_players, pose_dim), 
                                         dtype=pose_feat.dtype, device=pose_feat.device)
                padded_pose = torch.cat([pose_feat, pose_padding], dim=1)
            else:
                padded_bbox = bbox_feat
                padded_pose = pose_feat
            
            padded_bbox_features.append(padded_bbox)
            padded_pose_features.append(padded_pose)
        
        # すべての特徴をスタック
        ball_tensor = torch.stack(ball_features)
        player_bbox_tensor = torch.stack(padded_bbox_features)
        player_pose_tensor = torch.stack(padded_pose_features)
        court_tensor = torch.stack(court_features)
        
        # ターゲットの処理
        if isinstance(targets[0], torch.Tensor) and targets[0].dim() > 0:
            # バッチ次元が必要な場合（例："all"モード）
            target_tensor = torch.stack(targets)
        else:
            # スカラーの場合（例："last"モード）
            target_tensor = torch.tensor(targets, dtype=torch.long)
        
        return ball_tensor, player_bbox_tensor, player_pose_tensor, court_tensor, target_tensor, image_infos
        
    def get_feature_dims(self) -> Dict[str, int]:
        """モデルに必要な各特徴量の次元を返します"""
        if None in self.feature_dims.values() or self.max_players is None:
            raise ValueError("setup()が呼ばれる前にget_feature_dims()が呼ばれました。")
        return self.feature_dims
    
    def get_max_players(self) -> int:
        """データセット内の最大プレイヤー数を返します"""
        if self.max_players is None:
            raise ValueError("setup()が呼ばれる前にget_max_players()が呼ばれました。")
        return self.max_players 