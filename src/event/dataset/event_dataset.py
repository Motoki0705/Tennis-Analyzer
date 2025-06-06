import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import albumentations as A
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class EventDataset(Dataset):
    """
    EventDataset
    ------------
    時系列データからボールのイベント状態を予測するためのデータセット。
    入力として:
    - ボールの中心座標とスコア
    - プレーヤーのbboxとスコア + poseキーポイント（すべてplayerカテゴリに含まれる）
    - コートのキーポイントとスコア
    を使用し、ボールカテゴリのevent_statusを出力します。
    
    返り値:
    - ball_features: ボールの特徴量 [T, 3] (正規化済みx, y座標 + スコア)
    - player_bbox_features: プレイヤーのBBox特徴量 [T, max_players, 5] (正規化済みx1, y1, x2, y2 + スコア)
    - player_pose_features: プレイヤーのポーズ特徴量 [T, max_players, num_keypoints*3] (正規化済み座標 + 可視性)
    - court_features: コートの特徴量 [T, num_keypoints*3] (正規化済み座標 + 可視性)
    - target: イベントステータス [T] または スカラー
    - image_info: 最後のフレームの画像情報
    """

    def __init__(
        self,
        annotation_file: str,
        T: int,
        split: str = "train",
        use_group_split: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        output_type: str = "all",  # "all" or "last"
        skip_frames_range: Tuple[int, int] = (1, 5),
    ):
        assert output_type in {"all", "last"}

        self.T = T
        self.split = split
        self.skip_min, self.skip_max = skip_frames_range
        self.output_type = output_type

        # --- JSON 読み込み ---
        with open(annotation_file, "r") as f:
            data = json.load(f)
        
        self.images = {img["id"]: img for img in data["images"]}
        
        # カテゴリごとのアノテーション分類
        self.ball_anns_by_image = {}
        self.player_anns_by_image = defaultdict(list)
        self.court_anns_by_image = {}
        self.event_status_by_image = {}
        
        # カテゴリIDの特定（データにより異なる可能性あり）
        category_ids = {cat["name"]: cat["id"] for cat in data["categories"]}
        ball_id = category_ids.get("ball", 1)
        player_id = category_ids.get("player", 2)
        court_id = category_ids.get("court", 3)
        
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            
            if cat_id == ball_id:
                self.ball_anns_by_image[img_id] = ann
                if "event_status" in ann:
                    self.event_status_by_image[img_id] = ann["event_status"]
            elif cat_id == player_id:
                self.player_anns_by_image[img_id].append(ann)
            elif cat_id == court_id:
                self.court_anns_by_image[img_id] = ann
        
        # クリップ単位のグループ化
        clip_groups = defaultdict(list)
        for img in data["images"]:
            key = (img["game_id"], img["clip_id"])
            clip_groups[key].append(img["id"])

        # クリップ単位 Split
        clip_keys = list(clip_groups.keys())
        train_keys, val_keys, test_keys = (
            self.group_split_clip(clip_keys, train_ratio, val_ratio, seed)
            if use_group_split
            else self.group_split_clip(clip_keys, train_ratio, val_ratio, seed)
        )
        target_keys = {"train": train_keys, "val": val_keys, "test": test_keys}[split]

        # スライディングウィンドウ
        self.windows = []
        for key in target_keys:
            ids_sorted = sorted(clip_groups[key])
            if len(ids_sorted) < T:
                continue
            for start in range(0, len(ids_sorted) - T + 1):
                self.windows.append((key, start))

        self.clip_groups = clip_groups
        print(
            f"[{split.upper()}] clips: {len(target_keys)}, windows: {len(self.windows)}"
        )

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        clip_key, start = self.windows[idx]
        ids_sorted = sorted(self.clip_groups[clip_key])
        L = len(ids_sorted)

        # --- フレーム間 skip（訓練時のみ） ---
        if (
            self.T > 1
            and self.split == "train"
            and (self.skip_min, self.skip_max) != (1, 1)
        ):
            max_allowed = (L - 1 - start) // (self.T - 1)
            skip_upper = min(self.skip_max, max_allowed) if max_allowed > 0 else 1
            skip = random.randint(self.skip_min, skip_upper)
            frame_ids = [ids_sorted[start + i * skip] for i in range(self.T)]
        else:
            frame_ids = ids_sorted[start : start + self.T]

        # 各フレームの特徴量を格納するリスト
        ball_features = []  # ボール座標+スコア: (x, y, score)
        player_bbox_features_list = []  # プレイヤーBBox+スコア: (x1, y1, x2, y2, score)
        player_pose_features_list = []  # プレイヤーポーズ: キーポイント
        court_features = []  # コートキーポイント+スコア: (kp_x, kp_y, ..., score)
        event_statuses = []  # イベントステータス: int
        
        # 全フレームで見つかったプレイヤーの最大数
        max_players_count = 0
        
        # 最初のパスでプレイヤー数の最大値を計算
        for img_id in frame_ids:
            player_anns = self.player_anns_by_image.get(img_id, [])
            max_players_count = max(max_players_count, len(player_anns))
        
        # 各フレームの特徴を抽出
        for i, img_id in enumerate(frame_ids):
            img_info = self.images[img_id]
            img_width = img_info["width"]
            img_height = img_info["height"]
            
            # 1. ボールの特徴抽出と正規化
            ball_ann = self.ball_anns_by_image.get(img_id)
            if ball_ann is not None:
                x, y, v = ball_ann["keypoints"][:3]
                # 座標を画像サイズで正規化
                x_norm = x / img_width
                y_norm = y / img_height
                
                # キーポイントスコアを使用
                ball_score = 0.0
                if "keypoints_scores" in ball_ann and ball_ann["keypoints_scores"]:
                    ball_score = float(ball_ann["keypoints_scores"][0])
                elif v > 0:  # キーポイントスコアがない場合は可視性に基づいてスコアを設定
                    ball_score = 1.0 if v == 2 else 0.5
                
                ball_feat = torch.tensor([x_norm, y_norm, ball_score], dtype=torch.float32)
                # イベントステータスの取得（マッピングなしでそのまま使用）
                event_status = self.event_status_by_image.get(img_id, 0)
            else:
                ball_feat = torch.zeros(3, dtype=torch.float32)
                event_status = 0  # デフォルト値
            
            ball_features.append(ball_feat)
            event_statuses.append(event_status)
            
            # 2. プレイヤーの特徴抽出（すべてのプレイヤー）と正規化
            player_anns = self.player_anns_by_image.get(img_id, [])
            frame_bbox_features = []
            frame_pose_features = []
            
            for p_ann in player_anns:
                # a. BBox特徴（正規化）
                x1, y1, w, h = p_ann["bbox"]
                x2, y2 = x1 + w, y1 + h
                # 座標を画像サイズで正規化
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height
                # スコアの取得
                player_score = float(p_ann.get("score", 1.0))
                bbox_feat = torch.tensor([x1_norm, y1_norm, x2_norm, y2_norm, player_score], dtype=torch.float32)
                frame_bbox_features.append(bbox_feat)
                
                # b. Poseキーポイント特徴（正規化）
                keypoints = p_ann["keypoints"]
                keypoints_scores = p_ann.get("keypoints_scores", [])
                
                keypoints_norm = []
                for i in range(0, len(keypoints), 3):
                    kp_x, kp_y, kp_v = keypoints[i:i+3]
                    # 座標を正規化
                    kp_x_norm = kp_x / img_width
                    kp_y_norm = kp_y / img_height
                    
                    # キーポイントスコアがある場合は可視性を更新
                    if keypoints_scores and i//3 < len(keypoints_scores):
                        kp_score = keypoints_scores[i//3]
                        # 可視性の設定
                        if kp_score >= 0.5:
                            kp_v = 2
                        elif kp_score > 0.01:
                            kp_v = 1
                        else:
                            kp_v = 0
                    
                    keypoints_norm.extend([kp_x_norm, kp_y_norm, kp_v])
                
                keypoints_tensor = torch.tensor(keypoints_norm, dtype=torch.float32)
                frame_pose_features.append(keypoints_tensor)
            
            player_bbox_features_list.append(frame_bbox_features)
            player_pose_features_list.append(frame_pose_features)
            
            # 3. コートの特徴抽出と正規化
            court_ann = self.court_anns_by_image.get(img_id)
            if court_ann is not None:
                keypoints = court_ann["keypoints"]
                keypoints_scores = court_ann.get("keypoints_scores", [])
                
                keypoints_norm = []
                for i in range(0, len(keypoints), 3):
                    kp_x, kp_y, kp_v = keypoints[i:i+3]
                    # 座標を正規化
                    kp_x_norm = kp_x / img_width
                    kp_y_norm = kp_y / img_height
                    
                    # キーポイントスコアがある場合は可視性を更新
                    if keypoints_scores and i//3 < len(keypoints_scores):
                        kp_score = keypoints_scores[i//3]
                        # 可視性の設定
                        if kp_score >= 0.6:  # コート用の閾値
                            kp_v = 2
                        elif kp_score > 0.01:
                            kp_v = 1
                        else:
                            kp_v = 0
                    
                    keypoints_norm.extend([kp_x_norm, kp_y_norm, kp_v])
                
                # すべてのキーポイントを平坦化
                keypoints_flat = torch.tensor(keypoints_norm, dtype=torch.float32)
                court_feat = keypoints_flat
            else:
                # コートキーポイント数 * 3
                court_keypoints_count = 15  # 一般的なコートキーポイント数
                court_feat = torch.zeros(court_keypoints_count * 3, dtype=torch.float32)
            
            court_features.append(court_feat)
        
        # 時系列特徴量をスタック
        ball_tensor = torch.stack(ball_features)  # [T, 3]
        court_tensor = torch.stack(court_features)  # [T, court_keypoints_count*3]
        
        # プレイヤーのBBox特徴とポーズ特徴を別々に処理
        # BBox特徴の次元: 5 (x1, y1, x2, y2, score)
        # ポーズ特徴の次元: キーポイント数 * 3 (例: 17 keypoints * 3 = 51)
        
        # 最初のプレイヤーと最初のフレームからポーズキーポイントの次元を取得
        pose_feat_dim = 0
        if player_pose_features_list and player_pose_features_list[0]:
            pose_feat_dim = player_pose_features_list[0][0].shape[0]
        
        # 各フレームのプレイヤー特徴をパディング
        padded_bbox_features = []
        padded_pose_features = []
        
        for frame_idx in range(self.T):
            # 現在のフレームのプレイヤー数
            if frame_idx < len(player_bbox_features_list):
                frame_bbox = player_bbox_features_list[frame_idx]
                frame_pose = player_pose_features_list[frame_idx]
                num_players = len(frame_bbox)
            else:
                frame_bbox = []
                frame_pose = []
                num_players = 0
            
            # BBox特徴のパディング
            if num_players > 0:
                bbox_tensor = torch.stack(frame_bbox)  # [num_players, 5]
                
                # ポーズ特徴のパディング
                pose_tensor = torch.stack(frame_pose)  # [num_players, pose_feat_dim]
            else:
                bbox_tensor = torch.zeros((0, 5), dtype=torch.float32)
                pose_tensor = torch.zeros((0, pose_feat_dim), dtype=torch.float32)
            
            # max_players_countまでパディング
            if num_players < max_players_count:
                bbox_padding = torch.zeros((max_players_count - num_players, 5), dtype=torch.float32)
                bbox_tensor = torch.cat([bbox_tensor, bbox_padding], dim=0)
                
                pose_padding = torch.zeros((max_players_count - num_players, pose_feat_dim), dtype=torch.float32)
                pose_tensor = torch.cat([pose_tensor, pose_padding], dim=0)
            
            padded_bbox_features.append(bbox_tensor)
            padded_pose_features.append(pose_tensor)
        
        # フレーム間でスタック
        if max_players_count > 0:
            player_bbox_tensor = torch.stack(padded_bbox_features)  # [T, max_players, 5]
            player_pose_tensor = torch.stack(padded_pose_features)  # [T, max_players, pose_feat_dim]
        else:
            player_bbox_tensor = torch.zeros((self.T, 0, 5), dtype=torch.float32)
            player_pose_tensor = torch.zeros((self.T, 0, pose_feat_dim), dtype=torch.float32)
        
        # イベントステータスをテンソル化
        event_status_tensor = torch.tensor(event_statuses, dtype=torch.long)  # [T]
        
        # 出力フォーマット
        if self.output_type == "all":
            target_tensor = event_status_tensor  # [T]
        else:  # "last"
            target_tensor = event_status_tensor[-1]  # スカラー
        
        # 最後のフレームの画像情報を返す
        last_img_id = frame_ids[-1]
        image_info = self.images[last_img_id].copy()
        image_info["id"] = last_img_id
        
        return ball_tensor, player_bbox_tensor, player_pose_tensor, court_tensor, target_tensor, image_info

    @staticmethod
    def group_split_clip(clip_keys, train_ratio, val_ratio, seed):
        random.seed(seed)
        random.shuffle(clip_keys)
        n = len(clip_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            clip_keys[:n_train],
            clip_keys[n_train : n_train + n_val],
            clip_keys[n_train + n_val :],
        )

