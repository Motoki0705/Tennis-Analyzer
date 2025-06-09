import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

logger = logging.getLogger(__name__)


class EventWorker(BaseWorker):
    """
    イベント検知（バウンド・ショット）のためのパイプラインワーカー。
    
    ball、player、courtの特徴量を統合してLitTransformerV2で推論を実行します。
    """

    def __init__(self, name, predictor, preprocess_q, inference_q, postprocess_q, 
                 results_q, debug=False, sequence_length: int = 16):
        """
        Args:
            sequence_length (int): 時系列の長さ（TransformerV2の入力シーケンス長）
        """
        super().__init__(name, predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)
        self.sequence_length = sequence_length
        # 過去のフレーム情報を保持するバッファ
        self.frame_buffer: List[Dict[str, Any]] = []
        
    def _process_preprocess_task(self, task):
        """
        前処理タスクを処理します。
        
        他のワーカーからの結果を統合してシーケンスデータを構築します。
        task.frames には各フレームの統合情報が格納されていることを想定。
        """
        try:
            # バッファにフレーム情報を追加
            for i, frame_data in enumerate(task.frames):
                frame_idx = task.meta_data[i][0] if task.meta_data else i
                
                frame_info = {
                    'frame_idx': frame_idx,
                    'ball_features': frame_data.get('ball', None),
                    'player_bbox_features': frame_data.get('player_bbox', None),
                    'player_pose_features': frame_data.get('player_pose', None),
                    'court_features': frame_data.get('court', None),
                }
                
                self.frame_buffer.append(frame_info)
            
            # バッファサイズを制限
            if len(self.frame_buffer) > self.sequence_length * 2:
                self.frame_buffer = self.frame_buffer[-self.sequence_length:]
            
            # シーケンス長に達したら推論タスクを作成
            if len(self.frame_buffer) >= self.sequence_length:
                sequence_data = self._create_sequence_tensor(self.frame_buffer[-self.sequence_length:])
                self.inference_queue.put(InferenceTask(task.task_id, sequence_data, task.meta_data))
                
        except Exception as e:
            logger.error(f"EventWorker前処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _create_sequence_tensor(self, frame_sequence: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        フレームシーケンスからTransformerV2用のテンソルを作成します。
        
        Args:
            frame_sequence: フレーム情報のリスト
            
        Returns:
            Dict containing ball_features, player_bbox_features, player_pose_features, court_features
        """
        try:
            # 各特徴量の初期化
            ball_features = []
            player_bbox_features = []
            player_pose_features = []
            court_features = []
            
            max_players = 0  # バッチ内の最大プレイヤー数
            
            # まず最大プレイヤー数を確定
            for frame_info in frame_sequence:
                if frame_info['player_bbox_features'] is not None:
                    num_players = len(frame_info['player_bbox_features'])
                    max_players = max(max_players, num_players)
            
            # デフォルトの特徴量次元
            ball_dim = 3
            court_dim = 45  # 15 keypoints * 3
            player_bbox_dim = 5
            player_pose_dim = 51  # 17 keypoints * 3
            
            for frame_info in frame_sequence:
                # Ball features
                if frame_info['ball_features'] is not None:
                    ball_feat = torch.tensor(frame_info['ball_features'][:ball_dim], dtype=torch.float32)
                else:
                    ball_feat = torch.zeros(ball_dim, dtype=torch.float32)
                ball_features.append(ball_feat)
                
                # Court features
                if frame_info['court_features'] is not None:
                    court_feat = torch.tensor(frame_info['court_features'][:court_dim], dtype=torch.float32)
                else:
                    court_feat = torch.zeros(court_dim, dtype=torch.float32)
                court_features.append(court_feat)
                
                # Player features (with padding)
                frame_bbox_features = []
                frame_pose_features = []
                
                if (frame_info['player_bbox_features'] is not None and 
                    frame_info['player_pose_features'] is not None):
                    num_players = min(len(frame_info['player_bbox_features']), 
                                    len(frame_info['player_pose_features']))
                    
                    for p_idx in range(num_players):
                        bbox_feat = torch.tensor(frame_info['player_bbox_features'][p_idx][:player_bbox_dim], 
                                               dtype=torch.float32)
                        pose_feat = torch.tensor(frame_info['player_pose_features'][p_idx][:player_pose_dim], 
                                               dtype=torch.float32)
                        frame_bbox_features.append(bbox_feat)
                        frame_pose_features.append(pose_feat)
                
                # パディング
                while len(frame_bbox_features) < max_players:
                    frame_bbox_features.append(torch.zeros(player_bbox_dim, dtype=torch.float32))
                    frame_pose_features.append(torch.zeros(player_pose_dim, dtype=torch.float32))
                
                player_bbox_features.append(torch.stack(frame_bbox_features))
                player_pose_features.append(torch.stack(frame_pose_features))
            
            # バッチ次元を追加してテンソル化
            return {
                'ball_features': torch.stack(ball_features).unsqueeze(0),  # [1, T, 3]
                'player_bbox_features': torch.stack(player_bbox_features).unsqueeze(0),  # [1, T, P, 5]
                'player_pose_features': torch.stack(player_pose_features).unsqueeze(0),  # [1, T, P, 51]
                'court_features': torch.stack(court_features).unsqueeze(0),  # [1, T, 45]
            }
            
        except Exception as e:
            logger.error(f"シーケンステンソル作成エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー時はゼロテンソルを返す
            return {
                'ball_features': torch.zeros(1, self.sequence_length, ball_dim),
                'player_bbox_features': torch.zeros(1, self.sequence_length, max(max_players, 1), player_bbox_dim),
                'player_pose_features': torch.zeros(1, self.sequence_length, max(max_players, 1), player_pose_dim),
                'court_features': torch.zeros(1, self.sequence_length, court_dim),
            }

    def _process_inference_task(self, task):
        """推論タスクを処理します。"""
        try:
            with torch.no_grad():
                # predictorのinferenceメソッドを呼び出し
                preds = self.predictor.inference(task.tensor_data)
            
            self.postprocess_queue.put(PostprocessTask(task.task_id, preds, task.meta_data))
            
        except Exception as e:
            logger.error(f"EventWorker推論エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_postprocess_task(self, task):
        """後処理タスクを処理します。"""
        try:
            # predictorのpostprocessメソッドを呼び出し
            processed_output = self.predictor.postprocess(task.inference_output, task.meta_data)
            
            # 最新フレームのインデックスを取得
            if task.meta_data and len(task.meta_data) > 0:
                latest_frame_idx = max(meta[0] for meta in task.meta_data)
            else:
                latest_frame_idx = 0
            
            # 結果をキューに追加 (frame_index, task_name, result)
            self.results_queue.put((latest_frame_idx, "event", processed_output))
            
        except Exception as e:
            logger.error(f"EventWorker後処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc() 