# streaming_overlayer/event_utils.py

"""
Eventワーカー統合に関するユーティリティ関数群
"""

import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def normalize_coordinates(x: float, y: float, img_width: int, img_height: int) -> Tuple[float, float]:
    """
    座標を正規化します。
    
    Args:
        x: X座標
        y: Y座標
        img_width: 画像幅
        img_height: 画像高さ
        
    Returns:
        Tuple[float, float]: 正規化された座標 (x_norm, y_norm)
    """
    return x / img_width, y / img_height


def denormalize_coordinates(x_norm: float, y_norm: float, img_width: int, img_height: int) -> Tuple[int, int]:
    """
    正規化された座標を元に戻します。
    
    Args:
        x_norm: 正規化されたX座標
        y_norm: 正規化されたY座標
        img_width: 画像幅
        img_height: 画像高さ
        
    Returns:
        Tuple[int, int]: 元の座標 (x, y)
    """
    return int(x_norm * img_width), int(y_norm * img_height)


def extract_ball_features(ball_result: Dict[str, Any], img_width: int = 1920, img_height: int = 1080) -> np.ndarray:
    """
    BallWorkerの結果から特徴量を抽出します。
    
    Args:
        ball_result: BallWorkerの結果辞書
        img_width: 画像幅
        img_height: 画像高さ
        
    Returns:
        np.ndarray: Ball特徴量 [3] (正規化x, 正規化y, confidence)
    """
    if ball_result and isinstance(ball_result, dict) and 'x' in ball_result and 'y' in ball_result:
        x_norm, y_norm = normalize_coordinates(ball_result['x'], ball_result['y'], img_width, img_height)
        confidence = ball_result.get('confidence', 0.0)
        return np.array([x_norm, y_norm, confidence], dtype=np.float32)
    else:
        return np.zeros(3, dtype=np.float32)


def extract_court_features(court_result: List[Dict[str, Any]], 
                          img_width: int = 1920, img_height: int = 1080,
                          max_points: int = 15) -> np.ndarray:
    """
    CourtWorkerの結果から特徴量を抽出します。
    
    Args:
        court_result: CourtWorkerの結果リスト
        img_width: 画像幅
        img_height: 画像高さ
        max_points: 最大キーポイント数
        
    Returns:
        np.ndarray: Court特徴量 [max_points * 3]
    """
    court_features = np.zeros(max_points * 3, dtype=np.float32)
    
    if court_result and isinstance(court_result, list):
        for i, point in enumerate(court_result[:max_points]):
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                x_norm, y_norm = normalize_coordinates(point['x'], point['y'], img_width, img_height)
                confidence = point.get('confidence', 0.0)
                
                court_features[i*3] = x_norm
                court_features[i*3+1] = y_norm
                court_features[i*3+2] = confidence
    
    return court_features


def extract_player_features(pose_result: List[Dict[str, Any]], 
                           img_width: int = 1920, img_height: int = 1080,
                           max_players: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    PoseWorkerの結果からプレイヤー特徴量を抽出します。
    
    Args:
        pose_result: PoseWorkerの結果リスト
        img_width: 画像幅
        img_height: 画像高さ
        max_players: 最大プレイヤー数
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (bbox特徴量[max_players, 5], pose特徴量[max_players, 51])
    """
    player_bbox_features = np.zeros((max_players, 5), dtype=np.float32)
    player_pose_features = np.zeros((max_players, 51), dtype=np.float32)  # 17 keypoints * 3
    
    if pose_result and isinstance(pose_result, list):
        for i, person in enumerate(pose_result[:max_players]):
            if isinstance(person, dict):
                # BBox特徴量
                bbox = person.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x1_norm, y1_norm = normalize_coordinates(bbox[0], bbox[1], img_width, img_height)
                    x2_norm, y2_norm = normalize_coordinates(bbox[0] + bbox[2], bbox[1] + bbox[3], img_width, img_height)
                    det_score = person.get('det_score', 0.0)
                    
                    player_bbox_features[i] = np.array([x1_norm, y1_norm, x2_norm, y2_norm, det_score], dtype=np.float32)
                
                # Pose特徴量
                keypoints = person.get('keypoints', [])
                scores = person.get('scores', [])
                if keypoints and len(keypoints) >= 17:
                    for j, (x, y) in enumerate(keypoints[:17]):
                        x_norm, y_norm = normalize_coordinates(x, y, img_width, img_height)
                        score = scores[j] if j < len(scores) else 0.0
                        
                        player_pose_features[i, j*3] = x_norm
                        player_pose_features[i, j*3+1] = y_norm
                        player_pose_features[i, j*3+2] = score
    
    return player_bbox_features, player_pose_features


def combine_worker_results(ball_result: Dict[str, Any], 
                          court_result: List[Dict[str, Any]], 
                          pose_result: List[Dict[str, Any]],
                          img_width: int = 1920, img_height: int = 1080,
                          max_players: int = 4) -> Dict[str, np.ndarray]:
    """
    複数のワーカー結果を統合してevent推論用の特徴量に変換します。
    
    Args:
        ball_result: BallWorkerの結果
        court_result: CourtWorkerの結果
        pose_result: PoseWorkerの結果
        img_width: 画像幅
        img_height: 画像高さ
        max_players: 最大プレイヤー数
        
    Returns:
        Dict[str, np.ndarray]: 統合された特徴量辞書
    """
    try:
        # 各特徴量を抽出
        ball_features = extract_ball_features(ball_result, img_width, img_height)
        court_features = extract_court_features(court_result, img_width, img_height)
        player_bbox_features, player_pose_features = extract_player_features(
            pose_result, img_width, img_height, max_players
        )
        
        return {
            "ball": ball_features,
            "court": court_features,
            "player_bbox": player_bbox_features,
            "player_pose": player_pose_features
        }
        
    except Exception as e:
        logger.error(f"ワーカー結果統合エラー: {e}")
        # エラー時はゼロ特徴量を返す
        return {
            "ball": np.zeros(3, dtype=np.float32),
            "court": np.zeros(45, dtype=np.float32),
            "player_bbox": np.zeros((max_players, 5), dtype=np.float32),
            "player_pose": np.zeros((max_players, 51), dtype=np.float32)
        }


def create_sequence_tensors(feature_sequence: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """
    特徴量シーケンスをPyTorchテンソルに変換します。
    
    Args:
        feature_sequence: 特徴量辞書のリスト [T個の特徴量辞書]
        
    Returns:
        Dict[str, torch.Tensor]: テンソル辞書 (バッチ次元付き)
    """
    if not feature_sequence:
        raise ValueError("Empty feature sequence provided")
    
    try:
        # 各特徴量タイプ別にシーケンスを構築
        ball_sequence = [frame_features["ball"] for frame_features in feature_sequence]
        court_sequence = [frame_features["court"] for frame_features in feature_sequence]
        player_bbox_sequence = [frame_features["player_bbox"] for frame_features in feature_sequence]
        player_pose_sequence = [frame_features["player_pose"] for frame_features in feature_sequence]
        
        # NumPy配列をスタック
        ball_array = np.stack(ball_sequence, axis=0)  # [T, 3]
        court_array = np.stack(court_sequence, axis=0)  # [T, 45]
        player_bbox_array = np.stack(player_bbox_sequence, axis=0)  # [T, P, 5]
        player_pose_array = np.stack(player_pose_sequence, axis=0)  # [T, P, 51]
        
        # テンソルに変換（バッチ次元追加）
        tensor_dict = {
            'ball_features': torch.tensor(ball_array, dtype=torch.float32).unsqueeze(0),  # [1, T, 3]
            'court_features': torch.tensor(court_array, dtype=torch.float32).unsqueeze(0),  # [1, T, 45]
            'player_bbox_features': torch.tensor(player_bbox_array, dtype=torch.float32).unsqueeze(0),  # [1, T, P, 5]
            'player_pose_features': torch.tensor(player_pose_array, dtype=torch.float32).unsqueeze(0)   # [1, T, P, 51]
        }
        
        return tensor_dict
        
    except Exception as e:
        logger.error(f"シーケンステンソル作成エラー: {e}")
        raise


def validate_worker_results(ball_result: Any, court_result: Any, pose_result: Any) -> bool:
    """
    ワーカー結果の形式を検証します。
    
    Args:
        ball_result: BallWorkerの結果
        court_result: CourtWorkerの結果
        pose_result: PoseWorkerの結果
        
    Returns:
        bool: 全て有効な場合True
    """
    try:
        # Ball結果検証
        ball_valid = (
            isinstance(ball_result, dict) and 
            'x' in ball_result and 
            'y' in ball_result and
            isinstance(ball_result['x'], (int, float)) and
            isinstance(ball_result['y'], (int, float))
        )
        
        # Court結果検証
        court_valid = isinstance(court_result, list)
        if court_valid and court_result:
            court_valid = all(
                isinstance(point, dict) and 'x' in point and 'y' in point
                for point in court_result
            )
        
        # Pose結果検証
        pose_valid = isinstance(pose_result, list)
        if pose_valid and pose_result:
            pose_valid = all(
                isinstance(person, dict) and 'bbox' in person and 'keypoints' in person
                for person in pose_result
            )
        
        return ball_valid and court_valid and pose_valid
        
    except Exception as e:
        logger.error(f"ワーカー結果検証エラー: {e}")
        return False


def get_image_dimensions_from_frame(frame: Optional[np.ndarray]) -> Tuple[int, int]:
    """
    フレームから画像サイズを取得します。
    
    Args:
        frame: 画像フレーム
        
    Returns:
        Tuple[int, int]: (width, height)
    """
    if frame is not None and len(frame.shape) >= 2:
        height, width = frame.shape[:2]
        return width, height
    else:
        # デフォルト値
        return 1920, 1080


class EventResultFormatter:
    """Event推論結果のフォーマッター"""
    
    @staticmethod
    def format_for_overlay(event_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Event推論結果をオーバーレイ用にフォーマットします。
        
        Args:
            event_result: Event推論の結果辞書
            
        Returns:
            Dict[str, Any]: オーバーレイ用フォーマット済み結果
        """
        formatted = {
            'hit_detected': event_result.get('hit_detected', False),
            'bounce_detected': event_result.get('bounce_detected', False),
            'hit_probability': event_result.get('hit_probability', 0.0),
            'bounce_probability': event_result.get('bounce_probability', 0.0),
            'timestamp': event_result.get('timestamp', 0),
            'signal_history': event_result.get('signal_history', [])
        }
        
        # 平滑化された信号がある場合
        if 'smoothed_hit_signal' in event_result:
            formatted['smoothed_hit_signal'] = event_result['smoothed_hit_signal']
        if 'smoothed_bounce_signal' in event_result:
            formatted['smoothed_bounce_signal'] = event_result['smoothed_bounce_signal']
        
        return formatted
    
    @staticmethod
    def format_for_logging(event_result: Dict[str, Any]) -> str:
        """
        Event推論結果をログ用にフォーマットします。
        
        Args:
            event_result: Event推論の結果辞書
            
        Returns:
            str: ログ用文字列
        """
        hit_prob = event_result.get('hit_probability', 0.0)
        bounce_prob = event_result.get('bounce_probability', 0.0)
        hit_detected = event_result.get('hit_detected', False)
        bounce_detected = event_result.get('bounce_detected', False)
        timestamp = event_result.get('timestamp', 0)
        
        status = []
        if hit_detected:
            status.append("HIT")
        if bounce_detected:
            status.append("BOUNCE")
        
        status_str = " | ".join(status) if status else "NONE"
        
        return f"t={timestamp:06d} | {status_str} | hit={hit_prob:.3f} bounce={bounce_prob:.3f}"


def debug_print_features(features: Dict[str, np.ndarray], frame_idx: int):
    """
    デバッグ用に特徴量情報を出力します。
    
    Args:
        features: 特徴量辞書
        frame_idx: フレームインデックス
    """
    print(f"\n=== フレーム {frame_idx} 特徴量情報 ===")
    
    for feature_name, feature_array in features.items():
        print(f"{feature_name}: shape={feature_array.shape}, dtype={feature_array.dtype}")
        
        # 非ゼロ要素の割合を計算
        if feature_array.size > 0:
            non_zero_ratio = np.count_nonzero(feature_array) / feature_array.size
            print(f"  非ゼロ要素率: {non_zero_ratio:.2%}")
            
            # 統計情報
            if feature_array.size > 0:
                print(f"  min={np.min(feature_array):.3f}, max={np.max(feature_array):.3f}, mean={np.mean(feature_array):.3f}")
    
    print("="*50) 