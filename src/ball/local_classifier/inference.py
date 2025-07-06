"""
Local Classifier Inference System
3段階フィルタリングシステム（Stage 2: ローカル分類器）

Note: このファイルは新しいパイプラインシステム（src/ball/pipeline/）に移行されました。
互換性のため残していますが、新しい開発では pipeline パッケージを使用してください。
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

from .model import create_local_classifier

logger = logging.getLogger(__name__)


@dataclass
class BallDetection:
    """ボール検出結果（レガシー）"""
    x: float
    y: float
    confidence: float
    stage1_conf: Optional[float] = None  # ball_tracker confidence
    stage2_conf: Optional[float] = None  # local classifier confidence
    stage3_valid: Optional[bool] = None  # trajectory validation


class LocalClassifierInference:
    """
    ローカル分類器の推論システム
    
    ball_trackerと組み合わせた3段階フィルタリング:
    Stage 1: ball_tracker confidence filter
    Stage 2: 16x16 local classifier (this class)
    Stage 3: trajectory consistency check
    """
    
    def __init__(self,
                 model_path: str,
                 model_type: str = "standard",
                 patch_size: int = 16,
                 confidence_threshold: float = 0.5,
                 device: str = "cuda"):
        """
        Args:
            model_path (str): 学習済みモデルのパス
            model_type (str): モデルタイプ ("standard" or "efficient")
            patch_size (int): パッチサイズ
            confidence_threshold (float): 分類の信頼度閾値
            device (str): 使用デバイス
        """
        self.model_path = model_path
        self.model_type = model_type
        self.patch_size = patch_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"Local classifier loaded: {model_type}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
    def _load_model(self) -> torch.nn.Module:
        """学習済みモデルの読み込み"""
        # Create model
        model = create_local_classifier(self.model_type, input_size=self.patch_size)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        return model
        
    def extract_patch(self, image: np.ndarray, center_x: float, center_y: float) -> Optional[np.ndarray]:
        """
        指定位置から16x16パッチを抽出
        
        Args:
            image (np.ndarray): 入力画像 [H, W, 3]
            center_x (float): パッチ中心X座標
            center_y (float): パッチ中心Y座標
            
        Returns:
            Optional[np.ndarray]: 16x16パッチ または None
        """
        h, w = image.shape[:2]
        half_patch = self.patch_size // 2
        
        # Check bounds
        if (center_x < half_patch or center_x >= w - half_patch or
            center_y < half_patch or center_y >= h - half_patch):
            return None
            
        # Extract patch
        x1 = int(center_x - half_patch)
        y1 = int(center_y - half_patch)
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size
        
        patch = image[y1:y2, x1:x2]
        
        # Ensure exact size
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        
        # Ensure uint8 for consistent data type
        if patch.dtype != np.uint8:
            patch = patch.astype(np.uint8)
            
        return patch
        
    def preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        パッチの前処理
        
        Args:
            patch (np.ndarray): RGB patch [H, W, 3]
            
        Returns:
            torch.Tensor: 正規化済みテンソル [1, 3, H, W]
        """
        # Normalize to [0, 1] - ensure float32
        if patch.dtype == np.uint8:
            patch = patch.astype(np.float32) / 255.0
        else:
            patch = patch.astype(np.float32)
            
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        patch = (patch - mean) / std
        
        # HWC -> CHW
        patch = np.transpose(patch, (2, 0, 1))
        
        # Add batch dimension and ensure float32
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
        
        return patch_tensor
        
    def classify_patch(self, patch: np.ndarray) -> float:
        """
        単一パッチの分類
        
        Args:
            patch (np.ndarray): RGB patch [H, W, 3]
            
        Returns:
            float: ボール存在確率 [0, 1]
        """
        # Preprocess
        patch_tensor = self.preprocess_patch(patch)
        
        # Ensure tensor type and move to device
        if patch_tensor.dtype != torch.float32:
            patch_tensor = patch_tensor.float()
        patch_tensor = patch_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(patch_tensor)
            confidence = output.item()
            
        return confidence
        
    def classify_detections(self, 
                           image: np.ndarray, 
                           detections: List[BallDetection]) -> List[BallDetection]:
        """
        複数検出結果の分類
        
        Args:
            image (np.ndarray): 入力画像 [H, W, 3]
            detections (List[BallDetection]): Stage1通過検出結果
            
        Returns:
            List[BallDetection]: Stage2通過検出結果
        """
        filtered_detections = []
        
        for detection in detections:
            # Extract patch
            patch = self.extract_patch(image, detection.x, detection.y)
            if patch is None:
                continue
                
            # Classify patch
            confidence = self.classify_patch(patch)
            
            # Update detection
            detection.stage2_conf = confidence
            
            # Apply threshold
            if confidence >= self.confidence_threshold:
                filtered_detections.append(detection)
                
        logger.debug(f"Stage2 filtering: {len(detections)} -> {len(filtered_detections)}")
        return filtered_detections
        
    def batch_classify(self, 
                      image: np.ndarray, 
                      positions: List[Tuple[float, float]],
                      batch_size: int = 32) -> List[float]:
        """
        バッチ処理による高速分類
        
        Args:
            image (np.ndarray): 入力画像
            positions (List[Tuple]): 検証する座標リスト
            batch_size (int): バッチサイズ
            
        Returns:
            List[float]: 各位置の信頼度
        """
        confidences = []
        
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i + batch_size]
            batch_patches = []
            
            # Extract patches
            for x, y in batch_positions:
                patch = self.extract_patch(image, x, y)
                if patch is not None:
                    batch_patches.append(patch)
                else:
                    batch_patches.append(np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8))
                    
            if not batch_patches:
                continue
                
            # Batch preprocessing
            batch_tensor = torch.stack([
                self.preprocess_patch(patch).squeeze(0) 
                for patch in batch_patches
            ]).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                batch_confidences = outputs.squeeze().cpu().numpy()
                
                if len(batch_positions) == 1:
                    batch_confidences = [batch_confidences.item()]
                else:
                    batch_confidences = batch_confidences.tolist()
                    
                confidences.extend(batch_confidences)
                
        return confidences


class ThreeStageFilter:
    """
    3段階フィルタリングシステム統合クラス
    """
    
    def __init__(self,
                 ball_tracker_model,  # ball_tracker model instance
                 local_classifier: LocalClassifierInference,
                 stage1_threshold: float = 0.5,
                 stage3_max_distance: float = 50.0,
                 stage3_window_size: int = 5):
        """
        Args:
            ball_tracker_model: ball_trackerモデルインスタンス
            local_classifier: ローカル分類器インスタンス
            stage1_threshold: Stage1信頼度閾値
            stage3_max_distance: Stage3最大移動距離
            stage3_window_size: Stage3軌跡ウィンドウサイズ
        """
        self.ball_tracker = ball_tracker_model
        self.local_classifier = local_classifier
        self.stage1_threshold = stage1_threshold
        self.stage3_max_distance = stage3_max_distance
        self.stage3_window_size = stage3_window_size
        
        # Trajectory history for Stage 3
        self.trajectory_history = []
        
    def process_frame(self, frame: np.ndarray) -> List[BallDetection]:
        """
        単一フレームの3段階処理
        
        Args:
            frame (np.ndarray): 入力フレーム
            
        Returns:
            List[BallDetection]: 最終フィルタリング結果
        """
        # Stage 1: ball_tracker detection
        stage1_detections = self._stage1_detection(frame)
        if not stage1_detections:
            return []
            
        # Stage 2: Local classifier filtering
        stage2_detections = self.local_classifier.classify_detections(frame, stage1_detections)
        if not stage2_detections:
            return []
            
        # Stage 3: Trajectory consistency
        stage3_detections = self._stage3_trajectory_filter(stage2_detections)
        
        # Update trajectory history
        self._update_trajectory_history(stage3_detections)
        
        return stage3_detections
        
    def _stage1_detection(self, frame: np.ndarray) -> List[BallDetection]:
        """Stage 1: ball_tracker検出"""
        # ball_tracker inference (implementation depends on ball_tracker API)
        # This is a placeholder - actual implementation would depend on ball_tracker interface
        
        detections = []
        # Example: assuming ball_tracker returns list of (x, y, confidence)
        # results = self.ball_tracker.detect(frame)
        # for x, y, conf in results:
        #     if conf >= self.stage1_threshold:
        #         detection = BallDetection(x=x, y=y, confidence=conf, stage1_conf=conf)
        #         detections.append(detection)
        
        return detections
        
    def _stage3_trajectory_filter(self, detections: List[BallDetection]) -> List[BallDetection]:
        """Stage 3: 軌跡一貫性チェック"""
        if not self.trajectory_history or len(self.trajectory_history) < 2:
            # 履歴不足の場合はそのまま通す
            for det in detections:
                det.stage3_valid = True
            return detections
            
        valid_detections = []
        
        for detection in detections:
            # Check consistency with recent trajectory
            is_consistent = self._check_trajectory_consistency(detection)
            detection.stage3_valid = is_consistent
            
            if is_consistent:
                valid_detections.append(detection)
                
        return valid_detections
        
    def _check_trajectory_consistency(self, detection: BallDetection) -> bool:
        """軌跡一貫性の確認"""
        if not self.trajectory_history:
            return True
            
        # Get recent positions
        recent_positions = self.trajectory_history[-self.stage3_window_size:]
        
        # Calculate average velocity
        if len(recent_positions) >= 2:
            velocities = []
            for i in range(1, len(recent_positions)):
                prev_pos = recent_positions[i-1]
                curr_pos = recent_positions[i]
                
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                velocities.append((dx, dy))
                
            # Average velocity
            avg_vx = np.mean([v[0] for v in velocities])
            avg_vy = np.mean([v[1] for v in velocities])
            
            # Predict next position
            last_pos = recent_positions[-1]
            predicted_x = last_pos[0] + avg_vx
            predicted_y = last_pos[1] + avg_vy
            
            # Check distance from prediction
            distance = np.sqrt((detection.x - predicted_x)**2 + (detection.y - predicted_y)**2)
            
            return distance <= self.stage3_max_distance
        else:
            # Simple distance check from last position
            last_pos = recent_positions[-1]
            distance = np.sqrt((detection.x - last_pos[0])**2 + (detection.y - last_pos[1])**2)
            return distance <= self.stage3_max_distance
            
    def _update_trajectory_history(self, detections: List[BallDetection]):
        """軌跡履歴の更新"""
        if detections:
            # Use highest confidence detection
            best_detection = max(detections, key=lambda d: d.confidence)
            self.trajectory_history.append((best_detection.x, best_detection.y))
            
            # Keep only recent history
            max_history = self.stage3_window_size * 2
            if len(self.trajectory_history) > max_history:
                self.trajectory_history = self.trajectory_history[-max_history:] 