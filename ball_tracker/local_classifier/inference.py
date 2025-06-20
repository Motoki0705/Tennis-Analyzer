"""
Local Classifier Inference
ローカル分類器による2次検証システム
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging

from .model import LocalBallClassifier, EfficientLocalClassifier, create_local_classifier

logger = logging.getLogger(__name__)


class LocalClassifierInference:
    """
    ローカル分類器による推論・検証クラス
    ball_trackerの検出結果に対する2次フィルタリング
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda',
                 patch_size: int = 16,
                 confidence_threshold: float = 0.7,
                 model_type: str = "standard"):
        """
        Args:
            model_path (str): 学習済みモデルのパス
            device (str): 推論デバイス
            patch_size (int): パッチサイズ
            confidence_threshold (float): 分類信頼度閾値
            model_type (str): モデルタイプ
        """
        self.device = torch.device(device)
        self.patch_size = patch_size
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = create_local_classifier(model_type, input_size=patch_size)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        logger.info(f"Local classifier loaded: {model_path}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
    def verify_detection(self, 
                        frame: np.ndarray, 
                        detection: Dict,
                        return_patch: bool = False) -> Dict:
        """
        単一検出の検証
        
        Args:
            frame (np.ndarray): フレーム画像 [H, W, 3]
            detection (Dict): ball_trackerの検出結果 {'xy': [x, y], 'score': float}
            return_patch (bool): パッチ画像も返すか
            
        Returns:
            Dict: 検証結果
        """
        x, y = detection['xy']
        x, y = int(x), int(y)
        
        # Extract patch
        patch = self._extract_patch(frame, x, y)
        if patch is None:
            return {
                'verified': False,
                'local_confidence': 0.0,
                'reason': 'patch_extraction_failed'
            }
            
        # Classify patch
        with torch.no_grad():
            confidence = self._classify_patch(patch)
            
        verified = confidence >= self.confidence_threshold
        
        result = {
            'verified': verified,
            'local_confidence': confidence,
            'original_score': detection['score'],
            'position': [x, y]
        }
        
        if return_patch:
            result['patch'] = patch
            
        return result
        
    def verify_detections_batch(self, 
                               frame: np.ndarray, 
                               detections: List[Dict]) -> List[Dict]:
        """
        複数検出の一括検証
        
        Args:
            frame (np.ndarray): フレーム画像
            detections (List[Dict]): 検出結果リスト
            
        Returns:
            List[Dict]: 検証結果リスト
        """
        if not detections:
            return []
            
        # Extract all patches
        patches = []
        valid_indices = []
        
        for i, detection in enumerate(detections):
            x, y = int(detection['xy'][0]), int(detection['xy'][1])
            patch = self._extract_patch(frame, x, y)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)
                
        if not patches:
            return [{'verified': False, 'local_confidence': 0.0, 
                    'reason': 'no_valid_patches'} for _ in detections]
            
        # Batch classification
        with torch.no_grad():
            confidences = self._classify_patches_batch(patches)
            
        # Prepare results
        results = []
        patch_idx = 0
        
        for i, detection in enumerate(detections):
            if i in valid_indices:
                confidence = confidences[patch_idx]
                verified = confidence >= self.confidence_threshold
                
                results.append({
                    'verified': verified,
                    'local_confidence': confidence,
                    'original_score': detection['score'],
                    'position': [int(detection['xy'][0]), int(detection['xy'][1])]
                })
                patch_idx += 1
            else:
                results.append({
                    'verified': False,
                    'local_confidence': 0.0,
                    'reason': 'patch_extraction_failed'
                })
                
        return results
        
    def _extract_patch(self, frame: np.ndarray, center_x: int, center_y: int) -> Optional[np.ndarray]:
        """パッチの抽出"""
        h, w = frame.shape[:2]
        half_patch = self.patch_size // 2
        
        # Check bounds
        if (center_x < half_patch or center_x >= w - half_patch or
            center_y < half_patch or center_y >= h - half_patch):
            return None
            
        # Extract patch
        x1 = center_x - half_patch
        y1 = center_y - half_patch
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size
        
        patch = frame[y1:y2, x1:x2]
        
        # Ensure RGB format
        if patch.shape[2] == 3:
            return patch
        else:
            return None
            
    def _classify_patch(self, patch: np.ndarray) -> float:
        """単一パッチの分類"""
        # Preprocess
        transformed = self.transform(image=patch)
        tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        output = self.model(tensor)
        confidence = output.squeeze().cpu().item()
        
        return confidence
        
    def _classify_patches_batch(self, patches: List[np.ndarray]) -> List[float]:
        """複数パッチの一括分類"""
        # Preprocess batch
        tensors = []
        for patch in patches:
            transformed = self.transform(image=patch)
            tensors.append(transformed['image'])
            
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Batch inference
        outputs = self.model(batch_tensor)
        confidences = outputs.squeeze().cpu().tolist()
        
        # Handle single item case
        if isinstance(confidences, float):
            confidences = [confidences]
            
        return confidences


class EnhancedTracker:
    """
    3段階フィルタリング統合システム
    1. ball_tracker確信度フィルタ
    2. ローカル分類器検証
    3. 軌跡一貫性チェック
    """
    
    def __init__(self,
                 ball_tracker,
                 local_classifier: LocalClassifierInference,
                 primary_threshold: float = 0.5,
                 local_threshold: float = 0.7,
                 max_jump_distance: float = 150.0):
        """
        Args:
            ball_tracker: ball_trackerインスタンス
            local_classifier: ローカル分類器
            primary_threshold: 1次確信度閾値
            local_threshold: ローカル分類器閾値
            max_jump_distance: 最大ジャンプ距離（ピクセル）
        """
        self.ball_tracker = ball_tracker
        self.local_classifier = local_classifier
        self.primary_threshold = primary_threshold
        self.local_threshold = local_threshold
        self.max_jump_distance = max_jump_distance
        
        # Tracking history
        self.tracking_history = []
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        フレーム処理（3段階フィルタリング）
        
        Args:
            frame (np.ndarray): 入力フレーム
            
        Returns:
            Dict: 処理結果
        """
        # Stage 1: Primary detection
        detections = self.ball_tracker.detect(frame)
        stage1_filtered = [d for d in detections if d['score'] >= self.primary_threshold]
        
        if not stage1_filtered:
            result = {
                'detected': False,
                'position': None,
                'confidence': 0.0,
                'filter_stages': {
                    'stage1_candidates': len(detections),
                    'stage2_verified': 0,
                    'stage3_final': 0
                }
            }
            self.tracking_history.append(None)
            return result
            
        # Stage 2: Local classifier verification
        verifications = self.local_classifier.verify_detections_batch(frame, stage1_filtered)
        stage2_verified = []
        
        for detection, verification in zip(stage1_filtered, verifications):
            if verification['verified']:
                detection['local_confidence'] = verification['local_confidence']
                stage2_verified.append(detection)
                
        if not stage2_verified:
            result = {
                'detected': False,
                'position': None,
                'confidence': 0.0,
                'filter_stages': {
                    'stage1_candidates': len(detections),
                    'stage2_verified': 0,
                    'stage3_final': 0
                }
            }
            self.tracking_history.append(None)
            return result
            
        # Stage 3: Trajectory consistency check
        final_detection = self._trajectory_consistency_check(stage2_verified)
        
        result = {
            'detected': final_detection is not None,
            'position': final_detection['xy'] if final_detection else None,
            'confidence': final_detection['score'] if final_detection else 0.0,
            'local_confidence': final_detection.get('local_confidence', 0.0) if final_detection else 0.0,
            'filter_stages': {
                'stage1_candidates': len(detections),
                'stage2_verified': len(stage2_verified),
                'stage3_final': 1 if final_detection else 0
            }
        }
        
        # Update tracking history
        if final_detection:
            self.tracking_history.append(final_detection['xy'])
        else:
            self.tracking_history.append(None)
            
        # Keep only recent history
        if len(self.tracking_history) > 10:
            self.tracking_history.pop(0)
            
        return result
        
    def _trajectory_consistency_check(self, detections: List[Dict]) -> Optional[Dict]:
        """軌跡一貫性チェック"""
        if not self.tracking_history:
            # No history, return best detection
            return max(detections, key=lambda d: d['score'])
            
        # Get last valid position
        last_position = None
        for pos in reversed(self.tracking_history):
            if pos is not None:
                last_position = pos
                break
                
        if last_position is None:
            return max(detections, key=lambda d: d['score'])
            
        # Find closest detection to last position
        valid_detections = []
        for detection in detections:
            distance = np.linalg.norm(np.array(detection['xy']) - np.array(last_position))
            if distance <= self.max_jump_distance:
                detection['distance_from_last'] = distance
                valid_detections.append(detection)
                
        if not valid_detections:
            # All detections are too far, might be tracking failure
            # Return best detection but mark as suspicious
            best_detection = max(detections, key=lambda d: d['score'])
            best_detection['suspicious'] = True
            return best_detection
            
        # Return closest valid detection
        return min(valid_detections, key=lambda d: d['distance_from_last'])
        
    def reset_tracking(self):
        """トラッキング履歴のリセット"""
        self.tracking_history = []


def create_enhanced_tracker(ball_tracker_config: Dict,
                          local_classifier_config: Dict) -> EnhancedTracker:
    """
    強化トラッカーのファクトリ関数
    
    Args:
        ball_tracker_config (Dict): ball_tracker設定
        local_classifier_config (Dict): ローカル分類器設定
        
    Returns:
        EnhancedTracker: 強化トラッカーインスタンス
    """
    # Import ball_tracker components (assumed to be available)
    from ..video_demo import SimpleDetector, load_simple_config
    from ..online import OnlineTracker
    
    # Create ball_tracker
    cfg = load_simple_config()
    cfg.detector.model_path = ball_tracker_config['model_path']
    detector = SimpleDetector(cfg, ball_tracker_config.get('device', 'cuda'))
    tracker = OnlineTracker(cfg)
    
    # Create local classifier
    local_classifier = LocalClassifierInference(**local_classifier_config)
    
    # Create enhanced tracker
    enhanced_tracker = EnhancedTracker(
        ball_tracker={'detector': detector, 'tracker': tracker},
        local_classifier=local_classifier,
        primary_threshold=ball_tracker_config.get('primary_threshold', 0.5),
        local_threshold=local_classifier_config.get('confidence_threshold', 0.7),
        max_jump_distance=ball_tracker_config.get('max_jump_distance', 150.0)
    )
    
    return enhanced_tracker


if __name__ == "__main__":
    # Inference test
    print("🔍 Local Classifier Inference Test")
    print("=" * 50)
    
    # Test parameters
    model_path = "path/to/local_classifier.pth"  # 実際のパスに変更
    
    if Path(model_path).exists():
        # Create inference instance
        classifier = LocalClassifierInference(
            model_path=model_path,
            device='cuda',
            confidence_threshold=0.7
        )
        
        # Test with dummy data
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_detection = {'xy': [320, 240], 'score': 0.8}
        
        result = classifier.verify_detection(dummy_frame, dummy_detection)
        print(f"Verification result: {result}")
        
    else:
        print("⚠️ Model file not found. Please train local classifier first.") 