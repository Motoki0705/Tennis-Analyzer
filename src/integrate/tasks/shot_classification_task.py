"""
Shot classification task that depends on multiple other tasks.
Example of a complex task that uses multiple dependencies.
"""
import torch
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import logging

from ..core.base_task import BaseTask

log = logging.getLogger(__name__)


class ShotClassificationTask(BaseTask):
    """
    Tennis shot classification task.
    Depends on ball tracking and pose estimation to classify shot types.
    """
    
    def initialize(self) -> None:
        """Initialize shot classification model."""
        try:
            # Shot type mappings
            self.shot_types = self.config.get('shot_types', ['serve', 'forehand', 'backhand', 'volley', 'smash'])
            self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
            
            # Simple rule-based classifier (in production, use ML model)
            self.use_ml_model = self.config.get('use_ml_model', False)
            
            if self.use_ml_model:
                # Load trained ML model
                model_path = self.config.get('model_path')
                if model_path:
                    self.model = torch.load(model_path, map_location=self.device)
                    self.model.eval()
                    log.info(f"Loaded shot classification model from {model_path}")
                else:
                    raise ValueError("ML model path required when use_ml_model=True")
            else:
                log.info("Using rule-based shot classification")
            
        except Exception as e:
            log.error(f"Failed to initialize shot classification task: {e}")
            raise
    
    def preprocess(self, frames: List[Any], metadata: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Preprocess depends on other task results.
        """
        try:
            # Get dependency results
            dependencies = metadata.get('dependencies', {}) if metadata else {}
            ball_results = dependencies.get('ball_tracking')
            pose_results = dependencies.get('pose_estimation')
            
            # Check if we have required dependencies
            has_ball = ball_results and 'batch_results' in ball_results
            has_pose = pose_results and 'batch_results' in pose_results
            
            preprocess_meta = {
                'batch_size': len(frames),
                'has_ball_data': has_ball,
                'has_pose_data': has_pose,
                'original_frames': frames,
                'dependencies': dependencies,
                'task_config': {
                    'shot_types': self.shot_types,
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
            if not (has_ball and has_pose):
                return None, preprocess_meta
            
            # Extract features for classification
            features = self._extract_features(ball_results, pose_results, frames)
            
            return features, preprocess_meta
            
        except Exception as e:
            log.error(f"Shot classification preprocessing failed: {e}")
            raise
    
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """
        Classify shots based on extracted features.
        """
        if preprocessed_data is None:
            return None
        
        try:
            if self.use_ml_model:
                # ML-based classification
                with torch.no_grad():
                    features_tensor = torch.tensor(preprocessed_data, dtype=torch.float32).to(self.device)
                    predictions = self.model(features_tensor)
                    shot_probs = torch.softmax(predictions, dim=-1)
                    return shot_probs.cpu().numpy()
            else:
                # Rule-based classification
                return self._rule_based_classification(preprocessed_data, metadata)
                
        except Exception as e:
            log.error(f"Shot classification inference failed: {e}")
            raise
    
    def postprocess(self, inference_outputs: Any, metadata: Dict) -> Dict[str, Any]:
        """
        Structure shot classification results.
        """
        if inference_outputs is None:
            batch_size = metadata.get('batch_size', 1)
            return {
                'batch_results': [{'shot_type': 'unknown', 'confidence': 0.0, 'probabilities': {}}] * batch_size,
                'batch_size': batch_size,
                'classification_summary': {
                    'total_shots': 0,
                    'shot_distribution': {},
                    'avg_confidence': 0.0
                }
            }
        
        try:
            batch_results = []
            shot_counts = {shot: 0 for shot in self.shot_types}
            confidences = []
            
            for frame_predictions in inference_outputs:
                if isinstance(frame_predictions, dict):
                    # Rule-based output
                    shot_type = frame_predictions['shot_type']
                    confidence = frame_predictions['confidence']
                    probabilities = frame_predictions.get('probabilities', {})
                else:
                    # ML model output
                    shot_idx = np.argmax(frame_predictions)
                    confidence = float(frame_predictions[shot_idx])
                    shot_type = self.shot_types[shot_idx] if confidence >= self.confidence_threshold else 'unknown'
                    probabilities = {self.shot_types[i]: float(frame_predictions[i]) for i in range(len(self.shot_types))}
                
                frame_result = {
                    'shot_type': shot_type,
                    'confidence': confidence,
                    'probabilities': probabilities
                }
                
                batch_results.append(frame_result)
                
                if shot_type != 'unknown':
                    shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1
                    confidences.append(confidence)
            
            classification_summary = {
                'total_shots': sum(shot_counts.values()),
                'shot_distribution': shot_counts,
                'avg_confidence': np.mean(confidences) if confidences else 0.0
            }
            
            return {
                'batch_results': batch_results,
                'batch_size': len(batch_results),
                'classification_summary': classification_summary
            }
            
        except Exception as e:
            log.error(f"Shot classification postprocessing failed: {e}")
            raise
    
    def visualize(self, frame: Any, results: Dict[str, Any], vis_config: Dict) -> Any:
        """
        Draw shot classification visualization.
        """
        try:
            output_frame = frame.copy()
            
            if 'batch_results' in results and len(results['batch_results']) > 0:
                frame_result = results['batch_results'][0]
                
                shot_type = frame_result['shot_type']
                confidence = frame_result['confidence']
                
                # Draw shot classification
                if shot_type != 'unknown':
                    color = [0, 255, 0] if confidence >= self.confidence_threshold else [0, 255, 255]
                    text = f"Shot: {shot_type.upper()} ({confidence:.2f})"
                    
                    cv2.putText(
                        output_frame,
                        text,
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )
                    
                    # Draw probability distribution
                    y_offset = 180
                    for shot, prob in frame_result['probabilities'].items():
                        prob_text = f"{shot}: {prob:.2f}"
                        cv2.putText(
                            output_frame,
                            prob_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255, 255, 255],
                            1
                        )
                        y_offset += 20
            
            return output_frame
            
        except Exception as e:
            log.warning(f"Shot classification visualization failed: {e}")
            return frame
    
    def _extract_features(self, ball_results: Dict, pose_results: Dict, frames: List) -> List[Dict]:
        """
        Extract features from ball tracking and pose estimation results.
        """
        features = []
        
        ball_batch = ball_results.get('batch_results', [])
        pose_batch = pose_results.get('batch_results', [])
        
        for i, frame in enumerate(frames):
            ball_data = ball_batch[i] if i < len(ball_batch) else {}
            pose_data = pose_batch[i] if i < len(pose_batch) else []
            
            frame_features = {
                # Ball features
                'ball_visible': ball_data.get('visible', False),
                'ball_x': ball_data.get('x', -1),
                'ball_y': ball_data.get('y', -1),
                'ball_speed': self._estimate_ball_speed(ball_data),
                
                # Pose features
                'num_players': len(pose_data),
                'player_poses': self._extract_pose_features(pose_data),
                
                # Frame index for temporal analysis
                'frame_idx': i
            }
            
            features.append(frame_features)
        
        return features
    
    def _estimate_ball_speed(self, ball_data: Dict) -> float:
        """Estimate ball speed from tracking data."""
        # Simple speed estimation (in production, use ball history)
        return ball_data.get('tracking_confidence', 0.0) * 10.0
    
    def _extract_pose_features(self, pose_data: List[Dict]) -> List[Dict]:
        """Extract relevant pose features for shot classification."""
        pose_features = []
        
        for pose in pose_data:
            if pose and 'keypoints' in pose:
                keypoints = pose['keypoints']
                
                # Extract key joint positions and angles
                features = {
                    'right_wrist': keypoints[10] if len(keypoints) > 10 else [0, 0],
                    'left_wrist': keypoints[9] if len(keypoints) > 9 else [0, 0],
                    'right_elbow': keypoints[8] if len(keypoints) > 8 else [0, 0],
                    'left_elbow': keypoints[7] if len(keypoints) > 7 else [0, 0],
                    'confidence': pose.get('confidence', 0.0)
                }
                pose_features.append(features)
        
        return pose_features
    
    def _rule_based_classification(self, features: List[Dict], metadata: Dict) -> List[Dict]:
        """
        Simple rule-based shot classification.
        In production, replace with trained ML model.
        """
        results = []
        
        for frame_features in features:
            shot_type = 'unknown'
            confidence = 0.0
            probabilities = {shot: 0.0 for shot in self.shot_types}
            
            # Simple heuristics
            if frame_features['ball_visible'] and frame_features['num_players'] > 0:
                ball_y = frame_features['ball_y']
                ball_speed = frame_features['ball_speed']
                
                # Very basic classification rules
                if ball_y < 200:  # Ball high in frame
                    if ball_speed > 8:
                        shot_type = 'smash'
                        confidence = 0.8
                    else:
                        shot_type = 'serve'
                        confidence = 0.7
                elif ball_speed > 6:
                    shot_type = 'forehand'  # Default for fast shots
                    confidence = 0.6
                elif ball_speed > 3:
                    shot_type = 'backhand'
                    confidence = 0.5
                else:
                    shot_type = 'volley'
                    confidence = 0.4
                
                probabilities[shot_type] = confidence
            
            results.append({
                'shot_type': shot_type,
                'confidence': confidence,
                'probabilities': probabilities
            })
        
        return results
    
    def get_output_keys(self) -> List[str]:
        """Get list of output keys this task produces."""
        return ['batch_results', 'batch_size', 'classification_summary']
    
    def requires_batch_processing(self) -> bool:
        """Shot classification supports batch processing."""
        return True