"""
Ball tracking task with temporal tracking capabilities.
"""
import torch
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import logging
from collections import deque

from ..core.base_task import BaseTask

log = logging.getLogger(__name__)


class BallTrackingTask(BaseTask):
    """
    Advanced ball tracking task with temporal consistency.
    Extends basic ball detection with tracking capabilities.
    """
    
    def initialize(self) -> None:
        """Initialize ball tracking system."""
        try:
            # Import ball detection components
            from src.ball.pipeline.wasb_modules import load_default_config
            from src.ball.pipeline.wasb_modules.pipeline_modules import BallPreprocessor, BallDetector, DetectionPostprocessor
            
            # Load configuration
            self.ball_cfg = load_default_config()
            model_path = self.config.get('model_path')
            if model_path:
                self.ball_cfg['detector']['model_path'] = model_path
            
            score_threshold = self.config.get('score_threshold', 0.3)
            self.ball_cfg['detector']['postprocessor']['score_threshold'] = score_threshold
            
            # Initialize components
            self.preprocessor = BallPreprocessor(self.ball_cfg)
            self.detector = BallDetector(self.ball_cfg, self.device)
            self.postprocessor = DetectionPostprocessor(self.ball_cfg)
            
            # Tracking parameters
            self.max_tracking_distance = self.config.get('max_tracking_distance', 100)
            self.tracking_history_length = self.config.get('tracking_history_length', 10)
            self.min_detection_confidence = self.config.get('min_detection_confidence', 0.2)
            
            # Tracking state
            self.ball_history = deque(maxlen=self.tracking_history_length)
            self.frame_history = deque(maxlen=self.preprocessor.frames_in + 180) # 180 is buffer
            self.last_known_position = None
            self.tracking_id = 0
            self.frames_since_detection = 0
            
            log.info(f"Ball tracking task initialized with tracking enabled")
            
        except Exception as e:
            log.error(f"Failed to initialize ball tracking task: {e}")
            raise
    
    def preprocess(self, frames: List[Any], metadata: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Preprocess frames for ball tracking."""
        try:
            # Update frame history
            for frame in frames:
                self.frame_history.append(frame)
            
            # Prepare frame sequences
            ball_sequences = []
            if len(self.frame_history) >= self.preprocessor.frames_in:
                for i in range(len(frames)):
                    end_idx = len(self.frame_history) - len(frames) + i + 1
                    start_idx = max(0, end_idx - self.preprocessor.frames_in)
                    
                    if end_idx - start_idx == self.preprocessor.frames_in:
                        sequence = list(self.frame_history)[start_idx:end_idx]
                        ball_sequences.append(sequence)
            
            if not ball_sequences:
                return None, {
                    'batch_size': len(frames),
                    'sequences_available': False,
                    'tracking_state': self._get_tracking_state()
                }
            
            # Process sequences
            preprocessed_data, ball_meta = self.preprocessor.process_batch(ball_sequences)
            
            preprocess_meta = {
                'batch_size': len(frames),
                'sequences_available': True,
                'ball_meta': ball_meta,
                'tracking_state': self._get_tracking_state(),
                'task_config': {
                    'score_threshold': self.ball_cfg['detector']['postprocessor']['score_threshold'],
                    'tracking_enabled': True
                }
            }
            
            return preprocessed_data, preprocess_meta
            
        except Exception as e:
            log.error(f"Ball tracking preprocessing failed: {e}")
            raise
    
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """Run ball detection and apply tracking."""
        if not metadata.get('sequences_available', False):
            return None
        
        try:
            # Ball detection
            with torch.no_grad():
                raw_outputs = self.detector.predict_batch(preprocessed_data)
            
            return raw_outputs
            
        except Exception as e:
            log.error(f"Ball tracking inference failed: {e}")
            raise
    
    def postprocess(self, inference_outputs: Any, metadata: Dict) -> Dict[str, Any]:
        """Structure tracked ball detection results."""
        if inference_outputs is None:
            batch_size = metadata.get('batch_size', 1)
            return {
                'batch_results': [{'visible': False, 'x': -1, 'y': -1, 'score': 0.0, 'tracking_id': -1, 'tracking_confidence': 0.0}] * batch_size,
                'batch_size': batch_size,
                'tracking_summary': {
                    'active_tracks': 0,
                    'detection_rate': 0.0,
                    'tracking_stability': 0.0
                }
            }
        
        try:
            # Postprocess detections
            ball_meta = metadata.get('ball_meta', {})
            ball_detections = self.postprocessor.process_batch(inference_outputs, ball_meta, self.device)
            
            # Apply tracking
            tracked_detections = self._apply_tracking(ball_detections, metadata)
            batch_results = []
            for detection in tracked_detections:
                frame_result = {
                    'visible': detection.get('visi', False),
                    'x': float(detection.get('x', -1)),
                    'y': float(detection.get('y', -1)),
                    'score': float(detection.get('score', 0.0)),
                    'tracking_id': detection.get('tracking_id', -1),
                    'tracking_confidence': detection.get('tracking_confidence', 0.0),
                    'predicted': detection.get('predicted', False)  # Whether position was predicted vs detected
                }
                batch_results.append(frame_result)
            
            # Calculate tracking statistics
            visible_detections = [r for r in batch_results if r['visible']]
            tracked_detections = [r for r in batch_results if r['tracking_id'] >= 0]
            
            tracking_summary = {
                'active_tracks': len(set(r['tracking_id'] for r in tracked_detections if r['tracking_id'] >= 0)),
                'detection_rate': len(visible_detections) / len(batch_results) if batch_results else 0.0,
                'tracking_stability': len(tracked_detections) / len(batch_results) if batch_results else 0.0
            }
            
            return {
                'batch_results': batch_results,
                'batch_size': len(batch_results),
                'tracking_summary': tracking_summary
            }
            
        except Exception as e:
            log.error(f"Ball tracking postprocessing failed: {e}")
            raise
    
    def visualize(self, frame: Any, results: Dict[str, Any], vis_config: Dict) -> Any:
        """Draw ball tracking visualization."""
        try:
            ball_color = vis_config.get('color', [0, 0, 255])  # Red
            ball_radius = vis_config.get('radius', 8)
            trail_color = vis_config.get('trail_color', [255, 255, 0])  # Yellow
            show_trail = vis_config.get('show_trail', True)
            show_prediction = vis_config.get('show_prediction', True)
            
            output_frame = frame.copy()
            
            if 'batch_results' in results and len(results['batch_results']) > 0:
                frame_result = results['batch_results'][0]
                
                if frame_result['visible']:
                    x, y = int(frame_result['x']), int(frame_result['y'])
                    score = frame_result['score']
                    tracking_id = frame_result['tracking_id']
                    is_predicted = frame_result.get('predicted', False)
                    
                    # Choose color based on detection vs prediction
                    color = trail_color if is_predicted else ball_color
                    
                    # Draw ball
                    cv2.circle(output_frame, (x, y), ball_radius, color, -1)
                    
                    # Draw tracking info
                    info_text = f"ID:{tracking_id} {score:.2f}"
                    if is_predicted:
                        info_text += " (PRED)"
                    
                    cv2.putText(output_frame, info_text, (x+10, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw trail
                if show_trail and len(self.ball_history) > 1:
                    trail_points = [(int(h['x']), int(h['y'])) for h in self.ball_history if h['visible']]
                    for i in range(1, len(trail_points)):
                        cv2.line(output_frame, trail_points[i-1], trail_points[i], trail_color, 2)
                
                # Draw tracking summary
                tracking_summary = results.get('tracking_summary', {})
                cv2.putText(
                    output_frame,
                    f"Tracks: {tracking_summary.get('active_tracks', 0)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    ball_color,
                    2
                )
            
            return output_frame
            
        except Exception as e:
            log.warning(f"Ball tracking visualization failed: {e}")
            return frame
    
    def _apply_tracking(self, detections: List[Dict], metadata: Dict) -> List[Dict]:
        """Apply tracking logic to ball detections."""
        tracked_detections = []
        
        for detection in detections:
            if detection['visi'] and detection['score'] >= self.min_detection_confidence:
                # Valid detection
                current_pos = (detection['x'], detection['y'])
                
                # Associate with existing track or create new one
                tracking_id, tracking_confidence = self._associate_detection(current_pos)
                
                tracked_detection = detection.copy()
                tracked_detection['tracking_id'] = tracking_id
                tracked_detection['tracking_confidence'] = tracking_confidence
                tracked_detection['predicted'] = False
                
                # Update tracking state
                self._update_tracking_state(current_pos, detection['score'])
                self.frames_since_detection = 0
                
            else:
                # No detection - try to predict position
                predicted_pos = self._predict_position()
                
                if predicted_pos is not None:
                    tracked_detection = {
                        'visi': True,
                        'x': predicted_pos[0],
                        'y': predicted_pos[1],
                        'score': max(0.1, 0.8 - self.frames_since_detection * 0.1),
                        'tracking_id': self.tracking_id,
                        'tracking_confidence': max(0.1, 0.9 - self.frames_since_detection * 0.2),
                        'predicted': True
                    }
                    self.frames_since_detection += 1
                else:
                    tracked_detection = detection.copy()
                    tracked_detection['tracking_id'] = -1
                    tracked_detection['tracking_confidence'] = 0.0
                    tracked_detection['predicted'] = False
            
            tracked_detections.append(tracked_detection)
        
        return tracked_detections
    
    def _associate_detection(self, position: Tuple[float, float]) -> Tuple[int, float]:
        """Associate detection with existing track."""
        if self.last_known_position is None:
            # First detection
            self.tracking_id += 1
            return self.tracking_id, 1.0
        
        # Calculate distance to last known position
        distance = np.sqrt((position[0] - self.last_known_position[0])**2 + 
                          (position[1] - self.last_known_position[1])**2)
        
        if distance <= self.max_tracking_distance:
            # Associate with existing track
            tracking_confidence = max(0.1, 1.0 - distance / self.max_tracking_distance)
            return self.tracking_id, tracking_confidence
        else:
            # Create new track
            self.tracking_id += 1
            return self.tracking_id, 1.0
    
    def _predict_position(self) -> Optional[Tuple[float, float]]:
        """Predict ball position based on history."""
        if len(self.ball_history) < 2 or self.frames_since_detection > 5:
            return None
        
        # Simple linear prediction based on last two positions
        recent_positions = [(h['x'], h['y']) for h in list(self.ball_history)[-2:] if h['visible']]
        
        if len(recent_positions) == 2:
            dx = recent_positions[1][0] - recent_positions[0][0]
            dy = recent_positions[1][1] - recent_positions[0][1]
            
            predicted_x = recent_positions[1][0] + dx * (self.frames_since_detection + 1)
            predicted_y = recent_positions[1][1] + dy * (self.frames_since_detection + 1)
            
            return (predicted_x, predicted_y)
        
        return None
    
    def _update_tracking_state(self, position: Tuple[float, float], score: float):
        """Update internal tracking state."""
        self.last_known_position = position
        self.ball_history.append({
            'x': position[0],
            'y': position[1],
            'score': score,
            'visible': True,
            'timestamp': len(self.ball_history)
        })
    
    def _get_tracking_state(self) -> Dict:
        """Get current tracking state for debugging."""
        return {
            'tracking_id': self.tracking_id,
            'frames_since_detection': self.frames_since_detection,
            'history_length': len(self.ball_history),
            'last_position': self.last_known_position
        }
    
    def get_output_keys(self) -> List[str]:
        """Get list of output keys this task produces."""
        return ['batch_results', 'batch_size', 'tracking_summary']
    
    def requires_batch_processing(self) -> bool:
        """Ball tracking supports batch processing."""
        return True