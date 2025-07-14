"""
Ball tracking task for the flexible tennis analysis pipeline.
"""
import torch
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import logging
from collections import deque

from ..core.base_task import BaseTask
from src.ball.pipeline.wasb_modules import load_default_config
from src.ball.pipeline.wasb_modules.pipeline_modules import BallPreprocessor, BallDetector, DetectionPostprocessor

log = logging.getLogger(__name__)


class BallTrackingTask(BaseTask):
    """
    Tennis ball detection and tracking task.
    """
    
    def initialize(self) -> None:
        """Initialize ball tracking models and processors."""
        try:
            # Load default WASB configuration
            self.ball_cfg = load_default_config()
            
            # Override with task-specific config
            model_path = self.config.get('model_path')
            if model_path:
                self.ball_cfg['detector']['model_path'] = model_path
            
            score_threshold = self.config.get('score_threshold', 0.3)
            self.ball_cfg['detector']['postprocessor']['score_threshold'] = score_threshold
            
            # Initialize components
            self.preprocessor = BallPreprocessor(self.ball_cfg)
            self.detector = BallDetector(self.ball_cfg, self.device)
            self.postprocessor = DetectionPostprocessor(self.ball_cfg)
            
            # Frame history for temporal processing
            max_len = self.preprocessor.frames_in + 180 # 180 is buffer
            self.frame_history = deque(maxlen=max_len)
            
            # Tracking settings
            self.tracking_enabled = self.config.get('tracking_enabled', False)
            self.score_threshold = score_threshold
            
            log.info(f"Ball tracking task initialized (tracking: {self.tracking_enabled})")
            
        except Exception as e:
            log.error(f"Failed to initialize ball tracking task: {e}")
            raise
    
    def preprocess(self, frames: List[Any], metadata: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Preprocess frames for ball detection.
        
        Args:
            frames: List of input frames (BGR format)
            metadata: Optional metadata
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_metadata)
        """
        try:
            # Update frame history
            for frame in frames:
                self.frame_history.append(frame)
            
            # Prepare frame sequences for ball detection
            ball_sequences = []
            if len(self.frame_history) >= self.preprocessor.frames_in:
                # Create sequences for each frame in the batch
                for i in range(len(frames)):
                    # Get the sequence ending at current frame
                    end_idx = len(self.frame_history) - len(frames) + i + 1
                    start_idx = max(0, end_idx - self.preprocessor.frames_in)
                    
                    if end_idx - start_idx == self.preprocessor.frames_in:
                        sequence = list(self.frame_history)[start_idx:end_idx]
                        ball_sequences.append(sequence)
            
            if not ball_sequences:
                # Not enough frames for detection
                return None, {
                    'batch_size': len(frames),
                    'sequences_available': False,
                    'frames_needed': self.preprocessor.frames_in,
                    'frames_available': len(self.frame_history)
                }
            
            # Process sequences
            preprocessed_data, ball_meta = self.preprocessor.process_batch(ball_sequences)
            
            # Convert to consistent format
            preprocess_meta = {
                'batch_size': len(frames),
                'sequences_count': len(ball_sequences),
                'sequences_available': True,
                'ball_meta': ball_meta,  # Original ball preprocessing metadata
                'task_config': {
                    'score_threshold': self.score_threshold,
                    'tracking_enabled': self.tracking_enabled,
                    'frames_in': self.preprocessor.frames_in
                }
            }
            
            return preprocessed_data, preprocess_meta
            
        except Exception as e:
            log.error(f"Ball preprocessing failed: {e}")
            raise
    
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """
        Run ball detection inference.
        
        Args:
            preprocessed_data: Preprocessed tensor data
            metadata: Preprocessing metadata
            
        Returns:
            Raw model outputs.
        """
        if not metadata.get('sequences_available', False):
            return None
        
        try:
            with torch.no_grad():
                raw_outputs = self.detector.predict_batch(preprocessed_data)
            
            # MODIFIED: The postprocessing call is moved to the postprocess method.
            # Now, this method returns only the raw outputs from the model.
            return raw_outputs
            
        except Exception as e:
            log.error(f"Ball inference failed: {e}")
            raise
    
    def postprocess(self, inference_outputs: Any, metadata: Dict) -> Dict[str, Any]:
        """
        Post-process raw model outputs and structure ball detection results.
        
        Args:
            inference_outputs: Raw model outputs from inference stage.
            metadata: Preprocessing metadata
            
        Returns:
            Structured ball detection results
        """
        if inference_outputs is None:
            # No sequences available for detection
            batch_size = metadata.get('batch_size', 1)
            return {
                'batch_results': [{'visible': False, 'x': -1, 'y': -1, 'score': 0.0}] * batch_size,
                'batch_size': batch_size,
                'detection_summary': {
                    'detection_rate': 0.0,
                    'avg_confidence': 0.0,
                    'sequences_processed': 0
                }
            }
        
        try:
            # MODIFIED: Perform postprocessing on the raw outputs here.
            # This was previously in the inference method.
            ball_meta = metadata.get('ball_meta', {})
            ball_detections = self.postprocessor.process_batch(inference_outputs, ball_meta, self.device)
            
            # The rest of the logic now uses the processed `ball_detections`.
            
            # Structure results for each frame
            batch_results = []
            for detection in ball_detections:
                if detection['visi']:
                    frame_result = {
                        'visible': True,
                        'x': float(detection['x']),
                        'y': float(detection['y']),
                        'score': float(detection['score'])
                    }
                else:
                    frame_result = {
                        'visible': False,
                        'x': -1,
                        'y': -1,
                        'score': 0.0
                    }
                
                batch_results.append(frame_result)
            
            # Pad results to match batch size if needed
            batch_size = metadata.get('batch_size', len(batch_results))
            while len(batch_results) < batch_size:
                batch_results.append({'visible': False, 'x': -1, 'y': -1, 'score': 0.0})
            
            # Calculate summary statistics
            visible_detections = [r for r in batch_results if r['visible']]
            detection_rate = len(visible_detections) / len(batch_results) if batch_results else 0.0
            avg_confidence = np.mean([r['score'] for r in visible_detections]) if visible_detections else 0.0
            
            return {
                'batch_results': batch_results,
                'batch_size': len(batch_results),
                'detection_summary': {
                    'detection_rate': detection_rate,
                    'avg_confidence': avg_confidence,
                    'sequences_processed': metadata.get('sequences_count', 0)
                }
            }
            
        except Exception as e:
            log.error(f"Ball postprocessing failed: {e}")
            raise
    
    def visualize(self, frame: Any, results: Dict[str, Any], vis_config: Dict) -> Any:
        """
        Draw ball detection visualization on frame.
        
        Args:
            frame: Input frame to draw on
            results: Ball detection results
            vis_config: Visualization configuration
            
        Returns:
            Frame with ball visualization
        """
        try:
            # Get visualization config with defaults
            ball_color = vis_config.get('color', [0, 0, 255])  # Red
            ball_radius = vis_config.get('radius', 8)
            show_score = vis_config.get('show_score', True)
            
            output_frame = frame.copy()
            
            # Get results for the first frame in batch (assuming single frame visualization)
            if 'batch_results' in results and len(results['batch_results']) > 0:
                frame_result = results['batch_results'][0]
                
                if frame_result['visible']:
                    x = int(frame_result['x'])
                    y = int(frame_result['y'])
                    score = frame_result['score']
                    
                    # Draw ball circle
                    cv2.circle(output_frame, (x, y), ball_radius, ball_color, -1)
                    
                    # Draw score if enabled
                    if show_score:
                        cv2.putText(
                            output_frame,
                            f'{score:.2f}',
                            (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            ball_color,
                            1
                        )
                
                # Draw detection summary
                detection_rate = results.get('detection_summary', {}).get('detection_rate', 0.0)
                cv2.putText(
                    output_frame,
                    f"Ball: {detection_rate:.1%}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    ball_color,
                    2
                )
            
            return output_frame
            
        except Exception as e:
            log.warning(f"Ball visualization failed: {e}")
            return frame
    
    def get_output_keys(self) -> List[str]:
        """Get list of output keys this task produces."""
        return ['batch_results', 'batch_size', 'detection_summary']
    
    def requires_batch_processing(self) -> bool:
        """Ball detection supports batch processing."""
        return True