"""
Court detection task for the flexible tennis analysis pipeline.
"""
import torch
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import logging

from ..core.base_task import BaseTask
from src.court.pipeline.pipeline_modules import CourtPreprocessor, CourtDetector, CourtPostprocessor
from src.court.pipeline.drawing_utils import draw_keypoints_on_frame, draw_court_skeleton

log = logging.getLogger(__name__)


class CourtDetectionTask(BaseTask):
    """
    Tennis court detection and keypoint extraction task.
    """
    
    def initialize(self) -> None:
        """Initialize court detection models and processors."""
        try:
            # Initialize components
            input_size = tuple(self.config.get('input_size', [512, 288]))
            self.preprocessor = CourtPreprocessor(input_size=input_size)
            
            checkpoint_path = self.config.get('checkpoint')
            if not checkpoint_path:
                raise ValueError("Court checkpoint path is required")
            
            self.detector = CourtDetector(checkpoint_path, self.device)
            
            multi_channel = self.config.get('multi_channel', False)
            self.postprocessor = CourtPostprocessor(multi_channel=multi_channel)
            
            # Cache configuration
            self.score_threshold = self.config.get('score_threshold', 0.5)
            
            log.info(f"Court detection task initialized with input size {input_size}")
            
        except Exception as e:
            log.error(f"Failed to initialize court detection task: {e}")
            raise
    
    def preprocess(self, frames: List[Any], metadata: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Preprocess frames for court detection.
        
        Args:
            frames: List of input frames (BGR format)
            metadata: Optional metadata
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_metadata)
        """
        try:
            # Convert BGR to RGB for processing
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            
            # Process batch
            preprocessed_data, batch_meta_list = self.preprocessor.process_batch(rgb_frames)
            
            # Convert list to dict format for our task framework
            preprocess_meta = {
                'batch_size': len(frames),
                'original_frames_shape': [frame.shape for frame in frames],
                'batch_meta_list': batch_meta_list,  # Original metadata list
                'task_config': {
                    'input_size': (self.preprocessor.input_height, self.preprocessor.input_width),
                    'score_threshold': self.score_threshold
                }
            }
            
            return preprocessed_data, preprocess_meta
            
        except Exception as e:
            log.error(f"Court preprocessing failed: {e}")
            raise
    
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """
        Run court detection inference.
        
        Args:
            preprocessed_data: Preprocessed tensor data
            metadata: Preprocessing metadata
            
        Returns:
            Raw model outputs (heatmaps)
        """
        try:
            with torch.no_grad():
                raw_outputs = self.detector.predict(preprocessed_data)
            
            return raw_outputs
            
        except Exception as e:
            log.error(f"Court inference failed: {e}")
            raise
    
    def postprocess(self, raw_outputs: Any, metadata: Dict) -> Dict[str, Any]:
        """
        Post-process court detection outputs.
        
        Args:
            raw_outputs: Raw model outputs
            metadata: Preprocessing metadata
            
        Returns:
            Structured court detection results
        """
        try:
            # Process batch results using original metadata list
            batch_meta_list = metadata.get('batch_meta_list', [])
            court_results = self.postprocessor.process_batch(raw_outputs, batch_meta_list)
            
            # Structure results for each frame
            batch_results = []
            for i, result in enumerate(court_results):
                if result is not None:
                    # Filter keypoints by score threshold
                    valid_keypoints = []
                    valid_scores = []
                    
                    for j, (keypoint, score) in enumerate(zip(result['keypoints'], result['scores'])):
                        if score >= self.score_threshold:
                            valid_keypoints.append(keypoint)
                            valid_scores.append(score)
                        else:
                            valid_keypoints.append([0, 0])  # Invalid keypoint
                            valid_scores.append(0.0)
                    
                    frame_result = {
                        'keypoints': valid_keypoints,
                        'scores': valid_scores,
                        'valid_count': sum(1 for s in valid_scores if s >= self.score_threshold),
                        'confidence': np.mean(valid_scores) if valid_scores else 0.0
                    }
                else:
                    # No detection
                    frame_result = {
                        'keypoints': [[0, 0]] * 14,  # 14 court keypoints
                        'scores': [0.0] * 14,
                        'valid_count': 0,
                        'confidence': 0.0
                    }
                
                batch_results.append(frame_result)
            
            return {
                'batch_results': batch_results,
                'batch_size': len(batch_results),
                'detection_summary': {
                    'avg_confidence': np.mean([r['confidence'] for r in batch_results]),
                    'detection_rate': sum(1 for r in batch_results if r['valid_count'] > 0) / len(batch_results)
                }
            }
            
        except Exception as e:
            log.error(f"Court postprocessing failed: {e}")
            raise
    
    def visualize(self, frame: Any, results: Dict[str, Any], vis_config: Dict) -> Any:
        """
        Draw court detection visualization on frame.
        
        Args:
            frame: Input frame to draw on
            results: Court detection results
            vis_config: Visualization configuration
            
        Returns:
            Frame with court visualization
        """
        try:
            # Get visualization config with defaults
            keypoint_color = vis_config.get('keypoint_color', [0, 255, 0])
            keypoint_radius = vis_config.get('keypoint_radius', 3)
            skeleton_color = vis_config.get('skeleton_color', [255, 0, 0])
            skeleton_thickness = vis_config.get('skeleton_thickness', 2)
            score_threshold = vis_config.get('score_threshold', self.score_threshold)
            
            output_frame = frame.copy()
            
            # Get results for the first frame in batch (assuming single frame visualization)
            if 'batch_results' in results and len(results['batch_results']) > 0:
                frame_result = results['batch_results'][0]
                
                keypoints = np.array(frame_result['keypoints'])
                scores = np.array(frame_result['scores'])
                
                # Draw keypoints (function only accepts frame, keypoints, scores, threshold)
                draw_keypoints_on_frame(
                    output_frame, 
                    keypoints, 
                    scores, 
                    score_threshold
                )
                
                # Draw court skeleton (function only accepts frame, keypoints, scores, threshold)
                draw_court_skeleton(
                    output_frame,
                    keypoints,
                    scores,
                    score_threshold
                )
                
                # Draw confidence score
                confidence = frame_result['confidence']
                cv2.putText(
                    output_frame,
                    f"Court: {confidence:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    keypoint_color,
                    2
                )
            
            return output_frame
            
        except Exception as e:
            log.warning(f"Court visualization failed: {e}")
            return frame
    
    def get_output_keys(self) -> List[str]:
        """Get list of output keys this task produces."""
        return ['batch_results', 'batch_size', 'detection_summary']
    
    def requires_batch_processing(self) -> bool:
        """Court detection supports batch processing."""
        return True