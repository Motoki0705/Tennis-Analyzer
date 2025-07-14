"""
Player detection task for the flexible tennis analysis pipeline.
"""
import torch
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import logging

from ..core.base_task import BaseTask
from src.player.pipeline.pipeline_module import PlayerPreprocessor, PlayerDetector, PlayerPostprocessor

log = logging.getLogger(__name__)


class PlayerDetectionTask(BaseTask):
    """
    Tennis player detection task using RT-DETR.
    """
    
    def initialize(self) -> None:
        """Initialize player detection models and processors."""
        try:
            # Initialize components
            self.preprocessor = PlayerPreprocessor()
            
            checkpoint_path = self.config.get('checkpoint')
            if not checkpoint_path:
                raise ValueError("Player checkpoint path is required")
            
            self.detector = PlayerDetector(checkpoint_path, self.device)
            
            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            self.postprocessor = PlayerPostprocessor(confidence_threshold)
            
            # Cache configuration
            self.confidence_threshold = confidence_threshold
            
            log.info(f"Player detection task initialized with confidence threshold {confidence_threshold}")
            
        except Exception as e:
            log.error(f"Failed to initialize player detection task: {e}")
            raise
    
    def preprocess(self, frames: List[Any], metadata: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Preprocess frames for player detection.
        
        Args:
            frames: List of input frames (BGR format)
            metadata: Optional metadata
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_metadata)
        """
        try:
            # Process batch
            preprocessed_data, batch_meta_list = self.preprocessor.process_batch(frames)
            
            # Convert to dict format for our task framework
            preprocess_meta = {
                'batch_size': len(frames),
                'original_frames_shape': [frame.shape for frame in frames],
                'batch_meta_list': batch_meta_list,  # Original metadata list
                'task_config': {
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
            return preprocessed_data, preprocess_meta
            
        except Exception as e:
            log.error(f"Player preprocessing failed: {e}")
            raise
    
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """
        Run player detection inference.
        
        Args:
            preprocessed_data: Preprocessed tensor data
            metadata: Preprocessing metadata
            
        Returns:
            Raw model outputs (detections)
        """
        try:
            with torch.no_grad():
                raw_outputs = self.detector.predict(preprocessed_data)
            
            return raw_outputs
            
        except Exception as e:
            log.error(f"Player inference failed: {e}")
            raise
    
    def postprocess(self, raw_outputs: Any, metadata: Dict) -> Dict[str, Any]:
        """
        Post-process player detection outputs.
        
        Args:
            raw_outputs: Raw model outputs
            metadata: Preprocessing metadata
            
        Returns:
            Structured player detection results
        """
        try:
            # Process batch results using original metadata list
            batch_meta_list = metadata.get('batch_meta_list', [])
            player_detections = self.postprocessor.process_batch(raw_outputs, batch_meta_list)
            
            # Structure results for each frame
            batch_results = []
            for detections in player_detections:
                # Check if detections exist and have boxes (handle numpy arrays properly)
                has_detections = False
                if (detections is not None and 
                    isinstance(detections, dict) and 
                    'boxes' in detections):
                    
                    boxes = detections['boxes']
                    # Handle numpy array length check properly
                    if hasattr(boxes, '__len__'):
                        has_detections = len(boxes) > 0
                    elif hasattr(boxes, 'shape'):
                        has_detections = boxes.shape[0] > 0
                
                if has_detections:
                    # Convert numpy arrays to lists for consistency
                    boxes_list = detections['boxes'].tolist() if hasattr(detections['boxes'], 'tolist') else detections['boxes']
                    scores_list = detections['scores'].tolist() if hasattr(detections['scores'], 'tolist') else detections['scores']
                    
                    # Valid detections found
                    frame_result = {
                        'boxes': boxes_list,  # List of [x1, y1, x2, y2]
                        'scores': scores_list,  # List of confidence scores
                        'count': len(boxes_list),
                        'max_confidence': max(scores_list) if scores_list else 0.0
                    }
                else:
                    # No detections
                    frame_result = {
                        'boxes': [],
                        'scores': [],
                        'count': 0,
                        'max_confidence': 0.0
                    }
                
                batch_results.append(frame_result)
            
            # Calculate summary statistics
            total_players = sum(r['count'] for r in batch_results)
            frames_with_players = sum(1 for r in batch_results if r['count'] > 0)
            detection_rate = frames_with_players / len(batch_results) if batch_results else 0.0
            
            avg_confidence = 0.0
            if total_players > 0:
                all_scores = []
                for result in batch_results:
                    all_scores.extend(result['scores'])
                avg_confidence = np.mean(all_scores) if all_scores else 0.0
            
            return {
                'batch_results': batch_results,
                'batch_size': len(batch_results),
                'detection_summary': {
                    'total_players': total_players,
                    'avg_players_per_frame': total_players / len(batch_results) if batch_results else 0.0,
                    'detection_rate': detection_rate,
                    'avg_confidence': avg_confidence
                }
            }
            
        except Exception as e:
            log.error(f"Player postprocessing failed: {e}")
            raise
    
    def visualize(self, frame: Any, results: Dict[str, Any], vis_config: Dict) -> Any:
        """
        Draw player detection visualization on frame.
        
        Args:
            frame: Input frame to draw on
            results: Player detection results
            vis_config: Visualization configuration
            
        Returns:
            Frame with player visualization
        """
        try:
            # Get visualization config with defaults
            bbox_color = vis_config.get('bbox_color', [255, 255, 0])  # Yellow
            bbox_thickness = vis_config.get('bbox_thickness', 2)
            show_score = vis_config.get('show_score', True)
            
            output_frame = frame.copy()
            
            # Get results for the first frame in batch (assuming single frame visualization)
            if 'batch_results' in results and len(results['batch_results']) > 0:
                frame_result = results['batch_results'][0]
                
                boxes = frame_result.get('boxes', [])
                scores = frame_result.get('scores', [])
                
                # Draw each detection
                for i, (box, score) in enumerate(zip(boxes, scores)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
                    
                    # Draw score if enabled
                    if show_score:
                        label = f'Player: {score:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), bbox_color, -1)
                        cv2.putText(output_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Draw summary
                player_count = frame_result['count']
                cv2.putText(
                    output_frame,
                    f"Players: {player_count}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    bbox_color,
                    2
                )
            
            return output_frame
            
        except Exception as e:
            log.warning(f"Player visualization failed: {e}")
            return frame
    
    def get_output_keys(self) -> List[str]:
        """Get list of output keys this task produces."""
        return ['batch_results', 'batch_size', 'detection_summary']
    
    def requires_batch_processing(self) -> bool:
        """Player detection supports batch processing."""
        return True