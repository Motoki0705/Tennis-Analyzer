"""
Pose estimation task for the flexible tennis analysis pipeline - updated version.
"""
import torch
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import logging

from ..core.base_task import BaseTask
from src.pose.pipeline.pipeline_module import PosePreprocessor, PoseEstimator, PosePostprocessor
from src.pose.pipeline.drawing_utils import draw_results_on_frame

log = logging.getLogger(__name__)


class PoseEstimationTask(BaseTask):
    """
    Tennis player pose estimation task using VitPose.
    Depends on player detection results.
    Updated to match original pipeline behavior.
    """
    
    def initialize(self) -> None:
        """Initialize pose estimation models and processors."""
        try:
            # Initialize components
            self.preprocessor = PosePreprocessor()
            self.estimator = PoseEstimator(self.device)
            self.postprocessor = PosePostprocessor()
            
            # Cache configuration
            self.keypoint_threshold = self.config.get('keypoint_threshold', 0.3)
            self.model_name = self.config.get('model_name', "usyd-community/vitpose-base-simple")
            
            log.info(f"Pose estimation task initialized with model {self.model_name}")
            
        except Exception as e:
            log.error(f"Failed to initialize pose estimation task: {e}")
            raise
    
    def preprocess(self, frames: List[Any], metadata: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Preprocess frames and player detections for pose estimation.
        Simplified approach - just pass through frames and dependency data.
        """
        try:
            # Get player detection results from dependencies
            dependencies = metadata.get('dependencies', {}) if metadata else {}
            player_results = dependencies.get('player_detection')
            
            if not player_results or 'batch_results' not in player_results:
                # No player detections available
                return None, {
                    'batch_size': len(frames),
                    'player_detections_available': False,
                    'original_frames': frames,
                    'dependencies': dependencies,
                    'task_config': {
                        'keypoint_threshold': self.keypoint_threshold
                    }
                }
            
            # Pass through frames and dependency data for frame-by-frame processing in inference
            preprocess_meta = {
                'batch_size': len(frames),
                'player_detections_available': True,
                'original_frames': frames,
                'dependencies': dependencies,
                'task_config': {
                    'keypoint_threshold': self.keypoint_threshold
                }
            }
            
            return frames, preprocess_meta  # Just pass through frames
            
        except Exception as e:
            log.error(f"Pose preprocessing failed: {e}")
            raise
    
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """
        Run pose estimation inference frame by frame (like original pipeline).
        """
        if not metadata.get('player_detections_available', False):
            return None
        
        try:
            # Get data from metadata
            original_frames = metadata.get('original_frames', [])
            dependencies = metadata.get('dependencies', {})
            player_results = dependencies.get('player_detection', {})
            
            if not player_results or 'batch_results' not in player_results:
                return None
            
            player_batch_results = player_results['batch_results']
            pose_results = {}
            
            # Process frame by frame like original pipeline
            for frame_idx, (frame, player_detections) in enumerate(zip(original_frames, player_batch_results)):
                if player_detections['count'] > 0:
                    # Convert to expected format
                    detections = {
                        'boxes': np.array(player_detections['boxes'], dtype=np.float32),
                        'scores': np.array(player_detections['scores'], dtype=np.float32)
                    }
                    
                    # Process frame
                    pose_inputs, pose_meta = self.preprocessor.process_frame(frame, detections)
                    if pose_inputs is not None:
                        with torch.no_grad():
                            pose_outputs = self.estimator.predict(pose_inputs)
                            pose_results[frame_idx] = {
                                'outputs': pose_outputs,
                                'meta': pose_meta,
                                'detections': detections
                            }
            
            return pose_results
            
        except Exception as e:
            log.error(f"Pose inference failed: {e}")
            raise
    
    def postprocess(self, inference_outputs: Any, metadata: Dict) -> Dict[str, Any]:
        """
        Post-process pose estimation outputs (already processed per frame).
        """
        if inference_outputs is None:
            # No pose inputs available
            batch_size = metadata.get('batch_size', 1)
            return {
                'batch_results': [[] for _ in range(batch_size)],
                'batch_size': batch_size,
                'detection_summary': {
                    'total_poses': 0,
                    'avg_poses_per_frame': 0.0,
                    'avg_keypoint_confidence': 0.0
                }
            }
        
        try:
            # Initialize results for each frame
            batch_size = metadata.get('batch_size', 1)
            batch_results = [[] for _ in range(batch_size)]
            
            # Process each frame's pose results
            for frame_idx, frame_data in inference_outputs.items():
                try:
                    pose_outputs = frame_data['outputs']
                    pose_meta = frame_data['meta']
                    
                    # Postprocess using original postprocessor
                    pose_result = self.postprocessor.process_frame(pose_outputs, pose_meta)
                    
                    if pose_result and len(pose_result) > 0:
                        for person_result in pose_result:
                            keypoints = person_result['keypoints']
                            scores = person_result['scores']
                            
                            valid_keypoints = []
                            valid_scores = []
                            
                            for kp, score in zip(keypoints, scores):
                                if score >= self.keypoint_threshold:
                                    valid_keypoints.append(kp)
                                    valid_scores.append(score)
                                else:
                                    valid_keypoints.append([0, 0])
                                    valid_scores.append(0.0)
                            
                            processed_result = {
                                'keypoints': valid_keypoints,
                                'scores': valid_scores,
                                'valid_count': sum(1 for s in valid_scores if s >= self.keypoint_threshold),
                                'confidence': np.mean(valid_scores) if valid_scores else 0.0
                            }
                            
                            batch_results[frame_idx].append(processed_result)
                            
                except Exception as e:
                    log.debug(f"Pose postprocessing failed for frame {frame_idx}: {e}")
                    continue
            
            # Calculate summary statistics
            total_poses = sum(len(frame_poses) for frame_poses in batch_results)
            all_confidences = []
            
            for frame_poses in batch_results:
                for pose in frame_poses:
                    all_confidences.extend(pose['scores'])
            
            avg_keypoint_confidence = np.mean(all_confidences) if all_confidences else 0.0
            
            return {
                'batch_results': batch_results,
                'batch_size': batch_size,
                'detection_summary': {
                    'total_poses': total_poses,
                    'avg_poses_per_frame': total_poses / batch_size if batch_size > 0 else 0.0,
                    'avg_keypoint_confidence': avg_keypoint_confidence
                }
            }
            
        except Exception as e:
            log.error(f"Pose postprocessing failed: {e}")
            raise
    
    def visualize(self, frame: Any, results: Dict[str, Any], vis_config: Dict) -> Any:
        """
        Draw pose estimation visualization on frame.
        """
        try:
            # Get visualization config with defaults
            keypoint_color = vis_config.get('keypoint_color', [0, 255, 255])  # Cyan
            keypoint_radius = vis_config.get('keypoint_radius', 3)
            skeleton_color = vis_config.get('skeleton_color', [255, 0, 255])  # Magenta
            skeleton_thickness = vis_config.get('skeleton_thickness', 2)
            score_threshold = vis_config.get('score_threshold', self.keypoint_threshold)
            
            output_frame = frame.copy()
            
            # Get results for the first frame in batch (assuming single frame visualization)
            if 'batch_results' in results and len(results['batch_results']) > 0:
                frame_poses = results['batch_results'][0]
                
                # Draw each pose
                for pose_idx, pose in enumerate(frame_poses):
                    keypoints = np.array(pose['keypoints'])
                    scores = np.array(pose['scores'])
                    
                    # Draw keypoints
                    for kp, score in zip(keypoints, scores):
                        if score >= score_threshold:
                            cv2.circle(output_frame, tuple(map(int, kp)), 
                                     keypoint_radius, keypoint_color, -1)
                    
                    # Draw skeleton connections (simplified)
                    # COCO 17 keypoint connections
                    connections = [
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
                        (5, 11), (6, 12)  # Body
                    ]
                    
                    for start_idx, end_idx in connections:
                        if (start_idx < len(scores) and end_idx < len(scores) and
                            scores[start_idx] >= score_threshold and scores[end_idx] >= score_threshold):
                            start_pt = tuple(map(int, keypoints[start_idx]))
                            end_pt = tuple(map(int, keypoints[end_idx]))
                            cv2.line(output_frame, start_pt, end_pt, skeleton_color, skeleton_thickness)
                
                # Draw summary
                pose_count = len(frame_poses)
                cv2.putText(
                    output_frame,
                    f"Poses: {pose_count}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    keypoint_color,
                    2
                )
            
            return output_frame
            
        except Exception as e:
            log.warning(f"Pose visualization failed: {e}")
            return frame
    
    def get_output_keys(self) -> List[str]:
        """Get list of output keys this task produces."""
        return ['batch_results', 'batch_size', 'detection_summary']
    
    def requires_batch_processing(self) -> bool:
        """Pose estimation supports batch processing."""
        return True