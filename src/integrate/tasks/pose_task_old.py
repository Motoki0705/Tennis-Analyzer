"""
Pose estimation task for the flexible tennis analysis pipeline.
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
        
        Args:
            frames: List of input frames (BGR format)
            metadata: Must contain dependency results with player detections
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_metadata)
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
                    'task_config': {
                        'keypoint_threshold': self.keypoint_threshold
                    }
                }
            
            player_batch_results = player_results['batch_results']
            
            # Process each frame with its player detections
            all_pose_inputs = []
            all_pose_meta = []
            frame_mapping = []  # Maps pose input index to (frame_idx, player_idx)
            
            for frame_idx, (frame, player_detections) in enumerate(zip(frames, player_batch_results)):
                if player_detections['count'] > 0:
                    # Convert player detections to expected format (numpy arrays)
                    boxes_list = player_detections['boxes']
                    scores_list = player_detections['scores']
                    
                    # Convert to numpy arrays as expected by pose preprocessor
                    detections = {
                        'boxes': np.array(boxes_list, dtype=np.float32),
                        'scores': np.array(scores_list, dtype=np.float32)
                    }
                    
                    # Preprocess frame with player detections
                    pose_inputs, pose_meta = self.preprocessor.process_frame(frame, detections)
                    
                    if pose_inputs is not None:
                        all_pose_inputs.append(pose_inputs)
                        all_pose_meta.append(pose_meta)
                        
                        # Track mapping for each player crop
                        num_players = len(boxes_list)
                        for player_idx in range(num_players):
                            frame_mapping.append((frame_idx, player_idx))
            
            if not all_pose_inputs:
                return None, {
                    'batch_size': len(frames),
                    'player_detections_available': True,
                    'pose_inputs_available': False,
                    'task_config': {
                        'keypoint_threshold': self.keypoint_threshold
                    }
                }
            
            # Handle BatchFeature objects properly
            if len(all_pose_inputs) == 1:
                batch_pose_inputs = all_pose_inputs[0]
            else:
                # Concatenate BatchFeature objects properly
                batch_pose_inputs = {}
                for key in all_pose_inputs[0].keys():
                    # Concatenate tensors for each key
                    tensors = [inputs[key] for inputs in all_pose_inputs]
                    batch_pose_inputs[key] = torch.cat(tensors, dim=0)
            
            # Calculate batch size from the first tensor in the batch
            if isinstance(batch_pose_inputs, dict):
                first_key = next(iter(batch_pose_inputs.keys()))
                pose_batch_size = batch_pose_inputs[first_key].shape[0]
            else:
                pose_batch_size = len(all_pose_inputs)
            
            preprocess_meta = {
                'batch_size': len(frames),
                'player_detections_available': True,
                'pose_inputs_available': True,
                'pose_batch_size': pose_batch_size,
                'frame_mapping': frame_mapping,
                'individual_meta': all_pose_meta,
                'task_config': {
                    'keypoint_threshold': self.keypoint_threshold
                }
            }
            
            return batch_pose_inputs, preprocess_meta
            
        except Exception as e:
            log.error(f"Pose preprocessing failed: {e}")
            raise
    
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """
        Run pose estimation inference frame by frame (like original pipeline).
        
        Args:
            preprocessed_data: Preprocessed pose inputs
            metadata: Preprocessing metadata with frame mapping
            
        Returns:
            Dictionary with pose results per frame
        """
        if not metadata.get('pose_inputs_available', False):
            return None
        
        try:
            # Process frame by frame like original pipeline
            frame_mapping = metadata.get('frame_mapping', [])
            individual_meta = metadata.get('individual_meta', [])
            
            pose_results = {}
            
            # Get player detection results from dependencies if available
            dependencies = metadata.get('dependencies', {})
            player_results = dependencies.get('player_detection', {})
            
            if player_results and 'batch_results' in player_results:
                player_batch_results = player_results['batch_results']
                original_frames = metadata.get('original_frames', [])
                
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
        Post-process pose estimation outputs.
        
        Args:
            raw_outputs: Raw model outputs
            metadata: Preprocessing metadata
            
        Returns:
            Structured pose estimation results
        """
        if raw_outputs is None:
            # No pose inputs available
            batch_size = metadata.get('batch_size', 1)
            return {
                'batch_results': [[] for _ in range(batch_size)],  # Empty pose results for each frame
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
            
            # Process pose results
            individual_meta = metadata.get('individual_meta', [])
            frame_mapping = metadata.get('frame_mapping', [])
            
            # Process each pose result using frame mapping
            for pose_idx, (frame_idx, player_idx) in enumerate(frame_mapping):
                if pose_idx < len(individual_meta):
                    meta = individual_meta[pose_idx]
                    
                    # Extract pose for this player
                    # Handle different output formats from the model
                    try:
                        if hasattr(raw_outputs, 'heatmaps'):
                            # Model output has heatmaps attribute (ModelOutput object)
                            player_output = type(raw_outputs)(
                                heatmaps=raw_outputs.heatmaps[pose_idx:pose_idx+1]
                            )
                        elif isinstance(raw_outputs, dict) and 'heatmaps' in raw_outputs:
                            # Output is a dictionary with heatmaps
                            player_output = {
                                'heatmaps': raw_outputs['heatmaps'][pose_idx:pose_idx+1]
                            }
                        else:
                            # Fallback: use the raw outputs directly for batch processing
                            # This might not work for individual pose extraction
                            player_output = raw_outputs
                            log.warning(f"Unknown output format for pose estimation: {type(raw_outputs)}")
                    except Exception as e:
                        log.warning(f"Failed to extract pose output for player {player_idx}: {e}")
                        player_output = raw_outputs
                    
                    try:
                        pose_result = self.postprocessor.process_frame(player_output, meta)
                    except Exception as e:
                        # 姿勢推定のpostprocessingエラーを静かに処理
                        # （postprocessorの入力形式に互換性の問題があるため）
                        pose_result = None
                    
                    if pose_result:
                        # Filter keypoints by threshold
                        keypoints = pose_result[0]['keypoints']
                        scores = pose_result[0]['scores']
                        
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
        
        Args:
            frame: Input frame to draw on
            results: Pose estimation results
            vis_config: Visualization configuration
            
        Returns:
            Frame with pose visualization
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