"""
LiteTrackNet ball detector implementation.

This module provides a ball detector based on the LiteTrackNet architecture
for accurate ball tracking in tennis videos.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..base.detector import BaseBallDetector

logger = logging.getLogger(__name__)


def extract_tensor_from_output(model_output: Any, model_name: str = "model") -> torch.Tensor:
    """Extract tensor from model output (handles both tensor and dict outputs).
    
    Args:
        model_output: Model output (tensor or dict)
        model_name: Name of the model for error messages
        
    Returns:
        torch.Tensor: Extracted tensor
        
    Raises:
        ValueError: If no tensor found in output
    """
    if isinstance(model_output, torch.Tensor):
        return model_output
    
    if isinstance(model_output, dict):
        # Try common keys for main output
        main_keys = ['logits', 'predictions', 'output', 'out', 'heatmap', 'pred']
        for key in main_keys:
            if key in model_output:
                return model_output[key]
        
        # Fallback: use first available tensor
        for key, value in model_output.items():
            if hasattr(value, 'cpu'):  # Check if it's a tensor
                return value
        
        raise ValueError(f"No tensor found in {model_name} output dictionary with keys: {list(model_output.keys())}")
    
    raise ValueError(f"Unsupported {model_name} output type: {type(model_output)}")


class LiteTrackNetDetector(BaseBallDetector):
    """LiteTrackNet-based ball detector with integrated WASB-SBDT tracking.
    
    This detector uses a time-series approach requiring 3 consecutive frames
    and combines LiteTrackNet's detection capabilities with WASB-SBDT's
    advanced tracking system for enhanced accuracy and temporal consistency.
    
    Features:
    - 3-frame temporal analysis for accurate detection
    - Connected components analysis for precise localization
    - Integrated WASB-SBDT tracker for temporal consistency
    - Configurable postprocessing parameters
    """
    
    def __init__(self, model_path: str, device: str = "auto", input_size: Tuple[int, int] = (360, 640)):
        """Initialize LiteTrackNet detector.
        
        Args:
            model_path: Path to the trained model checkpoint (.ckpt file)
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            input_size: Input size as (height, width)
            
        Raises:
            ImportError: If LiteTrackNet modules are not available
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.device = self._setup_device(device)
        self.input_size = input_size
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup transforms
        self.transform = A.ReplayCompose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(),
            ToTensorV2(),
        ])
        
        # Initialize tracker using WASB-SBDT configuration
        self._setup_tracker()
        
        # Set postprocessor settings (compatible with WASB-SBDT)
        self.score_threshold = 0.5
        self.use_hm_weight = True
        
        logger.info(f"LiteTrackNet detector initialized on {self.device} with tracker integration")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the LiteTrackNet model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Import here to avoid circular imports
            from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss
            
            lit_model = LitLiteTracknetFocalLoss.load_from_checkpoint(
                model_path, map_location=self.device
            )
            model = lit_model.model
            model.to(self.device)
            model.eval()
            logger.info(f"Model loaded successfully from '{model_path}'")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _setup_tracker(self):
        """Initialize tracker using WASB-SBDT configuration."""
        try:
            from third_party.WASB_SBDT import load_default_config, build_tracker
            
            # Load default configuration for tracker
            config = load_default_config()
            
            # Build tracker
            self.tracker = build_tracker(config)
            
            logger.info("Tracker initialized using WASB-SBDT configuration")
        except ImportError as e:
            logger.warning(f"WASB-SBDT tracker not available: {e}")
            self.tracker = None
        except Exception as e:
            logger.warning(f"Failed to initialize tracker: {e}")
            self.tracker = None
    
    @property
    def frames_required(self) -> int:
        """LiteTrackNet requires 3 consecutive frames."""
        return 3
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'lite_tracknet',
            'model_path': self.model_path,
            'input_size': self.input_size,
            'frames_required': self.frames_required,
            'device': str(self.device),
            'architecture': 'LiteTrackNet with Focal Loss',
            'tracker_enabled': self.tracker is not None,
            'postprocessing': 'Connected Components + WASB-SBDT Tracking'
        }
    
    def preprocess(self, frame_data: List[Tuple[np.ndarray, dict]]) -> List[Tuple[Any, dict]]:
        """Preprocess frames for LiteTrackNet (requires 3 consecutive frames).
        
        Args:
            frame_data: List of (frame, metadata) tuples
            
        Returns:
            List of (model_input_tensor, metadata) tuples
            
        Raises:
            ValueError: If less than 3 frames provided
        """
        if len(frame_data) < self.frames_required:
            raise ValueError(f"LiteTrackNet requires at least {self.frames_required} consecutive frames")
        
        processed_data = []
        
        # Process frames in sliding windows of 3
        for i in range(len(frame_data) - 2):
            three_frames = [frame_data[i + j][0] for j in range(3)]
            # Use metadata from the target frame (last frame in the sequence)
            target_metadata = frame_data[i + 2][1]
            
            # Apply transforms to all three frames with consistent augmentation
            frames_transformed = []
            replay_data = self.transform(image=three_frames[0])
            frames_transformed.append(replay_data["image"])
            
            for frame in three_frames[1:]:
                replayed = A.ReplayCompose.replay(replay_data["replay"], image=frame)
                frames_transformed.append(replayed["image"])
            
            # Concatenate frames channel-wise: [3, H, W] -> [9, H, W]
            input_tensor = torch.cat(frames_transformed, dim=0)
            processed_data.append((input_tensor, target_metadata))
        
        return processed_data
    
    def infer(self, model_inputs: List[Tuple[Any, dict]]) -> List[Tuple[Any, dict]]:
        """Perform batch inference with LiteTrackNet.
        
        Args:
            model_inputs: List of (input_tensor, metadata) tuples
            
        Returns:
            List of (heatmap_output, metadata) tuples
        """
        if not model_inputs:
            return []
        
        inference_results = []
        
        with torch.no_grad():
            for input_tensor, metadata in model_inputs:
                # Add batch dimension and move to device
                batch_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                # Forward pass
                heatmap_pred = self.model(batch_tensor)
                
                # Extract tensor from output (handles dict and tensor outputs)
                heatmap_pred = extract_tensor_from_output(heatmap_pred, "LiteTrackNet")
                
                heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()
                
                inference_results.append((heatmap_prob, metadata))
        
        return inference_results
    
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert heatmaps to ball coordinates using connected components and tracker.
        
        This method follows the same approach as WASB-SBDT detector for consistency.
        
        Args:
            inference_results: List of (heatmap, metadata) tuples
            
        Returns:
            Dictionary with frame_id as keys and [[x, y, conf], ...] as values
        """
        detections = {}
        
        for heatmap, metadata in inference_results:
            frame_id = metadata.get('frame_id', 'unknown')
            
            # Extract candidate ball positions using connected components
            frame_detections = self._extract_ball_candidates(heatmap, self.score_threshold)
            
            # Use tracker to get the best ball position (if available)
            if self.tracker is not None and frame_detections:
                # Convert to tracker format
                tracker_input = [
                    {'xy': np.array([det[0] * self.input_size[1], det[1] * self.input_size[0]]), 'score': det[2]}
                    for det in frame_detections
                ]
                
                # Update tracker
                tracking_result = self.tracker.update(tracker_input)
                
                # Convert tracker result to standardized format
                if tracking_result.get('visi', False):
                    x_pixel = tracking_result['x']
                    y_pixel = tracking_result['y']
                    score = tracking_result.get('score', 0.0)
                    
                    # Normalize coordinates [0, 1]
                    x_norm = x_pixel / self.input_size[1]
                    y_norm = y_pixel / self.input_size[0]
                    
                    detections[frame_id] = [[x_norm, y_norm, score]]
                else:
                    detections[frame_id] = []
            elif frame_detections:
                # Use best candidate if tracker is not available
                detections[frame_id] = [frame_detections[0]]  # Take highest confidence
            else:
                # No detections found
                detections[frame_id] = []
        
        return detections
    
    def _extract_ball_candidates(self, heatmap: np.ndarray, threshold: float) -> List[List[float]]:
        """Extract ball candidate positions from heatmap using connected components method.
        
        This method follows the WASB-SBDT TrackNetV2 postprocessor implementation
        using connected components analysis for blob detection with weighted centroid calculation.
        
        Args:
            heatmap: 2D heatmap array [H, W]
            threshold: Confidence threshold for detection
            
        Returns:
            List of [x_norm, y_norm, confidence] detections
        """
        import cv2
        
        candidates = []
        
        # Check if any pixels exceed threshold
        if np.max(heatmap) > threshold:
            # Apply threshold to create binary mask
            _, hm_th = cv2.threshold(heatmap, threshold, 1, cv2.THRESH_BINARY)
            
            # Find connected components
            n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
            
            # Process each connected component (skip background label 0)
            for m in range(1, n_labels):
                # Get coordinates of pixels in this component
                ys, xs = np.where(labels == m)
                
                # Get heatmap weights for these pixels
                ws = heatmap[ys, xs]
                
                # Calculate centroid following WASB-SBDT configuration
                if len(ws) > 0:
                    if self.use_hm_weight:
                        # Weighted centroid calculation (default in WASB-SBDT)
                        score = ws.sum()  # Total weight as confidence score
                        x = np.sum(np.array(xs) * ws) / np.sum(ws)
                        y = np.sum(np.array(ys) * ws) / np.sum(ws)
                    else:
                        # Simple centroid calculation
                        score = float(ws.shape[0])  # Number of pixels as score
                        x = np.sum(np.array(xs)) / ws.shape[0]
                        y = np.sum(np.array(ys)) / ws.shape[0]
                    
                    # Normalize coordinates to [0, 1]
                    x_norm = float(x) / heatmap.shape[1]
                    y_norm = float(y) / heatmap.shape[0]
                    confidence = float(score)
                    
                    candidates.append([x_norm, y_norm, confidence])
        
        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Return top candidates (limit to avoid too many)
        return candidates[:10]