"""
WASB-SBDT ball detector implementation.

This module provides a ball detector based on the WASB-SBDT third-party
model for flexible ball detection in tennis videos.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch

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


class WASBSBDTDetector(BaseBallDetector):
    """HRNet-based WASB-SBDT ball detector using third-party models.
    
    This detector uses HRNet architecture for high-precision ball detection.
    It processes 3 consecutive frames and outputs 3 frame predictions,
    with temporal tracking for enhanced accuracy.
    
    Features:
    - 3-frame input, 3-frame output sliding window processing
    - HRNet backbone for precise localization
    - Integrated tracking for temporal consistency
    - Heatmap-based detection with peak extraction
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = "auto"):
        """Initialize WASB-SBDT detector.
        
        Args:
            model_path: Path to the trained model weights (.pth or .pth.tar file)
            config_path: Path to config file (optional, uses default if None)
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            
        Raises:
            ImportError: If WASB-SBDT modules are not available
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model initialization fails
        """
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration and initialize model
        self._load_config_and_model()
        
        logger.info(f"HRNet WASB-SBDT detector initialized, requires {self.frames_in} input frames, outputs {self.frames_out} predictions")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_config_and_model(self):
        """Load configuration and initialize model and tracker."""
        try:
            # Import WASB-SBDT modules
            from third_party.WASB_SBDT import create_hrnet_tracker, load_default_config
            
            # Load configuration
            if self.config_path and os.path.exists(self.config_path):
                from omegaconf import OmegaConf
                self.config = OmegaConf.load(self.config_path)
            else:
                self.config = load_default_config()
                
            # Override model path in config if provided
            if hasattr(self.config, 'detector') and self.model_path:
                self.config.detector.model_path = self.model_path
            
            # Initialize model and tracker
            self.model, self.tracker = create_hrnet_tracker(
                self.config, 
                model_path=self.model_path, 
                device=str(self.device)
            )
            
            # Ensure model is on correct device (redundant but safe)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set frames input/output (HRNet model uses 3 frames in, 3 frames out)
            if hasattr(self.config, 'model'):
                self.frames_in = self.config.model.get('frames_in', 3)
                self.frames_out = self.config.model.get('frames_out', 3)
            else:
                self.frames_in = 3
                self.frames_out = 3
            
            # Get postprocessor settings
            if hasattr(self.config, 'detector') and hasattr(self.config.detector, 'postprocessor'):
                self.score_threshold = self.config.detector.postprocessor.get('score_threshold', 0.5)
                self.use_hm_weight = self.config.detector.postprocessor.get('use_hm_weight', True)
            else:
                self.score_threshold = 0.5
                self.use_hm_weight = True
            
        except ImportError as e:
            raise ImportError(f"WASB-SBDT modules not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WASB-SBDT model: {e}")
    
    @property
    def frames_required(self) -> int:
        """Number of frames required by the HRNet WASB-SBDT model."""
        return getattr(self, 'frames_in', 3)  # HRNet always requires 3 frames
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'wasb_sbdt',
            'model_path': self.model_path,
            'config_path': self.config_path,
            'frames_required': self.frames_required,
            'device': str(self.device),
            'architecture': 'WASB-SBDT'
        }
    
    def preprocess(self, frame_data: List[Tuple[np.ndarray, dict]]) -> List[Tuple[Any, dict]]:
        """Preprocess frames for HRNet WASB-SBDT model.
        
        HRNet processes 3 consecutive frames and outputs 3 frame predictions.
        We use sliding window to process all frames, ensuring each frame gets predicted.
        
        Args:
            frame_data: List of (frame, metadata) tuples
            
        Returns:
            List of (frame_tensor_batch, metadata_dict) tuples
            
        Raises:
            ValueError: If insufficient frames provided
        """
        if len(frame_data) < self.frames_required:
            raise ValueError(f"HRNet WASB-SBDT requires at least {self.frames_required} consecutive frames")
        
        processed_data = []
        
        # Create sliding windows for 3-frame input
        # Each window will predict 3 frames, so we need to handle overlapping outputs
        for i in range(len(frame_data) - self.frames_required + 1):
            # Get 3 consecutive frames
            frame_sequence = [frame_data[i + j][0] for j in range(self.frames_required)]
            
            # Convert frames to model input format [B, 3*C, H, W]
            # Assuming frames are [H, W, C], we need to stack them as [3*C, H, W]
            frame_tensor = self._convert_frames_to_tensor(frame_sequence)
            
            # Create metadata for this window - includes info about all 3 target frames
            window_metadata = {
                'window_start_idx': i,
                'target_frame_indices': [i, i + 1, i + 2],
                'target_frame_ids': [frame_data[i + j][1].get('frame_id', f'frame_{i+j}') 
                                   for j in range(self.frames_required)]
            }
            
            processed_data.append((frame_tensor, window_metadata))
        
        return processed_data
    
    def _convert_frames_to_tensor(self, frame_sequence: List[np.ndarray]) -> torch.Tensor:
        """Convert 3 frame sequence to model input tensor.
        
        Args:
            frame_sequence: List of 3 frames [H, W, C]
            
        Returns:
            torch.Tensor: Shape [1, 3*C, H, W] ready for model input
        """
        # Resize frames to model input size (288, 512)
        target_height, target_width = 288, 512
        processed_frames = []
        
        for frame in frame_sequence:
            # Resize frame
            import cv2
            resized_frame = cv2.resize(frame, (target_width, target_height))
            
            # Convert to tensor and normalize [0, 1]
            if resized_frame.dtype != np.float32:
                resized_frame = resized_frame.astype(np.float32) / 255.0
            
            # Convert HWC to CHW
            frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1)
            processed_frames.append(frame_tensor)
        
        # Stack frames: [3, C, H, W] -> [3*C, H, W]
        stacked_tensor = torch.cat(processed_frames, dim=0)
        
        # Add batch dimension: [3*C, H, W] -> [1, 3*C, H, W]
        batch_tensor = stacked_tensor.unsqueeze(0)
        
        return batch_tensor.to(self.device)
    
    def infer(self, model_inputs: List[Tuple[Any, dict]]) -> List[Tuple[Any, dict]]:
        """Perform batch inference with HRNet WASB-SBDT model.
        
        Args:
            model_inputs: List of (frame_tensor_batch, metadata) tuples
            
        Returns:
            List of (heatmap_predictions, metadata) tuples
        """
        inference_results = []
        
        with torch.no_grad():
            for frame_tensor_batch, metadata in model_inputs:
                # Model inference: [B, 3*C, H, W] -> [B, 3, H, W]
                heatmap_output = self.model(frame_tensor_batch)
                
                # Extract tensor from output (handles dict and tensor outputs)
                heatmap_output = extract_tensor_from_output(heatmap_output, "WASB-SBDT")
                
                # Convert to numpy for postprocessing
                heatmap_np = heatmap_output.cpu().numpy()
                
                inference_results.append((heatmap_np, metadata))
        
        return inference_results
    
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert heatmap predictions to standardized detection format.
        
        The HRNet model outputs 3 heatmaps for 3 consecutive frames.
        We extract ball positions from each heatmap and use the tracker for temporal consistency.
        
        Args:
            inference_results: List of (heatmap_predictions, metadata) tuples
            
        Returns:
            Dictionary with frame_id as keys and [[x, y, conf], ...] as values
        """
        detections = {}
        
        for heatmap_predictions, metadata in inference_results:
            # heatmap_predictions shape: [1, 3, 288, 512] - 3 frames of heatmaps
            frame_ids = metadata['target_frame_ids']
            
            # Process each of the 3 predicted heatmaps
            for frame_idx in range(3):
                frame_id = frame_ids[frame_idx]
                heatmap = heatmap_predictions[0, frame_idx]  # [288, 512]
                
                # Extract candidate ball positions from heatmap
                frame_detections = self._extract_ball_candidates(heatmap, self.score_threshold)
                
                # Use tracker to get the best ball position
                if frame_detections:
                    # Convert to tracker format
                    tracker_input = [
                        {'xy': np.array([det[0] * 512, det[1] * 288]), 'score': det[2]}
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
                        x_norm = x_pixel / 512.0
                        y_norm = y_pixel / 288.0
                        
                        detections[frame_id] = [[x_norm, y_norm, score]]
                    else:
                        detections[frame_id] = []
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