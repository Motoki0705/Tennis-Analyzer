"""
Modular Ball Detection System

A flexible ball detection system that can integrate third-party ball detection models
with a three-stage pipeline: preprocess -> infer -> postprocess.

The module supports batch inference and provides a clean, extensible interface
for different ball detection models.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import cv2
from abc import ABC, abstractmethod

import albumentations as A

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle different model types
try:   
    from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss
    LITE_TRACKNET_AVAILABLE = True
    logger.info("LiteTrackNet modules available.")
except ImportError:
    LITE_TRACKNET_AVAILABLE = False
    raise ImportError("LiteTrackNet modules not available. Check imports.")

try:
    from third_party.WASB_SBDT.src import load_simple_config, create_ball_detector
    WASB_SBDT_AVAILABLE = True
    logger.info("WASB-SBDT modules available.")
except ImportError:
    WASB_SBDT_AVAILABLE = False
    raise ImportError("WASB-SBDT modules not available. Check imports.")


class BaseBallDetector(ABC):
    """Base class for ball detection implementations."""
    
    @abstractmethod
    def preprocess(self, frame_data: List[Tuple[np.array, dict]]) -> List[Tuple[Any, dict]]:
        """Convert frames to model input format while preserving metadata."""
        pass
    
    @abstractmethod
    def infer(self, model_inputs: List[Tuple[Any, dict]]) -> List[Tuple[Any, dict]]:
        """Perform batch inference while maintaining metadata association."""
        pass
    
    @abstractmethod
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert to frame_id keyed dictionary with [x, y, conf] values."""
        pass


class LiteTrackNetDetector(BaseBallDetector):
    """LiteTrackNet-based ball detector using internal models."""
    
    def __init__(self, model_path: str, device: str = "auto", input_size: Tuple[int, int] = (360, 640)):
        """Initialize LiteTrackNet detector.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            input_size: Input size as (height, width)
        """
        if not LITE_TRACKNET_AVAILABLE:
            raise ImportError("LiteTrackNet modules not available. Check imports.")
            
        self.device = self._setup_device(device)
        self.input_size = input_size
        self.model = self._load_model(model_path)
        
        # Setup transforms
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        self.transform = A.ReplayCompose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(),
            ToTensorV2(),
        ])
        
        logger.info(f"LiteTrackNet detector initialized on {self.device}")
    
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
    
    def preprocess(self, frame_data: List[Tuple[np.array, dict]]) -> List[Tuple[Any, dict]]:
        """Preprocess frames for LiteTrackNet (requires 3 consecutive frames).
        
        Args:
            frame_data: List of (frame, metadata) tuples
            
        Returns:
            List of (model_input_tensor, metadata) tuples
        """
        if len(frame_data) < 3:
            raise ValueError("LiteTrackNet requires at least 3 consecutive frames")
        
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
                heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()
                
                inference_results.append((heatmap_prob, metadata))
        
        return inference_results
    
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert heatmaps to ball coordinates.
        
        Args:
            inference_results: List of (heatmap, metadata) tuples
            
        Returns:
            Dictionary with frame_id as keys and [[x, y, conf], ...] as values
        """
        detections = {}
        
        for heatmap, metadata in inference_results:
            frame_id = metadata.get('frame_id', 'unknown')
            
            # Find peak in heatmap
            h, w = heatmap.shape
            max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y_resized, x_resized = max_pos
            confidence = float(heatmap[y_resized, x_resized])
            
            # Convert to normalized coordinates [0, 1]
            x_norm = x_resized / w
            y_norm = y_resized / h
            
            # Store detection
            if frame_id not in detections:
                detections[frame_id] = []
            
            detections[frame_id].append([x_norm, y_norm, confidence])
        
        return detections


class WASBSBDTDetector(BaseBallDetector):
    """WASB-SBDT based ball detector using third-party models."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = "auto"):
        """Initialize WASB-SBDT detector.
        
        Args:
            model_path: Path to the trained model weights
            config_path: Path to config file (optional, uses default if None)
            device: Device to run inference on
        """
        if not WASB_SBDT_AVAILABLE:
            raise ImportError("WASB-SBDT modules not available. Check imports.")
            
        self.device = self._setup_device(device)
        self.model_path = model_path
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            from omegaconf import OmegaConf
            self.config = OmegaConf.load(config_path)
        else:
            self.config = load_simple_config()
            
        # Override model path in config
        self.config.detector.model_path = model_path
        
        # Initialize detector and tracker
        self.detector, self.tracker = create_ball_detector(self.config, device=str(self.device))
        self.frames_in = self.detector.frames_in
        
        logger.info(f"WASB-SBDT detector initialized, requires {self.frames_in} input frames")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def preprocess(self, frame_data: List[Tuple[np.array, dict]]) -> List[Tuple[Any, dict]]:
        """Preprocess frames for WASB-SBDT.
        
        Args:
            frame_data: List of (frame, metadata) tuples
            
        Returns:
            List of (frame_sequence, metadata) tuples
        """
        if len(frame_data) < self.frames_in:
            raise ValueError(f"WASB-SBDT requires at least {self.frames_in} consecutive frames")
        
        processed_data = []
        
        # Process frames in sliding windows
        for i in range(len(frame_data) - self.frames_in + 1):
            frame_sequence = [frame_data[i + j][0] for j in range(self.frames_in)]
            # Use metadata from the target frame (last frame in the sequence)
            target_metadata = frame_data[i + self.frames_in - 1][1]
            
            processed_data.append((frame_sequence, target_metadata))
        
        return processed_data
    
    def infer(self, model_inputs: List[Tuple[Any, dict]]) -> List[Tuple[Any, dict]]:
        """Perform batch inference with WASB-SBDT.
        
        Args:
            model_inputs: List of (frame_sequence, metadata) tuples
            
        Returns:
            List of (detections, metadata) tuples
        """
        inference_results = []
        
        for frame_sequence, metadata in model_inputs:
            # Process frame sequence
            detections = self.detector.process_frames(frame_sequence)
            inference_results.append((detections, metadata))
        
        return inference_results
    
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert raw detections to standardized format.
        
        Args:
            inference_results: List of (detections, metadata) tuples
            
        Returns:
            Dictionary with frame_id as keys and [[x, y, conf], ...] as values
        """
        detections = {}
        
        for raw_detections, metadata in inference_results:
            frame_id = metadata.get('frame_id', 'unknown')
            
            if frame_id not in detections:
                detections[frame_id] = []
            
            # Convert WASB-SBDT detections to standard format
            for detection in raw_detections:
                if isinstance(detection, dict):
                    xy = detection.get('xy', [0, 0])
                    score = detection.get('score', 0.0)
                    
                    # Normalize coordinates if needed (assuming they're already normalized)
                    x_norm = float(xy[0])
                    y_norm = float(xy[1])
                    confidence = float(score)
                    
                    detections[frame_id].append([x_norm, y_norm, confidence])
        
        return detections


class BallDetectionModule:
    """Main ball detection module with unified interface."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, 
                 device: str = "auto", model_type: str = "auto"):
        """Initialize ball detection module.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to configuration file (optional)
            device: Device for inference ('cuda', 'cpu', or 'auto')
            model_type: Type of model ('lite_tracknet', 'wasb_sbdt', or 'auto')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_path)
        
        # Initialize appropriate detector
        self.detector = self._create_detector(model_type)
        logger.info(f"BallDetectionModule initialized with {model_type} detector")
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type from file extension and availability."""
        if model_path.endswith('.ckpt') and LITE_TRACKNET_AVAILABLE:
            return "lite_tracknet"
        elif (model_path.endswith(('.pth', '.pth.tar')) and WASB_SBDT_AVAILABLE):
            return "wasb_sbdt"
        else:
            # Fallback to available detectors
            if LITE_TRACKNET_AVAILABLE:
                return "lite_tracknet"
            elif WASB_SBDT_AVAILABLE:
                return "wasb_sbdt"
            else:
                raise RuntimeError("No ball detection models available")
    
    def _create_detector(self, model_type: str) -> BaseBallDetector:
        """Create appropriate detector instance."""
        if model_type == "lite_tracknet":
            return LiteTrackNetDetector(self.model_path, self.device)
        elif model_type == "wasb_sbdt":
            return WASBSBDTDetector(self.model_path, self.config_path, self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def preprocess(self, frame_data: List[Tuple[np.array, dict]]) -> List[Tuple[Any, dict]]:
        """Convert frames to model input format while preserving metadata.
        
        Args:
            frame_data: List of (frame, metadata) tuples where:
                - frame: numpy array [H, W, C] in BGR format
                - metadata: dict with 'frame_id' and other information
                
        Returns:
            List of (model_input, metadata) tuples
        """
        return self.detector.preprocess(frame_data)
    
    def infer(self, model_inputs: List[Tuple[Any, dict]]) -> List[Tuple[Any, dict]]:
        """Perform batch inference while maintaining metadata association.
        
        Args:
            model_inputs: List of (model_input, metadata) tuples
            
        Returns:
            List of (raw_output, metadata) tuples
        """
        return self.detector.infer(model_inputs)
    
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert to frame_id keyed dictionary with [x, y, conf] values.
        
        Args:
            inference_results: List of (raw_output, metadata) tuples
            
        Returns:
            Dictionary with frame_id as keys and [[x, y, conf], ...] as values
            Coordinates are normalized to [0, 1], confidence in [0, 1]
        """
        return self.detector.postprocess(inference_results)
    
    def detect_balls(self, frame_data: List[Tuple[np.array, dict]]) -> Dict[str, List[List[float]]]:
        """End-to-end ball detection pipeline.
        
        Args:
            frame_data: List of (frame, metadata) tuples
            
        Returns:
            Dictionary with frame_id as keys and [[x, y, conf], ...] as values
        """
        try:
            # Preprocess
            model_inputs = self.preprocess(frame_data)
            logger.debug(f"Preprocessed {len(frame_data)} frames into {len(model_inputs)} model inputs")
            
            # Inference
            inference_results = self.infer(model_inputs)
            logger.debug(f"Performed inference on {len(model_inputs)} inputs")
            
            # Postprocess
            detections = self.postprocess(inference_results)
            logger.debug(f"Generated detections for {len(detections)} frames")
            
            return detections
            
        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'device': str(self.device),
            'detector_type': type(self.detector).__name__,
            'frames_required': getattr(self.detector, 'frames_in', 3)
        }


def create_ball_detection_module(model_path: str, config_path: Optional[str] = None,
                               device: str = "auto", model_type: str = "auto") -> BallDetectionModule:
    """Factory function to create a BallDetectionModule.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to configuration file (optional)
        device: Device for inference
        model_type: Type of model to use
        
    Returns:
        Configured BallDetectionModule instance
    """
    return BallDetectionModule(model_path, config_path, device, model_type)