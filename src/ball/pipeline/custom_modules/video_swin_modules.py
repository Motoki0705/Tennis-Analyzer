"""
VideoSwinTransformer pipeline modules for ball detection inference.
"""
import os
import cv2
import torch
import numpy as np
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional, Any
from scipy.ndimage import gaussian_filter
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    # For newer versions of scikit-image
    from scipy.ndimage import maximum_filter
    from scipy.ndimage import label, center_of_mass
    
    def peak_local_maxima(image, min_distance=1, threshold_abs=None):
        """
        Fallback implementation for peak_local_maxima using scipy.
        """
        # Apply threshold
        if threshold_abs is not None:
            mask = image >= threshold_abs
        else:
            mask = np.ones_like(image, dtype=bool)
        
        # Find local maxima using maximum filter
        footprint = np.ones((min_distance * 2 + 1, min_distance * 2 + 1))
        local_maxima = (maximum_filter(image, footprint=footprint) == image) & mask
        
        # Get coordinates
        coords = np.where(local_maxima)
        return coords

from src.ball.lit_module.lit_video_swin_transformer_focal import LitVideoSwinTransformerFocalLoss

log = logging.getLogger(__name__)


class FrameSequenceManager:
    """Manages frame sequences for VideoSwinTransformer inference."""
    
    def __init__(self, sequence_length: int = 5, overlap: int = 2, frame_interval: int = 1):
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.frame_interval = frame_interval
        self.frame_buffer = deque(maxlen=sequence_length * 2)
        self.frame_count = 0
        
    def add_frame(self, frame: np.ndarray) -> bool:
        """Add a frame to the buffer and return True if ready for inference."""
        if self.frame_count % self.frame_interval == 0:
            self.frame_buffer.append(frame)
        self.frame_count += 1
        
        return len(self.frame_buffer) >= self.sequence_length
    
    def get_sequence(self) -> Optional[List[np.ndarray]]:
        """Get the next sequence of frames for inference."""
        if len(self.frame_buffer) < self.sequence_length:
            return None
        
        sequence = list(self.frame_buffer)[-self.sequence_length:]
        return sequence
    
    def advance_sequence(self):
        """Advance the sequence by removing overlap frames."""
        frames_to_remove = self.sequence_length - self.overlap
        for _ in range(min(frames_to_remove, len(self.frame_buffer))):
            self.frame_buffer.popleft()
    
    def reset(self):
        """Reset the frame buffer."""
        self.frame_buffer.clear()
        self.frame_count = 0


class VideoSwinPostProcessor:
    """Post-processor for VideoSwinTransformer heatmap outputs."""
    
    def __init__(self, 
                 peak_threshold: float = 0.5,
                 min_distance: int = 10,
                 gaussian_sigma: float = 2.0,
                 use_temporal_smoothing: bool = True,
                 temporal_weight: float = 0.6):
        self.peak_threshold = peak_threshold
        self.min_distance = min_distance
        self.gaussian_sigma = gaussian_sigma
        self.use_temporal_smoothing = use_temporal_smoothing
        self.temporal_weight = temporal_weight
        
    def smooth_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to heatmap."""
        if self.gaussian_sigma > 0:
            return gaussian_filter(heatmap, sigma=self.gaussian_sigma)
        return heatmap
    
    def temporal_smooth_sequence(self, heatmaps: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing across sequence of heatmaps."""
        if not self.use_temporal_smoothing or heatmaps.shape[0] < 3:
            return heatmaps
        
        # Weight center frame more heavily
        center_idx = heatmaps.shape[0] // 2
        weights = np.ones(heatmaps.shape[0]) * (1 - self.temporal_weight) / (heatmaps.shape[0] - 1)
        weights[center_idx] = self.temporal_weight
        
        # Apply weighted average
        smoothed = np.average(heatmaps, axis=0, weights=weights)
        return smoothed
    
    def detect_peaks(self, heatmap: np.ndarray, scale_factors: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Detect peaks in heatmap and convert to original coordinates."""
        # Apply smoothing
        smoothed = self.smooth_heatmap(heatmap)
        
        # Find peaks
        peaks = peak_local_maxima(
            smoothed,
            min_distance=self.min_distance,
            threshold_abs=self.peak_threshold
        )
        
        detections = []
        scale_x, scale_y = scale_factors
        
        for y, x in zip(peaks[0], peaks[1]):
            confidence = float(smoothed[y, x])
            
            # Convert to original coordinates
            orig_x = x * scale_x
            orig_y = y * scale_y
            
            detections.append({
                'x': orig_x,
                'y': orig_y,
                'confidence': confidence,
                'heatmap_x': x,
                'heatmap_y': y
            })
        
        # Sort by confidence
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections
    
    def process_sequence(self, heatmaps: np.ndarray, scale_factors: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Process a sequence of heatmaps and return detections."""
        # Apply temporal smoothing
        if heatmaps.shape[0] > 1:
            center_heatmap = self.temporal_smooth_sequence(heatmaps)
        else:
            center_heatmap = heatmaps[0]
        
        # Detect peaks
        detections = self.detect_peaks(center_heatmap, scale_factors)
        return detections


class VideoSwinBallTracker:
    """Main tracker class for VideoSwinTransformer ball detection."""
    
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        
        # Initialize model
        self._load_model()
        
        # Initialize components
        self.sequence_manager = FrameSequenceManager(
            sequence_length=cfg['input']['sequence_length'],
            overlap=cfg['input']['overlap'],
            frame_interval=cfg['input']['frame_interval']
        )
        
        self.postprocessor = VideoSwinPostProcessor(
            peak_threshold=cfg['postprocess']['peak_threshold'],
            min_distance=cfg['postprocess']['min_distance'],
            gaussian_sigma=cfg['postprocess']['gaussian_sigma'],
            use_temporal_smoothing=cfg['postprocess']['use_temporal_smoothing'],
            temporal_weight=cfg['postprocess']['temporal_weight']
        )
        
        # Model input configuration
        self.img_size = tuple(cfg['model']['img_size'])
        self.n_frames = cfg['model']['n_frames']
        
        # Processing state
        self.processed_frames = 0
        self.total_inference_time = 0.0
        
    def _load_model(self):
        """Load VideoSwinTransformer model from checkpoint."""
        checkpoint_path = self.cfg['model']['checkpoint_path']
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        log.info(f"Loading VideoSwinTransformer from {checkpoint_path}")
        
        # Load model from Lightning checkpoint
        self.model = LitVideoSwinTransformerFocalLoss.load_from_checkpoint(
            checkpoint_path
        )
        
        self.model.to(self.device)
        self.model.eval()
        log.info("Model loaded successfully")
    
    def _preprocess_sequence(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """Preprocess frame sequence for model input."""
        # Get original dimensions from reference frame
        ref_frame = frames[-1]
        orig_h, orig_w = ref_frame.shape[:2]
        
        # Calculate scale factors
        target_h, target_w = self.img_size
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        
        # Process each frame
        processed_frames = []
        for frame in frames:
            # Resize to model input size
            resized = cv2.resize(frame, (target_w, target_h))
            
            # Convert BGR to RGB and normalize
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Convert to tensor
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
            processed_frames.append(tensor)
        
        # Stack frames: (N, C, H, W)
        sequence_tensor = torch.stack(processed_frames, dim=0)
        
        # Add batch dimension: (1, N, C, H, W)
        batch_tensor = sequence_tensor.unsqueeze(0).to(self.device)
        
        return batch_tensor, (scale_x, scale_y)
    
    def process_frame(self, frame: np.ndarray, return_heatmap: bool = False) -> Optional[List[Dict[str, Any]]]:
        # Add frame to sequence manager
        ready_for_inference = self.sequence_manager.add_frame(frame)
        
        if not ready_for_inference:
            return None
        
        # Get sequence for inference
        frame_sequence = self.sequence_manager.get_sequence()
        if frame_sequence is None:
            return None
        
        # Preprocess sequence
        input_tensor, scale_factors = self._preprocess_sequence(frame_sequence)
        
        # Run inference
        with torch.no_grad():
            import time
            start_time = time.time()
            
            # Model inference
            output = self.model(input_tensor)  # (B, N, C_out, H, W)
            
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            
            if self.cfg['logging']['log_inference_time']:
                log.debug(f"Inference time: {inference_time:.4f}s")
        
        # Extract heatmaps from output
        heatmaps = output[0].cpu().numpy()  # (N, C_out, H, W)
        heatmaps = heatmaps.squeeze(1)  # Remove channel dimension: (N, H, W)
        
        # Apply sigmoid to get probabilities
        heatmaps = 1 / (1 + np.exp(-heatmaps))
        
        # Post-process to get detections
        detections = self.postprocessor.process_sequence(heatmaps, scale_factors)
        
        # Advance sequence for next inference
        self.sequence_manager.advance_sequence()
        self.processed_frames += 1
        
        if return_heatmap:
            # Return center frame heatmap for visualization
            center_idx = len(heatmaps) // 2
            center_heatmap = heatmaps[center_idx] if len(heatmaps) > 1 else heatmaps[0]
            return detections, center_heatmap
        else:
            return detections
    
    def reset(self):
        """Reset tracker state."""
        self.sequence_manager.reset()
        self.processed_frames = 0
        self.total_inference_time = 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        avg_inference_time = (self.total_inference_time / max(1, self.processed_frames))
        fps = 1.0 / max(avg_inference_time, 1e-6)
        
        return {
            'processed_frames': self.processed_frames,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': avg_inference_time,
            'fps': fps
        }


def create_video_swin_tracker(cfg: Dict[str, Any], device: torch.device) -> VideoSwinBallTracker:
    """Factory function to create VideoSwinBallTracker."""
    return VideoSwinBallTracker(cfg, device)