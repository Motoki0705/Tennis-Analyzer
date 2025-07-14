import os
import cv2
import torch
import numpy as np
import logging

# Local imports
from .models import build_model

log = logging.getLogger(__name__)


class BallPreprocessor:
    """Simplified frame sequence preprocessor for ball detection."""

    def __init__(self, cfg):
        self._frames_in = cfg['model']['frames_in']
        self._input_wh = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        # ImageNet normalization values (ensure float32)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def frames_in(self):
        return self._frames_in

    def process_batch(self, frame_sequences):
        """Process batch of frame sequences with simple resize and normalization."""
        batch_tensors = []
        input_w, input_h = self._input_wh
        
        # Calculate scale factors for coordinate conversion
        scale_factors = []

        for frames in frame_sequences:
            assert len(frames) == self._frames_in, f"Expected {self._frames_in} frames, got {len(frames)}"
            
            # Get original frame dimensions from reference frame
            ref_frame = frames[-1]
            orig_h, orig_w = ref_frame.shape[:2]
            
            # Calculate scale factors for coordinate conversion
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            scale_factors.append((scale_x, scale_y))

            frame_tensors = []
            for frame in frames:
                # Simple resize instead of affine transform
                resized = cv2.resize(frame, (input_w, input_h))
                
                # Convert BGR to RGB and normalize to [0, 1]
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Apply ImageNet normalization
                normalized = (rgb - self._mean) / self._std
                
                # Convert to tensor and change to CHW format (ensure float32)
                tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
                frame_tensors.append(tensor)
            
            # Concatenate frames in channel dimension
            batch_tensors.append(torch.cat(frame_tensors, dim=0))

        batch_tensor = torch.stack(batch_tensors, dim=0)
        batch_meta = {'scale_factors': scale_factors}
        
        return batch_tensor, batch_meta


class BallDetector:
    """Ball detector using neural network model."""

    def __init__(self, cfg, device):
        self._device = device
        
        log.info("Building model...")
        self._model = build_model(cfg)
        
        model_path = cfg['detector']['model_path']
        # For demo, we'll allow missing model file and use random weights
        if os.path.exists(model_path):
            log.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            try:
                self._model.load_state_dict(new_state_dict)
            except:
                log.warning("Could not load model weights, using random initialization")
        else:
            log.warning(f"Model file not found: {model_path}, using random initialization")
            
        self._model = self._model.to(device)
        self._model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_tensor):
        """Run inference on batch tensor."""
        batch_tensor = batch_tensor.to(self._device)
        device_type = 'cuda' if self._device.type == 'cuda' else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type=='cuda'):
            preds = self._model(batch_tensor)
        return preds


class DetectionPostprocessor:
    """Simplified detection postprocessor for extracting ball coordinates using maximum detection."""

    def __init__(self, cfg):
        self._score_threshold = cfg['detector']['postprocessor']['score_threshold']
        self._frames_out = cfg['model']['frames_out']

    def process_batch(self, batch_preds, batch_meta, device):
        """Process batch predictions with simple maximum detection."""
        scale_factors_list = batch_meta['scale_factors']
        batch_size = len(scale_factors_list)
        
        # Handle predictions format
        if isinstance(batch_preds, dict):
            # Extract predictions from dict (assume scale 0)
            preds = batch_preds[0] if 0 in batch_preds else list(batch_preds.values())[0]
        else:
            preds = batch_preds
        
        # Apply sigmoid to get probabilities
        if hasattr(preds, 'sigmoid'):
            heatmaps = preds.sigmoid()
        else:
            heatmaps = torch.sigmoid(preds)
        
        # Move to CPU for processing
        heatmaps = heatmaps.detach().cpu().numpy()
        
        batch_detections = []
        for batch_idx in range(batch_size):
            scale_x, scale_y = scale_factors_list[batch_idx]
            
            # Get the last frame heatmap (most recent prediction)
            if heatmaps.shape[1] >= self._frames_out:
                hm = heatmaps[batch_idx, self._frames_out - 1]  # Last frame
            else:
                hm = heatmaps[batch_idx, -1]  # Use last available frame
            
            # Simple maximum detection
            max_score = np.max(hm)
            if max_score > self._score_threshold:
                # Find the position of maximum value
                max_pos = np.unravel_index(np.argmax(hm), hm.shape)
                y, x = max_pos
                
                # Scale back to original image coordinates
                x_orig = x * scale_x
                y_orig = y * scale_y
                
                detection = {
                    'x': float(x_orig),
                    'y': float(y_orig), 
                    'score': float(max_score),
                    'visi': True
                }
            else:
                # No detection above threshold
                detection = {
                    'x': -1,
                    'y': -1,
                    'score': 0.0,
                    'visi': False
                }
            
            batch_detections.append(detection)
        
        return batch_detections