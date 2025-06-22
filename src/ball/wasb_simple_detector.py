"""
WASB-SBDT SimpleDetector for Enhanced Ball Analysis
video_demo.pyのSimpleDetectorパターンを統合
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WASBSimpleDetector:
    """WASB-SBDT SimpleDetector (video_demo pattern)"""
    
    def __init__(self, cfg: Any, device: torch.device):
        """
        Args:
            cfg: OmegaConf configuration
            device: torch device
        """
        self._frames_in = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out'] 
        self._input_wh = (cfg['model']['inp_width'], cfg['model']['inp_height'])
        self._device = device
        
        # Import WASB-SBDT modules
        from models import build_model
        from detectors.postprocessor import TracknetV2Postprocessor
        import dataloaders.img_transforms as T
        from utils.image import get_affine_transform
        
        self.get_affine_transform = get_affine_transform
        
        # Build model
        self._model = build_model(cfg)
        
        # Load weights
        model_path = cfg['detector']['model_path']
        if not model_path or not torch.cuda.is_available():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading WASB-SBDT model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self._model.load_state_dict(new_state_dict)
        self._model = self._model.to(device)
        self._model.eval()
        
        # Image transforms
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Postprocessor
        self._postprocessor = TracknetV2Postprocessor(cfg)
        
        print(f"✅ WASB-SBDT model loaded: frames_in={self._frames_in}")
        
    @property
    def frames_in(self):
        return self._frames_in
        
    @property
    def input_wh(self):
        return self._input_wh
    
    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Process frames and return ball detections
        
        Args:
            frames: List of BGR frames [H, W, 3]
            
        Returns:
            List of detections: [{'xy': (x, y), 'score': float}, ...]
        """
        if len(frames) != self._frames_in:
            print(f"⚠️ Expected {self._frames_in} frames, got {len(frames)}")
            return []
            
        try:
            # Prepare input tensor
            imgs_t = []
            input_w, input_h = self._input_wh
            
            # Use the last frame for transform reference
            ref_frame = frames[-1]
            trans_in = self.get_affine_transform(
                center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
                scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
                rot=0,
                output_size=[input_w, input_h],
            )
            
            # Inverse transform for output mapping
            trans_inv = self.get_affine_transform(
                center=np.array([ref_frame.shape[1] / 2.0, ref_frame.shape[0] / 2.0], dtype=np.float32),
                scale=max(ref_frame.shape[0], ref_frame.shape[1]) * 1.0,
                rot=0,
                output_size=[input_w, input_h],
                inv=1,
            )
            
            for frm in frames:
                frm_warp = cv2.warpAffine(frm, trans_in, (input_w, input_h), flags=cv2.INTER_LINEAR)
                img_pil = Image.fromarray(cv2.cvtColor(frm_warp, cv2.COLOR_BGR2RGB))
                imgs_t.append(self._transform(img_pil))
                
            imgs_tensor = torch.cat(imgs_t, dim=0).unsqueeze(0).to(self._device)
            
            # Forward pass
            with torch.no_grad():
                preds = self._model(imgs_tensor)
                
            # Post-process
            affine_mats = {0: torch.from_numpy(trans_inv).unsqueeze(0).to(self._device)}
            pp_results = self._postprocessor.run(preds, affine_mats)
            
            # Extract detections for the last frame (current frame)
            detections = []
            if 0 in pp_results and (self._frames_out - 1) in pp_results[0]:
                last_frame_results = pp_results[0][self._frames_out - 1]
                if 0 in last_frame_results:
                    scale_results = last_frame_results[0]
                    for xy, score in zip(scale_results['xys'], scale_results['scores']):
                        detections.append({'xy': xy, 'score': score})
                        
            print(f"🎯 WASB-SBDT detected {len(detections)} balls")
            return detections
            
        except Exception as e:
            print(f"❌ WASB-SBDT detection failed: {e}")
            logger.error(f"WASB-SBDT detection failed: {e}")
            return [] 