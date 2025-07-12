# Standalone WASB_SBDT modules

import os
from typing import Optional, Union, Tuple
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Internal module imports
from .models import build_model
from .pipeline_modules import (
    BallPreprocessor,
    BallDetector,
    DetectionPostprocessor,
)


def load_default_config() -> DictConfig:
    """Load default configuration for tennis ball detection and tracking."""
    cfg = {
        'model': {
            'name': 'hrnet',
            'frames_in': 3,
            'frames_out': 3,
            'inp_height': 288,
            'inp_width': 512,
            'out_height': 288,
            'out_width': 512,
            'rgb_diff': False,
            'out_scales': [0],
            'MODEL': {
                'EXTRA': {
                    'FINAL_CONV_KERNEL': 1,
                    'PRETRAINED_LAYERS': ['*'],
                    'STEM': {'INPLANES': 64, 'STRIDES': [1, 1]},
                    'STAGE1': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK',
                        'NUM_BLOCKS': [1], 'NUM_CHANNELS': [32], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE2': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2], 'NUM_CHANNELS': [16, 32], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2], 'NUM_CHANNELS': [16, 32, 64], 'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 1, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2, 2], 'NUM_CHANNELS': [16, 32, 64, 128], 'FUSE_METHOD': 'SUM'
                    },
                    'DECONV': {'NUM_DECONVS': 0, 'KERNEL_SIZE': [], 'NUM_BASIC_BLOCKS': 2}
                },
                'INIT_WEIGHTS': True
            }
        },
        'detector': {
            'name': 'tracknetv2',
            'model_path': 'third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar',
            'postprocessor': {
                'name': 'tracknetv2',
                'score_threshold': 0.5,
                'scales': [0],
                'blob_det_method': 'concomp',
                'use_hm_weight': True
            }
        },
        'tracker': {
            'name': 'online',
            'max_disp': 100
        },
        'dataloader': {
            'heatmap': {
                'sigmas': {0: 2.0}
            }
        },
        'runner': {
            'device': 'cuda',
            'gpus': [0]
        }
    }
    return OmegaConf.create(cfg)


class OnlineTracker:
    """Simple online ball tracker."""
    
    def __init__(self, max_disp=100):
        self.max_disp = max_disp
        self.refresh()
    
    def refresh(self):
        """Reset tracker state."""
        self.fid = 0
        self.prev_track = None
    
    def update(self, detections):
        """Update tracker with new detections."""
        self.fid += 1
        
        if not detections:
            return {"visi": False, "x": -1, "y": -1, "score": 0.0}
        
        # Filter detections by distance if we have previous track
        if self.prev_track is not None and self.prev_track.get("visi", False):
            prev_x, prev_y = self.prev_track["x"], self.prev_track["y"]
            filtered_detections = []
            for det in detections:
                x, y = det["xy"]
                dist = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
                if dist <= self.max_disp:
                    filtered_detections.append(det)
            detections = filtered_detections
        
        if not detections:
            return {"visi": False, "x": -1, "y": -1, "score": 0.0}
        
        # Select best detection (highest score)
        best_det = max(detections, key=lambda d: d["score"])
        x, y = best_det["xy"]
        
        result = {
            "visi": True,
            "x": float(x),
            "y": float(y),
            "score": float(best_det["score"])
        }
        
        self.prev_track = result
        return result


def build_tracker(cfg):
    """Build tracker based on configuration."""
    tracker_cfg = cfg.get('tracker', {})
    tracker_name = tracker_cfg.get('name', 'online')
    
    if tracker_name == 'online':
        max_disp = tracker_cfg.get('max_disp', 100)
        return OnlineTracker(max_disp=max_disp)
    else:
        # Default to online tracker
        return OnlineTracker()


__all__ = [
    "BallPreprocessor",
    "BallDetector",
    "DetectionPostprocessor",
    "OnlineTracker",
    "build_tracker",
    "load_default_config",
]