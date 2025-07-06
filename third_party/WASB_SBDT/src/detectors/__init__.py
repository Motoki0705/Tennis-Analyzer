from typing import Any, Optional
from omegaconf import DictConfig
import torch.nn as nn

from .detector import TracknetV2Detector
from .deepball_detector import DeepBallDetector

__factory = {
    'tracknetv2' : TracknetV2Detector,
    'deepball': DeepBallDetector
    }

def build_detector(cfg: DictConfig, model: Optional[nn.Module] = None) -> Any:
    """Build a detector based on configuration.
    
    Args:
        cfg: Configuration object containing detector parameters
        model: Optional pre-built model to use
        
    Returns:
        Detector instance ready for ball detection
        
    Raises:
        KeyError: If detector name is not supported
    """
    detector_name = cfg['detector']['name'] 
    if not detector_name in __factory.keys():
        raise KeyError('invalid detector: {}'.format(detector_name ))
    return __factory[detector_name](cfg, model=model)

