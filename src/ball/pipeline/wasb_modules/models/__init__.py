from typing import Any
import torch.nn as nn
from omegaconf import DictConfig

# Import the real models
from .hrnet import HRNet

# Model factory
__factory = {
    'hrnet': HRNet
}

def build_model(cfg: DictConfig) -> nn.Module:
    """Build a model based on configuration."""
    model_name = cfg['model']['name']
    if model_name == 'hrnet':
        # HRNet uses full config
        model = __factory[model_name](cfg['model'])
    else:
        # Simple models use basic parameters
        frames_in = cfg['model']['frames_in']
        frames_out = cfg['model']['frames_out']
        model = __factory[model_name](frames_in, frames_out)
    
    return model