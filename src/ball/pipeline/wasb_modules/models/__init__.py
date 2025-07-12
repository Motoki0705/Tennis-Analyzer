from typing import Any
import torch.nn as nn
from omegaconf import DictConfig

# Import the real models
from .hrnet import HRNet

# Simple placeholder model for demo purposes
class SimpleBallModel(nn.Module):
    def __init__(self, frames_in=3, frames_out=3):
        super().__init__()
        self.frames_in = frames_in
        self.frames_out = frames_out
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(frames_in * 3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, frames_out, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return {0: x}  # Return dict format like HRNet


# Model factory
__factory = {
    'hrnet': HRNet,
    'tracknetv2': SimpleBallModel,
    'simple': SimpleBallModel,
}


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a model based on configuration."""
    model_name = cfg['model']['name']
    if model_name not in __factory.keys():
        # Default to simple model if model not found
        model_name = 'simple'
    
    if model_name == 'hrnet':
        # HRNet uses full config
        model = __factory[model_name](cfg['model'])
    else:
        # Simple models use basic parameters
        frames_in = cfg['model']['frames_in']
        frames_out = cfg['model']['frames_out']
        model = __factory[model_name](frames_in, frames_out)
    
    return model