"""
Predictor API Module

Provides command-line interfaces for tennis ball detection and video processing.
"""

from .inference import main as inference_main
from .batch_process import main as batch_main

__all__ = [
    'inference_main',
    'batch_main',
] 