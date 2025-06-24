"""
Base classes for the prediction system.

This module contains abstract base classes that define the interface
for all prediction components in the tennis analysis system.
"""

from .detector import BaseBallDetector
from .preprocessor import BasePreprocessor
from .postprocessor import BasePostprocessor

__all__ = [
    'BaseBallDetector',
    'BasePreprocessor', 
    'BasePostprocessor',
]