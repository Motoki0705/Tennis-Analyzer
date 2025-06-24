"""
Abstract base class for preprocessing components.

This module defines the interface for preprocessing components that
can be used independently or as part of detector implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
import numpy as np


class BasePreprocessor(ABC):
    """Abstract base class for preprocessing components.
    
    Preprocessors are responsible for converting raw frame data
    into the format expected by specific models while preserving
    metadata for downstream processing.
    """
    
    @abstractmethod
    def transform(self, frame_data: List[Tuple[np.ndarray, dict]]) -> List[Tuple[Any, dict]]:
        """Transform raw frames to model input format.
        
        Args:
            frame_data: List of (frame, metadata) tuples where:
                - frame: RGB image as numpy array [H, W, C]
                - metadata: Dictionary containing frame information
                
        Returns:
            List of (transformed_input, metadata) tuples ready for model inference
            
        Raises:
            ValueError: If input data is invalid or insufficient
        """
        pass
    
    @property
    @abstractmethod
    def frames_required(self) -> int:
        """Minimum number of frames required for preprocessing.
        
        Returns:
            Number of consecutive frames needed
        """
        pass
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Get preprocessor configuration.
        
        Returns:
            Dictionary containing preprocessor settings
        """
        pass