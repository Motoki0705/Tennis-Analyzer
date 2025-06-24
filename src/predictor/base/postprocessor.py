"""
Abstract base class for postprocessing components.

This module defines the interface for postprocessing components that
convert raw model outputs to standardized detection formats.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
import numpy as np


class BasePostprocessor(ABC):
    """Abstract base class for postprocessing components.
    
    Postprocessors are responsible for converting raw model outputs
    into standardized detection formats while preserving metadata
    from the preprocessing stage.
    """
    
    @abstractmethod
    def process(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Process raw model outputs to standardized detection format.
        
        Args:
            inference_results: List of (raw_output, metadata) tuples from model inference
            
        Returns:
            Dictionary mapping frame_id to list of detections.
            Each detection is [x_norm, y_norm, confidence] where:
            - x_norm, y_norm: Normalized coordinates in [0, 1] 
            - confidence: Detection confidence score in [0, 1]
        """
        pass
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Get postprocessor configuration.
        
        Returns:
            Dictionary containing postprocessor settings such as:
            - confidence_threshold: Minimum confidence for detections
            - nms_threshold: Non-maximum suppression threshold
            - max_detections: Maximum number of detections per frame
        """
        pass
    
    def filter_detections(self, detections: Dict[str, List[List[float]]], 
                         confidence_threshold: float = 0.5) -> Dict[str, List[List[float]]]:
        """Filter detections by confidence threshold.
        
        Args:
            detections: Dictionary of frame_id to detection lists
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Filtered detections dictionary
        """
        filtered = {}
        for frame_id, detection_list in detections.items():
            filtered_list = [
                det for det in detection_list 
                if len(det) >= 3 and det[2] >= confidence_threshold
            ]
            if filtered_list:
                filtered[frame_id] = filtered_list
        return filtered