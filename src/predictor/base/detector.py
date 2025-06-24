"""
Abstract base class for ball detectors.

This module defines the interface that all ball detection implementations
must follow to ensure consistency and interoperability.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class BaseBallDetector(ABC):
    """Abstract base class for all ball detectors.
    
    This class defines the three-stage pipeline that all ball detectors must implement:
    1. Preprocess: Convert raw frames to model-compatible format
    2. Infer: Perform batch inference on preprocessed data
    3. Postprocess: Convert raw outputs to standardized coordinates
    
    All stages preserve metadata to maintain frame association throughout the pipeline.
    """
    
    @abstractmethod
    def preprocess(self, frame_data: List[Tuple[np.ndarray, dict]]) -> List[Tuple[Any, dict]]:
        """Convert frames to model input format while preserving metadata.
        
        Args:
            frame_data: List of (frame, metadata) tuples where:
                - frame: RGB image as numpy array [H, W, C]
                - metadata: Dictionary containing frame_id and optional info
                
        Returns:
            List of (model_input, metadata) tuples ready for inference
            
        Raises:
            ValueError: If insufficient frames provided for the model
        """
        pass
    
    @abstractmethod
    def infer(self, model_inputs: List[Tuple[Any, dict]]) -> List[Tuple[Any, dict]]:
        """Perform batch inference while maintaining metadata association.
        
        Args:
            model_inputs: List of (preprocessed_input, metadata) tuples
            
        Returns:
            List of (raw_output, metadata) tuples from model inference
        """
        pass
    
    @abstractmethod
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert raw outputs to standardized ball coordinates.
        
        Args:
            inference_results: List of (raw_output, metadata) tuples
            
        Returns:
            Dictionary mapping frame_id to list of detections.
            Each detection is [x_norm, y_norm, confidence] where:
            - x_norm, y_norm: Normalized coordinates in [0, 1]
            - confidence: Detection confidence score in [0, 1]
        """
        pass
    
    @property
    @abstractmethod
    def frames_required(self) -> int:
        """Number of consecutive frames required by the model.
        
        Returns:
            Minimum number of frames needed for detection
        """
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Get model information and configuration.
        
        Returns:
            Dictionary containing model metadata such as:
            - model_type: Type of the model (e.g., 'lite_tracknet', 'wasb_sbdt')
            - input_size: Expected input dimensions
            - frames_required: Number of frames needed
            - device: Device the model is running on
        """
        pass
    
    def detect_balls(self, frame_data: List[Tuple[np.ndarray, dict]]) -> Dict[str, List[List[float]]]:
        """End-to-end ball detection pipeline.
        
        This method combines preprocess, infer, and postprocess steps
        to provide a simple interface for ball detection.
        
        Args:
            frame_data: List of (frame, metadata) tuples
            
        Returns:
            Dictionary mapping frame_id to list of ball detections
            
        Raises:
            ValueError: If insufficient frames for the model
        """
        if len(frame_data) < self.frames_required:
            raise ValueError(
                f"Model requires at least {self.frames_required} frames, "
                f"but only {len(frame_data)} provided"
            )
        
        # Execute three-stage pipeline
        model_inputs = self.preprocess(frame_data)
        inference_results = self.infer(model_inputs)
        detections = self.postprocess(inference_results)
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (alias for model_info property).
        
        Returns:
            Dictionary containing model metadata
        """
        return self.model_info