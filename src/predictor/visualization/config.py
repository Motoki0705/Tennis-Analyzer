"""
Configuration management for visualization components.

This module provides configuration classes and utilities for managing
visualization settings and parameters.
"""

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class VisualizationConfig:
    """Configuration for ball detection visualization.
    
    This class holds all visualization parameters including colors,
    sizes, and display options for ball detection overlays.
    """
    
    # Ball appearance
    ball_radius: int = 8
    center_radius: int = 3
    ball_color: Tuple[int, int, int] = (0, 0, 255)  # Red in BGR
    center_color: Tuple[int, int, int] = (255, 255, 255)  # White in BGR
    
    # Text and score display
    score_color: Tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    font_scale: float = 0.5
    font_thickness: int = 1
    text_offset: Tuple[int, int] = (15, -10)
    
    # Trajectory settings
    show_trajectory: bool = True
    trajectory_length: int = 15
    trajectory_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow in BGR
    trajectory_max_thickness: int = 3
    trajectory_min_thickness: int = 1
    
    # Detection filtering
    confidence_threshold: float = 0.3
    
    # Video output settings
    video_codec: str = 'mp4v'
    video_quality: int = 95
    
    # Progress and logging
    show_progress: bool = True
    progress_interval: int = 100
    
    # Advanced settings
    enable_smoothing: bool = False
    smoothing_window: int = 3
    enable_prediction: bool = False
    prediction_frames: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'ball_radius': self.ball_radius,
            'center_radius': self.center_radius,
            'ball_color': self.ball_color,
            'center_color': self.center_color,
            'score_color': self.score_color,
            'font_scale': self.font_scale,
            'font_thickness': self.font_thickness,
            'text_offset': self.text_offset,
            'show_trajectory': self.show_trajectory,
            'trajectory_length': self.trajectory_length,
            'trajectory_color': self.trajectory_color,
            'trajectory_max_thickness': self.trajectory_max_thickness,
            'trajectory_min_thickness': self.trajectory_min_thickness,
            'confidence_threshold': self.confidence_threshold,
            'video_codec': self.video_codec,
            'video_quality': self.video_quality,
            'show_progress': self.show_progress,
            'progress_interval': self.progress_interval,
            'enable_smoothing': self.enable_smoothing,
            'smoothing_window': self.smoothing_window,
            'enable_prediction': self.enable_prediction,
            'prediction_frames': self.prediction_frames
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VisualizationConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            VisualizationConfig instance
        """
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


# Predefined configurations
DEFAULT_CONFIG = VisualizationConfig()

HIGH_QUALITY_CONFIG = VisualizationConfig(
    ball_radius=10,
    center_radius=4,
    trajectory_length=20,
    trajectory_max_thickness=4,
    font_scale=0.6,
    enable_smoothing=True,
    smoothing_window=5
)

MINIMAL_CONFIG = VisualizationConfig(
    ball_radius=6,
    center_radius=2,
    show_trajectory=False,
    font_scale=0.4,
    show_progress=False
)

TRAJECTORY_FOCUSED_CONFIG = VisualizationConfig(
    ball_radius=6,
    trajectory_length=30,
    trajectory_max_thickness=5,
    enable_smoothing=True,
    enable_prediction=True,
    prediction_frames=3
)