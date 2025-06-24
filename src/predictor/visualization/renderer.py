"""
Detection rendering utilities.

This module provides utilities for drawing detection results on images
and managing visual elements like trajectories and confidence displays.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from collections import deque

from .config import VisualizationConfig

logger = logging.getLogger(__name__)


class DetectionRenderer:
    """Renderer for ball detection visualization.
    
    This class handles the drawing of detection results on video frames
    including ball positions, trajectories, and confidence scores.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize detection renderer.
        
        Args:
            config: Visualization configuration (uses default if None)
        """
        self.config = config or VisualizationConfig()
        self.trajectory_points = deque(maxlen=self.config.trajectory_length)
        self.smoothed_points = deque(maxlen=self.config.smoothing_window)
        
        logger.info("DetectionRenderer initialized")
    
    def render_frame(
        self, 
        frame: np.ndarray, 
        detections: List[List[float]],
        frame_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Render detections on a single frame.
        
        Args:
            frame: Input frame as BGR image
            detections: List of detections [[x_norm, y_norm, confidence], ...]
            frame_info: Optional frame metadata
            
        Returns:
            Frame with detection overlays
        """
        # Copy frame to avoid modifying original
        rendered_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Filter detections by confidence
        valid_detections = [
            det for det in detections 
            if len(det) >= 3 and det[2] >= self.config.confidence_threshold
        ]
        
        if not valid_detections:
            return rendered_frame
        
        # Use highest confidence detection for trajectory
        best_detection = max(valid_detections, key=lambda x: x[2])
        x_norm, y_norm, confidence = best_detection[:3]
        
        # Convert to pixel coordinates
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)
        current_point = (x_pixel, y_pixel)
        
        # Update trajectory
        self._update_trajectory(current_point)
        
        # Apply smoothing if enabled
        if self.config.enable_smoothing:
            smoothed_point = self._get_smoothed_position(current_point)
            display_point = smoothed_point
        else:
            display_point = current_point
        
        # Draw trajectory
        if self.config.show_trajectory:
            self._draw_trajectory(rendered_frame)
        
        # Draw all valid detections
        for detection in valid_detections:
            x_norm, y_norm, confidence = detection[:3]
            x_pixel = int(x_norm * width)
            y_pixel = int(y_norm * height)
            
            self._draw_ball_detection(
                rendered_frame, 
                (x_pixel, y_pixel), 
                confidence
            )
        
        # Draw predictions if enabled
        if self.config.enable_prediction:
            self._draw_predictions(rendered_frame)
        
        # Add frame info if provided
        if frame_info:
            self._draw_frame_info(rendered_frame, frame_info)
        
        return rendered_frame
    
    def _update_trajectory(self, point: Tuple[int, int]):
        """Update trajectory with new point.
        
        Args:
            point: New trajectory point as (x, y)
        """
        self.trajectory_points.append(point)
        
        if self.config.enable_smoothing:
            self.smoothed_points.append(point)
    
    def _get_smoothed_position(self, current_point: Tuple[int, int]) -> Tuple[int, int]:
        """Get smoothed position using moving average.
        
        Args:
            current_point: Current detection point
            
        Returns:
            Smoothed position
        """
        if len(self.smoothed_points) < self.config.smoothing_window:
            return current_point
        
        # Calculate moving average
        avg_x = sum(p[0] for p in self.smoothed_points) / len(self.smoothed_points)
        avg_y = sum(p[1] for p in self.smoothed_points) / len(self.smoothed_points)
        
        return (int(avg_x), int(avg_y))
    
    def _draw_trajectory(self, frame: np.ndarray):
        """Draw trajectory on frame.
        
        Args:
            frame: Frame to draw on
        """
        if len(self.trajectory_points) < 2:
            return
        
        points = list(self.trajectory_points)
        
        for i in range(1, len(points)):
            # Calculate thickness based on recency
            thickness_ratio = i / len(points)
            thickness = int(
                self.config.trajectory_min_thickness + 
                (self.config.trajectory_max_thickness - self.config.trajectory_min_thickness) * thickness_ratio
            )
            thickness = max(1, thickness)
            
            cv2.line(
                frame, 
                points[i-1], 
                points[i], 
                self.config.trajectory_color, 
                thickness, 
                cv2.LINE_AA
            )
    
    def _draw_ball_detection(
        self, 
        frame: np.ndarray, 
        position: Tuple[int, int], 
        confidence: float
    ):
        """Draw a single ball detection.
        
        Args:
            frame: Frame to draw on
            position: Ball position as (x, y)
            confidence: Detection confidence
        """
        x, y = position
        
        # Draw ball circle
        cv2.circle(
            frame, 
            (x, y), 
            self.config.ball_radius, 
            self.config.ball_color, 
            -1
        )
        
        # Draw center circle
        cv2.circle(
            frame, 
            (x, y), 
            self.config.center_radius, 
            self.config.center_color, 
            -1
        )
        
        # Draw confidence score
        score_text = f"Ball: {confidence:.2f}"
        text_x = x + self.config.text_offset[0]
        text_y = y + self.config.text_offset[1]
        
        cv2.putText(
            frame, 
            score_text, 
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.font_scale, 
            self.config.score_color, 
            self.config.font_thickness
        )
    
    def _draw_predictions(self, frame: np.ndarray):
        """Draw predicted future positions.
        
        Args:
            frame: Frame to draw on
        """
        if len(self.trajectory_points) < 2:
            return
        
        # Calculate velocity from last two points
        last_point = self.trajectory_points[-1]
        prev_point = self.trajectory_points[-2]
        
        velocity_x = last_point[0] - prev_point[0]
        velocity_y = last_point[1] - prev_point[1]
        
        # Draw predicted positions
        for i in range(1, self.config.prediction_frames + 1):
            pred_x = last_point[0] + velocity_x * i
            pred_y = last_point[1] + velocity_y * i
            
            # Ensure predictions are within frame bounds
            height, width = frame.shape[:2]
            if 0 <= pred_x < width and 0 <= pred_y < height:
                # Draw prediction with decreasing opacity
                alpha = 1.0 - (i / (self.config.prediction_frames + 1))
                
                # Simple prediction circle (could be enhanced with alpha blending)
                pred_color = tuple(int(c * alpha) for c in self.config.ball_color)
                cv2.circle(
                    frame, 
                    (int(pred_x), int(pred_y)), 
                    self.config.ball_radius // 2, 
                    pred_color, 
                    1
                )
    
    def _draw_frame_info(self, frame: np.ndarray, frame_info: Dict[str, Any]):
        """Draw frame information on the frame.
        
        Args:
            frame: Frame to draw on
            frame_info: Frame metadata to display
        """
        height, width = frame.shape[:2]
        
        # Position info text at top-left corner
        info_lines = []
        
        if 'frame_number' in frame_info:
            info_lines.append(f"Frame: {frame_info['frame_number']}")
        
        if 'timestamp' in frame_info:
            info_lines.append(f"Time: {frame_info['timestamp']:.2f}s")
        
        if 'detection_count' in frame_info:
            info_lines.append(f"Detections: {frame_info['detection_count']}")
        
        # Draw info lines
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(
                frame, 
                line, 
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.font_scale, 
                (255, 255, 255),  # White text
                self.config.font_thickness
            )
    
    def reset(self):
        """Reset trajectory and smoothing state."""
        self.trajectory_points.clear()
        self.smoothed_points.clear()
    
    def get_trajectory_data(self) -> List[Tuple[int, int]]:
        """Get current trajectory points.
        
        Returns:
            List of trajectory points
        """
        return list(self.trajectory_points)
    
    def set_config(self, config: VisualizationConfig):
        """Update visualization configuration.
        
        Args:
            config: New visualization configuration
        """
        self.config = config
        
        # Update trajectory buffer size if needed
        if len(self.trajectory_points) > config.trajectory_length:
            # Keep only the most recent points
            new_deque = deque(
                list(self.trajectory_points)[-config.trajectory_length:],
                maxlen=config.trajectory_length
            )
            self.trajectory_points = new_deque