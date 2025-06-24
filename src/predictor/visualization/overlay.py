"""
Video overlay functionality for ball detection.

This module provides the VideoOverlay class for creating videos with
ball detection overlays and managing the video processing pipeline.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
import cv2
import numpy as np
from pathlib import Path

from .config import VisualizationConfig
from .renderer import DetectionRenderer

logger = logging.getLogger(__name__)


class VideoOverlay:
    """Handles video overlay functionality for ball detection.
    
    This class manages the creation of overlay videos with ball detection
    results, including trajectory visualization and confidence displays.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize video overlay handler.
        
        Args:
            config: Visualization configuration (uses default if None)
        """
        self.config = config or VisualizationConfig()
        self.renderer = DetectionRenderer(self.config)
        
        logger.info("VideoOverlay initialized")
    
    def create_overlay_video(
        self, 
        video_path: str, 
        detections: Dict[str, List[List[float]]], 
        output_path: str,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Create video with detection overlays.
        
        Args:
            video_path: Path to input video file
            detections: Detection results mapping frame_id to detection lists
            output_path: Path to save overlay video
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the output video file
            
        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If video cannot be opened or written
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Processing video: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")
            
            # Process frames
            frame_idx = 0
            detection_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_id = f'frame_{frame_idx:06d}'
                
                # Get detections for current frame
                frame_detections = detections.get(frame_id, [])
                
                # Create frame info
                frame_info = {
                    'frame_number': frame_idx,
                    'timestamp': frame_idx / fps if fps > 0 else frame_idx,
                    'detection_count': len(frame_detections)
                }
                
                # Render detections on frame
                rendered_frame = self.renderer.render_frame(
                    frame, 
                    frame_detections, 
                    frame_info
                )
                
                # Write frame
                out.write(rendered_frame)
                
                # Update counters
                frame_idx += 1
                detection_count += len(frame_detections)
                
                # Progress reporting
                if self.config.show_progress and frame_idx % self.config.progress_interval == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
                
                # Progress callback
                if progress_callback:
                    progress = frame_idx / total_frames
                    progress_callback(progress)
            
            logger.info(
                f"Overlay video created: {output_path} "
                f"({frame_idx} frames, {detection_count} total detections)"
            )
            
            return output_path
            
        finally:
            cap.release()
            out.release()
    
    def render_frame_with_detections(
        self,
        frame: np.ndarray,
        detections: List[List[float]],
        frame_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Render a single frame with detection overlays.
        
        Args:
            frame: Input frame as BGR image
            detections: List of detections for this frame
            frame_info: Optional frame metadata
            
        Returns:
            Frame with detection overlays
        """
        return self.renderer.render_frame(frame, detections, frame_info)
    
    def create_detection_summary_video(
        self,
        video_path: str,
        detections: Dict[str, List[List[float]]],
        output_path: str,
        summary_frames: int = 30
    ) -> str:
        """Create a summary video showing only frames with detections.
        
        Args:
            video_path: Path to input video file
            detections: Detection results
            output_path: Path to save summary video
            summary_frames: Maximum number of frames to include
            
        Returns:
            Path to the summary video file
        """
        # Find frames with detections
        detection_frames = []
        for frame_id, frame_detections in detections.items():
            if frame_detections:
                frame_number = int(frame_id.replace('frame_', ''))
                max_confidence = max(det[2] for det in frame_detections if len(det) >= 3)
                detection_frames.append((frame_number, max_confidence))
        
        # Sort by confidence and take top frames
        detection_frames.sort(key=lambda x: x[1], reverse=True)
        selected_frames = detection_frames[:summary_frames]
        selected_frames.sort(key=lambda x: x[0])  # Sort by frame number
        
        logger.info(f"Creating summary video with {len(selected_frames)} frames")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame_number, confidence in selected_frames:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frame_id = f'frame_{frame_number:06d}'
                    frame_detections = detections.get(frame_id, [])
                    
                    frame_info = {
                        'frame_number': frame_number,
                        'timestamp': frame_number / fps if fps > 0 else frame_number,
                        'detection_count': len(frame_detections),
                        'max_confidence': confidence
                    }
                    
                    rendered_frame = self.renderer.render_frame(
                        frame, 
                        frame_detections, 
                        frame_info
                    )
                    
                    out.write(rendered_frame)
            
            return output_path
            
        finally:
            cap.release()
            out.release()
    
    def extract_detection_frames(
        self,
        video_path: str,
        detections: Dict[str, List[List[float]]],
        output_dir: str,
        confidence_threshold: Optional[float] = None
    ) -> List[str]:
        """Extract individual frames with detections as images.
        
        Args:
            video_path: Path to input video file
            detections: Detection results
            output_dir: Directory to save extracted frames
            confidence_threshold: Minimum confidence (uses config default if None)
            
        Returns:
            List of paths to extracted frame images
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        extracted_files = []
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for frame_id, frame_detections in detections.items():
                # Filter detections by confidence
                valid_detections = [
                    det for det in frame_detections 
                    if len(det) >= 3 and det[2] >= confidence_threshold
                ]
                
                if not valid_detections:
                    continue
                
                # Get frame number and seek to frame
                frame_number = int(frame_id.replace('frame_', ''))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frame_info = {
                        'frame_number': frame_number,
                        'timestamp': frame_number / fps if fps > 0 else frame_number,
                        'detection_count': len(valid_detections)
                    }
                    
                    rendered_frame = self.renderer.render_frame(
                        frame, 
                        frame_detections, 
                        frame_info
                    )
                    
                    # Save frame
                    output_file = output_path / f"{frame_id}_detections.jpg"
                    cv2.imwrite(str(output_file), rendered_frame)
                    extracted_files.append(str(output_file))
            
            logger.info(f"Extracted {len(extracted_files)} frames to {output_dir}")
            return extracted_files
            
        finally:
            cap.release()
    
    def get_processing_stats(
        self, 
        detections: Dict[str, List[List[float]]]
    ) -> Dict[str, Any]:
        """Get statistics about detection results.
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary with processing statistics
        """
        total_frames = len(detections)
        frames_with_detections = 0
        total_detections = 0
        confidence_scores = []
        
        for frame_detections in detections.values():
            if frame_detections:
                frames_with_detections += 1
                total_detections += len(frame_detections)
                
                for det in frame_detections:
                    if len(det) >= 3:
                        confidence_scores.append(det[2])
        
        stats = {
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'detection_rate': frames_with_detections / total_frames if total_frames > 0 else 0,
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
        }
        
        if confidence_scores:
            stats.update({
                'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                'min_confidence': min(confidence_scores),
                'max_confidence': max(confidence_scores)
            })
        
        return stats
    
    def set_config(self, config: VisualizationConfig):
        """Update visualization configuration.
        
        Args:
            config: New visualization configuration
        """
        self.config = config
        self.renderer.set_config(config)
    
    def reset_renderer(self):
        """Reset renderer state (clears trajectory)."""
        self.renderer.reset()