#!/usr/bin/env python3
"""
VideoSwinTransformer ball detection inference demo.
"""

import hydra
from omegaconf import DictConfig
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import logging
import csv
import time
from typing import Dict, List, Any

# Local imports
from .custom_modules import VideoSwinBallTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class VideoSwinTennisDetector:
    """VideoSwinTransformer tennis ball detector."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # Initialize device
        self._initialize_device()
        
        # Initialize tracker
        self._initialize_tracker()
        
        self.video_writer = None
        self.heatmap_writer = None
        self.video_properties = {}
        self.all_detection_results = []
        self.show_heatmap = cfg.visualization.show_heatmap
        
    def _initialize_device(self):
        """Initialize device."""
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        log.info(f"Using device: {self.device}")
    
    def _initialize_tracker(self):
        """Initialize VideoSwin tracker."""
        self.tracker = VideoSwinBallTracker(self.cfg, self.device)
        log.info("VideoSwinTransformer tracker initialized")
    
    def _initialize_video_io(self):
        """Initialize video input and output."""
        cap = cv2.VideoCapture(self.cfg.io.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.cfg.io.video}")

        self.video_properties = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        
        log.info(f"Video: {self.video_properties['width']}x{self.video_properties['height']}, "
                 f"{self.video_properties['fps']:.2f}fps, {self.video_properties['total_frames']} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.cfg.io.output, 
            fourcc, 
            self.video_properties['fps'],
            (self.video_properties['width'], self.video_properties['height'])
        )
        
        # Initialize heatmap video writer if enabled
        if self.show_heatmap:
            heatmap_output = self.cfg.io.output.replace('.mp4', '_heatmap.mp4')
            self.heatmap_writer = cv2.VideoWriter(
                heatmap_output,
                fourcc,
                self.video_properties['fps'],
                (self.video_properties['width'], self.video_properties['height'])
            )
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detections on frame."""
        frame_viz = frame.copy()
        
        if not detections:
            return frame_viz
        
        # Draw the best detection
        detection = detections[0]
        x, y = int(detection['x']), int(detection['y'])
        confidence = detection['confidence']
        
        # Draw circle
        color = self.cfg.visualization.colors.ball
        radius = self.cfg.visualization.circle_radius
        cv2.circle(frame_viz, (x, y), radius, color, 2)
        
        # Draw confidence score if enabled
        if self.cfg.visualization.show_score:
            text = f"{confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2
            text_y = y - radius - 10
            
            # Draw background for text
            cv2.rectangle(frame_viz, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (0, 0, 0), -1)
            cv2.putText(frame_viz, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame_viz
    
    def _create_heatmap_visualization(self, heatmap: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Create heatmap visualization overlay."""
        # Resize heatmap to original frame size
        h, w = original_shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Normalize heatmap to 0-255
        heatmap_norm = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap (red for high values)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def _blend_heatmap_with_frame(self, frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Blend heatmap with original frame."""
        heatmap_viz = self._create_heatmap_visualization(heatmap, frame.shape)
        
        # Blend frame and heatmap
        blended = cv2.addWeighted(frame, 1 - alpha, heatmap_viz, alpha, 0)
        
        return blended
    
    def _save_results_as_csv(self):
        """Save detection results to CSV file."""
        log.info(f"Saving detection results to {self.cfg.io.results_csv}...")
        try:
            with open(self.cfg.io.results_csv, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'visible', 'confidence', 'x', 'y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for frame_idx, result in enumerate(self.all_detection_results):
                    if result is None or not result:
                        # No detection
                        row = {
                            'frame': frame_idx,
                            'visible': 0,
                            'confidence': 0.0,
                            'x': -1,
                            'y': -1
                        }
                    else:
                        # Best detection
                        detection = result[0]
                        row = {
                            'frame': frame_idx,
                            'visible': 1,
                            'confidence': detection['confidence'],
                            'x': detection['x'],
                            'y': detection['y']
                        }
                    writer.writerow(row)
        except IOError as e:
            log.error(f"Failed to write CSV file: {e}")
    
    def _memory_cleanup(self, frame_idx: int):
        """Perform periodic memory cleanup."""
        if frame_idx % self.cfg.memory.clear_cache_interval == 0:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def run(self):
        """Run the VideoSwinTransformer inference pipeline."""
        if self.cfg.io.video is None:
            raise ValueError("Video path is required. Use --io.video=path/to/video.mp4")
        
        # Initialize video I/O
        self._initialize_video_io()
        
        # Open video capture
        cap = cv2.VideoCapture(self.cfg.io.video)
        
        try:
            # Determine frame range
            max_frames = self.cfg.processing.max_frames
            total_frames = min(max_frames or self.video_properties['total_frames'], 
                             self.video_properties['total_frames'])
            
            log.info(f"Processing {total_frames} frames...")
            
            # Process frames
            frame_idx = 0
            with tqdm(total=total_frames, desc="Processing") as pbar:
                while frame_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame with tracker
                    if self.show_heatmap:
                        result = self.tracker.process_frame(frame, return_heatmap=True)
                        if result is not None:
                            detections, heatmap = result
                        else:
                            detections, heatmap = None, None
                    else:
                        detections = self.tracker.process_frame(frame, return_heatmap=False)
                        heatmap = None
                    
                    # Store results (even if None)
                    self.all_detection_results.append(detections)
                    
                    # Draw visualizations if enabled
                    if self.cfg.visualization.enabled:
                        if detections:
                            # Filter detections by confidence threshold
                            valid_detections = [
                                d for d in detections 
                                if d['confidence'] >= self.cfg.detection.confidence_threshold
                            ]
                            frame_viz = self._draw_detections(frame, valid_detections)
                        else:
                            frame_viz = frame
                        
                        # Write frame to output video
                        self.video_writer.write(frame_viz)
                        
                        # Write heatmap video if enabled
                        if self.show_heatmap and heatmap is not None:
                            heatmap_viz = self._blend_heatmap_with_frame(frame, heatmap)
                            self.heatmap_writer.write(heatmap_viz)
                        elif self.show_heatmap:
                            # Write original frame when no heatmap available
                            self.heatmap_writer.write(frame)
                    else:
                        # Write original frame
                        self.video_writer.write(frame)
                        if self.show_heatmap:
                            self.heatmap_writer.write(frame)
                    
                    # Memory cleanup
                    self._memory_cleanup(frame_idx)
                    
                    frame_idx += 1
                    pbar.update(1)
                    
                    # Update progress with stats
                    if frame_idx % 100 == 0:
                        stats = self.tracker.get_stats()
                        pbar.set_postfix({
                            'FPS': f"{stats['fps']:.1f}",
                            'Avg_time': f"{stats['avg_inference_time']:.3f}s"
                        })
        
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            if self.heatmap_writer:
                self.heatmap_writer.release()
        
        # Save results
        self._save_results_as_csv()
        
        # Print final statistics
        stats = self.tracker.get_stats()
        log.info("Processing completed!")
        log.info(f"Total frames processed: {stats['processed_frames']}")
        log.info(f"Total inference time: {stats['total_inference_time']:.2f}s")
        log.info(f"Average inference time: {stats['avg_inference_time']:.4f}s")
        log.info(f"Average FPS: {stats['fps']:.2f}")
        log.info(f"Output video saved: {self.cfg.io.output}")
        if self.show_heatmap:
            heatmap_output = self.cfg.io.output.replace('.mp4', '_heatmap.mp4')
            log.info(f"Heatmap video saved: {heatmap_output}")
        log.info(f"Results CSV saved: {self.cfg.io.results_csv}")


@hydra.main(version_base=None, config_path="../../../configs/infer/ball", config_name="video_swin_transformer")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    try:
        detector = VideoSwinTennisDetector(cfg)
        detector.run()
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()