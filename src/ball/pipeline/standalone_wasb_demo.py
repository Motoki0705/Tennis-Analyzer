#!/usr/bin/env python3
# standalone_wasb_demo.py
# Standalone WASB-SBDT ball tracking demo (first 10 frames)

import argparse
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import logging
from collections import deque
import csv

# Local WASB modules
from wasb_modules import load_default_config, build_tracker
from wasb_modules.pipeline_modules import BallPreprocessor, BallDetector, DetectionPostprocessor
from wasb_modules.drawing_utils import draw_on_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class StandaloneTennisTracker:
    """Standalone tennis ball tracker (first 10 frames only)."""
    
    def __init__(self, args):
        self.args = args
        self.cfg = load_default_config()
        if args.model_path is not None:
            self.cfg.detector.model_path = args.model_path

        # Initialize device
        self._initialize_device()
        
        # Initialize pipeline modules
        self._initialize_pipeline_modules()
        
        self.video_writer = None
        self.video_properties = {}
        self.all_tracking_results = []

    def _initialize_device(self):
        """Initialize device."""
        if self.args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.args.device)
        log.info(f"Using device: {self.device}")

    def _initialize_pipeline_modules(self):
        """Initialize pipeline modules."""
        self.preprocessor = BallPreprocessor(self.cfg)
        self.detector = BallDetector(self.cfg, self.device)
        self.postprocessor = DetectionPostprocessor(self.cfg)
        self.tracker = build_tracker(self.cfg)
        log.info("Pipeline modules initialized.")

    def _initialize_video_io(self):
        """Initialize video input and output."""
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.args.video}")

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
        self.video_writer = cv2.VideoWriter(self.args.output, fourcc, self.video_properties['fps'],
                                            (self.video_properties['width'], self.video_properties['height']))

    def _save_results_as_csv(self):
        """Save tracking results to CSV file."""
        log.info(f"Saving tracking results to {self.args.results_csv}...")
        try:
            with open(self.args.results_csv, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'visible', 'score', 'x', 'y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for frame_idx, result in enumerate(self.all_tracking_results):
                    row = {
                        'frame': frame_idx,
                        'visible': 1 if result.get("visi", False) else 0,
                        'score': result.get("score", 0.0),
                        'x': result.get("x", -1),
                        'y': result.get("y", -1)
                    }
                    writer.writerow(row)
        except IOError as e:
            log.error(f"Failed to write CSV file: {e}")

    def run(self):
        """Run the standalone pipeline on first 10 frames."""
        self._initialize_video_io()
        
        cap = cv2.VideoCapture(self.args.video)
        frames_in = self.preprocessor.frames_in
        frame_history = deque(maxlen=frames_in)
        
        # Process only first 10 frames
        max_frames = min(50, self.video_properties['total_frames'])
        log.info(f"Processing first {max_frames} frames...")
        
        self.tracker.refresh()
        
        # Add dummy results for first frames_in - 1 frames
        for _ in range(frames_in - 1):
            self.all_tracking_results.append(self.tracker.update([]))
        
        pbar = tqdm(total=max_frames, desc="Processing frames")
        
        for frame_idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_history.append(frame)
            if len(frame_history) < frames_in:
                pbar.update(1)
                continue
            
            # Process current frame sequence
            frame_sequence = list(frame_history)
            
            try:
                # Preprocessing
                batch_tensor, batch_meta = self.preprocessor.process_batch([frame_sequence])
                
                # Inference
                with torch.no_grad():
                    batch_preds = self.detector.predict_batch(batch_tensor)
                
                # Postprocessing
                batch_detections = self.postprocessor.process_batch(batch_preds, batch_meta, self.device)
                
                # Tracking
                detections = batch_detections[0]  # Get first (and only) batch item
                tracking_output = self.tracker.update(detections)
                self.all_tracking_results.append(tracking_output)
                
                # Draw and save frame
                output_frame = draw_on_frame(frame, tracking_output)
                self.video_writer.write(output_frame)
                
            except Exception as e:
                log.warning(f"Error processing frame {frame_idx}: {e}")
                # Add dummy tracking result for failed frames
                self.all_tracking_results.append(self.tracker.update([]))
                self.video_writer.write(frame)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Save results
        self._save_results_as_csv()
        
        # Clean up
        if self.video_writer:
            self.video_writer.release()
        
        log.info(f"Output video saved to: {self.args.output}")
        log.info(f"Tracking results saved to: {self.args.results_csv}")


def main():
    parser = argparse.ArgumentParser(description="Standalone WASB-SBDT Tennis Ball Tracking (First 10 frames)")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--output", default="standalone_wasb_output.mp4", help="Output video file")
    parser.add_argument("--results_csv", default="standalone_tracking_results.csv", help="Output CSV file for tracking results")
    parser.add_argument("--model_path", default=r"C:\Users\kamim\code\tennis_systems\third_party\WASB_SBDT\pretrained_weights\wasb_tennis_best.pth.tar", help="Path to a trained model (.pth.tar or .pth)")
    parser.add_argument("--device", default="auto", help="Device to use (cuda/cpu/auto)")
    args = parser.parse_args()

    try:
        tracker_pipeline = StandaloneTennisTracker(args)
        tracker_pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()