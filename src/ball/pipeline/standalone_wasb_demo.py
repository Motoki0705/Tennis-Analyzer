#!/usr/bin/env python3
# standalone_wasb_demo.py
# Standalone WASB-SBDT ball detection demo (simplified without tracking)

import hydra
from omegaconf import DictConfig
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import logging
from collections import deque
import csv

# Local WASB modules
from wasb_modules import load_default_config
from wasb_modules.pipeline_modules import BallPreprocessor, BallDetector, DetectionPostprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class StandaloneTennisDetector:
    """Standalone tennis ball detector (simplified without tracking)."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg_hydra = cfg
        self.cfg = load_default_config()
        if cfg.ball.model_path is not None:
            self.cfg.detector.model_path = cfg.ball.model_path

        # Initialize device
        self._initialize_device()
        
        # Initialize pipeline modules
        self._initialize_pipeline_modules()
        
        self.video_writer = None
        self.video_properties = {}
        self.all_detection_results = []

    def _initialize_device(self):
        """Initialize device."""
        if self.cfg_hydra.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg_hydra.device)
        log.info(f"Using device: {self.device}")

    def _initialize_pipeline_modules(self):
        """Initialize pipeline modules."""
        self.preprocessor = BallPreprocessor(self.cfg)
        self.detector = BallDetector(self.cfg, self.device)
        self.postprocessor = DetectionPostprocessor(self.cfg)
        log.info("Pipeline modules initialized (no tracking).")

    def _initialize_video_io(self):
        """Initialize video input and output."""
        cap = cv2.VideoCapture(self.cfg_hydra.io.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.cfg_hydra.io.video}")

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
        self.video_writer = cv2.VideoWriter(self.cfg_hydra.io.output, fourcc, self.video_properties['fps'],
                                            (self.video_properties['width'], self.video_properties['height']))

    def _save_results_as_csv(self):
        """Save detection results to CSV file."""
        log.info(f"Saving detection results to {self.cfg_hydra.io.results_csv}...")
        try:
            with open(self.cfg_hydra.io.results_csv, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'visible', 'score', 'x', 'y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for frame_idx, result in enumerate(self.all_detection_results):
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
        """Run the standalone pipeline with simple detection."""
        self._initialize_video_io()
        
        cap = cv2.VideoCapture(self.cfg_hydra.io.video)
        frames_in = self.preprocessor.frames_in
        frame_history = deque(maxlen=frames_in)
        
        # Process only first N frames
        processing_limit = self.cfg_hydra.processing.get('max_frames', 50)
        max_frames = min(processing_limit, self.video_properties['total_frames']) if processing_limit else self.video_properties['total_frames']
        log.info(f"Processing first {max_frames} frames...")
        
        # Add dummy results for first frames_in - 1 frames (warm-up period)
        for _ in range(frames_in - 1):
            self.all_detection_results.append({'visi': False, 'x': -1, 'y': -1, 'score': 0.0})
        
        pbar = tqdm(total=max_frames, desc="Processing frames")
        
        for frame_idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_history.append(frame)
            
            # Skip until we have enough frames for sequence
            if len(frame_history) < frames_in:
                output_frame = frame.copy()
                if self.cfg_hydra.visualization.enabled:
                    cv2.putText(output_frame, "Warming up...", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                self.video_writer.write(output_frame)
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
                
                # Get detection result
                detection_result = batch_detections[0]  # Get first (and only) batch item
                self.all_detection_results.append(detection_result)
                
                # Draw and save frame
                output_frame = frame.copy()
                if self.cfg_hydra.visualization.enabled and detection_result['visi']:
                    x, y = int(detection_result['x']), int(detection_result['y'])
                    score = detection_result['score']
                    # Draw ball as red circle
                    cv2.circle(output_frame, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(output_frame, f'{score:.2f}', (x+10, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                self.video_writer.write(output_frame)
                
            except Exception as e:
                log.warning(f"Error processing frame {frame_idx}: {e}")
                # Add dummy detection result for failed frames
                self.all_detection_results.append({'visi': False, 'x': -1, 'y': -1, 'score': 0.0})
                self.video_writer.write(frame)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Save results
        self._save_results_as_csv()
        
        # Clean up
        if self.video_writer:
            self.video_writer.release()
        
        log.info(f"Output video saved to: {self.cfg_hydra.io.output}")
        log.info(f"Detection results saved to: {self.cfg_hydra.io.results_csv}")


@hydra.main(config_path="../../../configs/infer/ball", config_name="pipeline_demo", version_base=None)
def main(cfg: DictConfig) -> None:
    # Validate required config
    if cfg.io.video is None:
        raise ValueError("Video path is required. Please set io.video in config or via command line.")
    
    # Set up logging from config
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format=cfg.logging.format
    )
    
    try:
        detector_pipeline = StandaloneTennisDetector(cfg)
        detector_pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()