"""
Example usage of the BallDetectionModule

This script demonstrates how to use the modular ball detection system
with different model types and configurations.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict

from .ball_detection_module import BallDetectionModule, create_ball_detection_module

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_video_frames(video_path: str, max_frames: int = 100) -> List[Tuple[np.array, dict]]:
    """Load video frames with metadata.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        
    Returns:
        List of (frame, metadata) tuples
    """
    frame_data = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create metadata
        metadata = {
            'frame_id': f"frame_{frame_idx:06d}",
            'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
            'frame_number': frame_idx,
            'video_path': video_path
        }
        
        frame_data.append((frame_rgb, metadata))
        frame_idx += 1
    
    cap.release()
    logger.info(f"Loaded {len(frame_data)} frames from {video_path}")
    return frame_data


def example_lite_tracknet_detection():
    """Example using LiteTrackNet model."""
    logger.info("=== LiteTrackNet Detection Example ===")
    
    # Model configuration
    model_path = "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
    
    try:
        # Create detection module
        detector = create_ball_detection_module(
            model_path=model_path,
            device="auto",
            model_type="lite_tracknet"
        )
        
        # Load sample video frames
        video_path = "datasets/test/input_video2.mp4"
        frame_data = load_video_frames(video_path, max_frames=50)
        
        if len(frame_data) < 3:
            logger.warning("Need at least 3 frames for LiteTrackNet detection")
            return
        
        # Run detection
        logger.info("Running ball detection...")
        detections = detector.detect_balls(frame_data)
        
        # Display results
        logger.info(f"Detected balls in {len(detections)} frames")
        for frame_id, balls in detections.items():
            if balls:  # Only show frames with detections
                for ball in balls:
                    x, y, conf = ball
                    logger.info(f"Frame {frame_id}: Ball at ({x:.3f}, {y:.3f}) confidence {conf:.3f}")
        
        # Model info
        model_info = detector.get_model_info()
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"LiteTrackNet detection failed: {e}")


def example_wasb_sbdt_detection():
    """Example using WASB-SBDT model."""
    logger.info("=== WASB-SBDT Detection Example ===")
    
    # Model configuration
    model_path = "third_party/WASB-SBDT/pretrained_weights/wasb_tennis_best.pth.tar"
    
    try:
        # Create detection module
        detector = create_ball_detection_module(
            model_path=model_path,
            device="auto",
            model_type="wasb_sbdt"
        )
        
        # Load sample video frames
        video_path = "datasets/test/input_video2.mp4"
        frame_data = load_video_frames(video_path, max_frames=50)
        
        # Run detection
        logger.info("Running ball detection...")
        detections = detector.detect_balls(frame_data)
        
        # Display results
        logger.info(f"Detected balls in {len(detections)} frames")
        for frame_id, balls in detections.items():
            if balls:
                for ball in balls:
                    x, y, conf = ball
                    logger.info(f"Frame {frame_id}: Ball at ({x:.3f}, {y:.3f}) confidence {conf:.3f}")
        
        # Model info
        model_info = detector.get_model_info()
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"WASB-SBDT detection failed: {e}")


def example_batch_processing():
    """Example of batch processing with different frame window sizes."""
    logger.info("=== Batch Processing Example ===")
    
    model_path = "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
    
    try:
        detector = create_ball_detection_module(model_path, model_type="lite_tracknet")
        
        # Create synthetic frame data for demonstration
        frame_data = []
        for i in range(10):
            # Create dummy frame (640x360 RGB)
            frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            metadata = {
                'frame_id': f"synthetic_frame_{i:03d}",
                'frame_number': i,
                'timestamp': i * 0.033  # 30 FPS
            }
            frame_data.append((frame, metadata))
        
        # Process frames step by step
        logger.info("Step 1: Preprocessing")
        model_inputs = detector.preprocess(frame_data)
        logger.info(f"Preprocessed {len(frame_data)} frames into {len(model_inputs)} model inputs")
        
        logger.info("Step 2: Inference")
        inference_results = detector.infer(model_inputs)
        logger.info(f"Generated {len(inference_results)} inference results")
        
        logger.info("Step 3: Postprocessing")
        detections = detector.postprocess(inference_results)
        logger.info(f"Final detections for {len(detections)} frames")
        
        # Show results
        for frame_id, balls in detections.items():
            if balls:
                ball = balls[0]  # Take first detection
                logger.info(f"{frame_id}: Ball at ({ball[0]:.3f}, {ball[1]:.3f}) conf={ball[2]:.3f}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")


def example_custom_configuration():
    """Example with custom configuration and error handling."""
    logger.info("=== Custom Configuration Example ===")
    
    # Test with invalid model path
    try:
        detector = create_ball_detection_module(
            model_path="nonexistent_model.ckpt",
            device="cpu"
        )
    except FileNotFoundError as e:
        logger.warning(f"Expected error caught: {e}")
    
    # Test with insufficient frames
    try:
        model_path = "third_party/WASB-SBDT/pretrained_weights/wasb_tennis_best.pth.tar"
        detector = create_ball_detection_module(model_path, model_type="wasb_sbdt")
        
        # Only provide 2 frames (need 3 for LiteTrackNet)
        frame_data = []
        for i in range(2):
            frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            metadata = {'frame_id': f"frame_{i}"}
            frame_data.append((frame, metadata))
        
        detections = detector.detect_balls(frame_data)
        
    except ValueError as e:
        logger.warning(f"Expected error caught: {e}")


def visualize_detections(frame_data: List[Tuple[np.array, dict]], 
                        detections: Dict[str, List[List[float]]], 
                        output_path: str = "output_with_detections.mp4"):
    """Visualize ball detections on video frames.
    
    Args:
        frame_data: Original frame data
        detections: Detection results
        output_path: Output video path
    """
    if not frame_data:
        logger.warning("No frames to visualize")
        return
    
    # Get frame dimensions
    first_frame = frame_data[0][0]
    height, width = first_frame.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    try:
        for frame, metadata in frame_data:
            frame_id = metadata['frame_id']
            
            # Convert RGB back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw detections if available
            if frame_id in detections:
                for ball in detections[frame_id]:
                    x_norm, y_norm, conf = ball
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_pixel = int(x_norm * width)
                    y_pixel = int(y_norm * height)
                    
                    # Draw ball detection
                    cv2.circle(frame_bgr, (x_pixel, y_pixel), 8, (0, 0, 255), -1)
                    cv2.circle(frame_bgr, (x_pixel, y_pixel), 3, (255, 255, 255), -1)
                    
                    # Add confidence text
                    text = f"Ball: {conf:.2f}"
                    cv2.putText(frame_bgr, text, (x_pixel + 15, y_pixel - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            out.write(frame_bgr)
    
    finally:
        out.release()
    
    logger.info(f"Visualization saved to {output_path}")


def main():
    """Run all examples."""
    logger.info("Ball Detection Module Examples")
    logger.info("=" * 50)
    
    # Run examples
    example_batch_processing()
    print()
    
    example_custom_configuration()
    print()
    
        # Only run model-specific examples if the models exist
    lite_tracknet_path = "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
    if Path(lite_tracknet_path).exists():
        example_lite_tracknet_detection()
        print()
    else:
        logger.info(f"Skipping LiteTrackNet example - model not found: {lite_tracknet_path}")
    
    wasb_sbdt_path = "third_party/WASB-SBDT/models/tennis_model.pth"
    if Path(wasb_sbdt_path).exists():
        example_wasb_sbdt_detection()
        print()
    else:
        logger.info(f"Skipping WASB-SBDT example - model not found: {wasb_sbdt_path}")
    
    logger.info("Examples completed!")


if __name__ == "__main__":
    main()