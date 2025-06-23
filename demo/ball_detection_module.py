#!/usr/bin/env python3
"""
Ball Detection Module Demo

This demo showcases the ball detection module's capabilities with different models
and processing options. It provides an interactive Gradio interface for testing
ball detection on uploaded videos.
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.ball.ball_detection_module import create_ball_detection_module
except ImportError as e:
    print(f"Warning: Could not import ball detection module: {e}")
    print("Make sure the ball detection module is properly installed.")
    sys.exit(1)


def detect_balls_in_video(
    video_path: str,
    model_path: str,
    model_type: str = "auto",
    device: str = "auto",
    max_frames: int = 100,
    batch_size: int = 16,
    confidence_threshold: float = 0.5
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect balls in uploaded video using the ball detection module.
    
    Args:
        video_path: Path to input video file
        model_path: Path to model checkpoint
        model_type: Type of model ("auto", "lite_tracknet", "wasb_sbdt")
        device: Device for inference ("auto", "cpu", "cuda")
        max_frames: Maximum number of frames to process
        batch_size: Batch size for processing
        confidence_threshold: Minimum confidence threshold for detections
    
    Returns:
        Tuple of (status_message, results_dict)
    """
    try:
        # Validate inputs
        if not os.path.exists(video_path):
            return "Error: Video file not found", {}
        
        if not os.path.exists(model_path):
            return "Error: Model file not found", {}
        
        # Create detector
        detector = create_ball_detection_module(
            model_path=model_path,
            model_type=model_type,
            device=device
        )
        
        # Get model info
        model_info = detector.get_model_info()
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_data = []
        frame_idx = 0
        all_detections = {}
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while frame_idx < min(max_frames, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            metadata = {
                'frame_id': f'frame_{frame_idx:06d}',
                'timestamp': frame_idx / fps,
                'frame_number': frame_idx
            }
            frame_data.append((frame_rgb, metadata))
            frame_idx += 1
            
            # Process in batches
            if len(frame_data) >= batch_size:
                batch_detections = detector.detect_balls(frame_data)
                all_detections.update(batch_detections)
                frame_data = []
        
        # Process remaining frames
        if frame_data:
            batch_detections = detector.detect_balls(frame_data)
            all_detections.update(batch_detections)
        
        cap.release()
        
        # Filter by confidence threshold
        filtered_detections = {}
        total_detections = 0
        
        for frame_id, balls in all_detections.items():
            filtered_balls = [
                ball for ball in balls 
                if len(ball) >= 3 and ball[2] >= confidence_threshold
            ]
            if filtered_balls:
                filtered_detections[frame_id] = filtered_balls
                total_detections += len(filtered_balls)
        
        # Prepare results
        results = {
            'model_info': model_info,
            'video_info': {
                'total_frames': total_frames,
                'processed_frames': frame_idx,
                'fps': fps,
                'duration': total_frames / fps if fps > 0 else 0
            },
            'detection_summary': {
                'total_detections': total_detections,
                'frames_with_detections': len(filtered_detections),
                'detection_rate': len(filtered_detections) / frame_idx if frame_idx > 0 else 0
            },
            'detections': filtered_detections
        }
        
        status = f"✅ Successfully processed {frame_idx} frames. Found {total_detections} ball detections in {len(filtered_detections)} frames."
        
        return status, results
        
    except Exception as e:
        error_msg = f"❌ Error processing video: {str(e)}"
        print(f"Error details: {e}")
        return error_msg, {}


def format_results_display(results: Dict[str, Any]) -> str:
    """Format detection results for display."""
    if not results:
        return "No results to display."
    
    display_text = []
    
    # Model information
    if 'model_info' in results:
        model_info = results['model_info']
        display_text.append("## Model Information")
        for key, value in model_info.items():
            display_text.append(f"- **{key}**: {value}")
        display_text.append("")
    
    # Video information
    if 'video_info' in results:
        video_info = results['video_info']
        display_text.append("## Video Information")
        display_text.append(f"- **Total Frames**: {video_info.get('total_frames', 'N/A')}")
        display_text.append(f"- **Processed Frames**: {video_info.get('processed_frames', 'N/A')}")
        display_text.append(f"- **FPS**: {video_info.get('fps', 'N/A'):.2f}")
        display_text.append(f"- **Duration**: {video_info.get('duration', 'N/A'):.2f}s")
        display_text.append("")
    
    # Detection summary
    if 'detection_summary' in results:
        summary = results['detection_summary']
        display_text.append("## Detection Summary")
        display_text.append(f"- **Total Detections**: {summary.get('total_detections', 0)}")
        display_text.append(f"- **Frames with Detections**: {summary.get('frames_with_detections', 0)}")
        display_text.append(f"- **Detection Rate**: {summary.get('detection_rate', 0):.2%}")
        display_text.append("")
    
    # Sample detections
    if 'detections' in results and results['detections']:
        display_text.append("## Sample Detections")
        detections = results['detections']
        sample_frames = list(detections.keys())[:5]  # Show first 5 frames
        
        for frame_id in sample_frames:
            balls = detections[frame_id]
            display_text.append(f"**{frame_id}**:")
            for i, ball in enumerate(balls):
                if len(ball) >= 3:
                    x, y, conf = ball[:3]
                    display_text.append(f"  - Ball {i+1}: Position ({x:.3f}, {y:.3f}), Confidence: {conf:.3f}")
            display_text.append("")
    
    return "\n".join(display_text)


def create_demo_interface():
    """Create Gradio interface for ball detection demo."""
    
    # Check for available models
    checkpoint_dir = project_root / "checkpoints" / "ball"
    available_models = []
    
    if checkpoint_dir.exists():
        for model_file in checkpoint_dir.rglob("*.ckpt"):
            available_models.append(str(model_file))
        for model_file in checkpoint_dir.rglob("*.pth"):
            available_models.append(str(model_file))
    
    # Check third-party models
    third_party_dir = project_root / "third_party" / "WASB-SBDT"
    if third_party_dir.exists():
        for model_file in third_party_dir.rglob("*.pth"):
            available_models.append(str(model_file))
        for model_file in third_party_dir.rglob("*.pth.tar"):
            available_models.append(str(model_file))
    
    if not available_models:
        available_models = ["No models found - please add model files to checkpoints/ball/"]
    
    with gr.Blocks(title="Ball Detection Module Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎾 Ball Detection Module Demo
        
        Upload a tennis video to detect ball positions using the modular ball detection system.
        The system supports multiple model types including LiteTrackNet and WASB-SBDT.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Configuration")
                
                video_input = gr.File(
                    label="Upload Video",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    file_count="single"
                )
                
                model_path = gr.Dropdown(
                    label="Model Path",
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    allow_custom_value=True
                )
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        label="Model Type",
                        choices=["auto", "lite_tracknet", "wasb_sbdt"],
                        value="auto"
                    )
                    
                    device = gr.Dropdown(
                        label="Device",
                        choices=["auto", "cpu", "cuda"],
                        value="auto"
                    )
                
                with gr.Row():
                    max_frames = gr.Slider(
                        label="Max Frames",
                        minimum=10,
                        maximum=1000,
                        value=100,
                        step=10
                    )
                    
                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=64,
                        value=16,
                        step=1
                    )
                
                confidence_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05
                )
                
                detect_btn = gr.Button("🎾 Detect Balls", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=3
                )
                
                results_output = gr.Markdown(
                    label="Detection Results",
                    value="Upload a video and click 'Detect Balls' to see results here."
                )
        
        # Event handlers
        def process_video(video_file, model_path, model_type, device, max_frames, batch_size, confidence_threshold):
            if video_file is None:
                return "Please upload a video file.", "No video uploaded."
            
            status, results = detect_balls_in_video(
                video_path=video_file.name,
                model_path=model_path,
                model_type=model_type,
                device=device,
                max_frames=int(max_frames),
                batch_size=int(batch_size),
                confidence_threshold=confidence_threshold
            )
            
            formatted_results = format_results_display(results)
            return status, formatted_results
        
        detect_btn.click(
            fn=process_video,
            inputs=[
                video_input, model_path, model_type, device,
                max_frames, batch_size, confidence_threshold
            ],
            outputs=[status_output, results_output]
        )
        
        # Examples section
        gr.Markdown("""
        ### Usage Tips
        
        1. **Model Selection**: Use "auto" to let the system detect model type from file extension
        2. **Device**: Use "auto" for automatic GPU detection, "cpu" for CPU-only inference
        3. **Batch Size**: Larger batches are faster but use more memory
        4. **Confidence Threshold**: Lower values detect more balls but may include false positives
        5. **Max Frames**: Limit processing for long videos to reduce computation time
        
        ### Supported Formats
        - **Video**: MP4, AVI, MOV, MKV
        - **Models**: PyTorch Lightning checkpoints (.ckpt), PyTorch weights (.pth, .pth.tar)
        """)
    
    return demo


def main():
    """Main function to run the demo."""
    print("Starting Ball Detection Module Demo...")
    
    # Create and launch demo
    demo = create_demo_interface()
    
    # Launch with appropriate settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    main()