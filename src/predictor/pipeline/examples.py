"""
Example usage of the parallel video processing pipeline.

This module provides example scripts demonstrating how to use
the VideoPipeline for various scenarios and configurations.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from .video_pipeline import VideoPipeline
from .config import PipelineConfig, HIGH_PERFORMANCE_CONFIG, MEMORY_EFFICIENT_CONFIG, REALTIME_CONFIG
from ..visualization import VisualizationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_processing():
    """Basic video processing example."""
    print("🎾 Basic Video Processing Example")
    print("=" * 50)
    
    # Configuration
    video_path = "samples/tennis_match.mp4"
    model_path = "checkpoints/wasb_model.pth"
    output_path = "output/annotated_match.mp4"
    
    # Create pipeline with default configuration
    pipeline = VideoPipeline()
    
    # Detector configuration
    detector_config = {
        "model_type": "wasb_sbdt",
        "model_path": model_path,
        "device": "auto"
    }
    
    # Progress callback
    def progress_callback(progress: float):
        print(f"Progress: {progress*100:.1f}%", end='\r')
    
    try:
        # Process video
        result = pipeline.process_video(
            video_path=video_path,
            detector_config=detector_config,
            output_path=output_path,
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Processing completed!")
        print(f"📁 Output: {result['output_path']}")
        print(f"⏱️  Time: {result['processing_time']:.2f}s")
        print(f"🎞️  FPS: {result['average_fps']:.2f}")
        print(f"🎯 Frames: {result['frames_processed']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_high_performance_processing():
    """High performance processing example."""
    print("🚀 High Performance Processing Example")
    print("=" * 50)
    
    # Use high performance configuration
    config = HIGH_PERFORMANCE_CONFIG
    pipeline = VideoPipeline(config)
    
    video_path = "samples/tennis_match.mp4"
    model_path = "checkpoints/lite_tracknet.ckpt"
    output_path = "output/high_perf_match.mp4"
    
    detector_config = {
        "model_type": "lite_tracknet",
        "model_path": model_path,
        "device": "cuda"
    }
    
    # Custom visualization settings
    vis_config = VisualizationConfig(
        ball_radius=12,
        trajectory_length=25,
        enable_smoothing=True,
        enable_prediction=True
    )
    
    try:
        result = pipeline.process_video(
            video_path=video_path,
            detector_config=detector_config,
            output_path=output_path,
            visualization_config=vis_config
        )
        
        print(f"🏆 High performance processing completed!")
        print(f"⚡ Average FPS: {result['average_fps']:.2f}")
        print(f"📊 Stats: {result['stats']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_memory_efficient_processing():
    """Memory efficient processing example."""
    print("💾 Memory Efficient Processing Example")
    print("=" * 50)
    
    # Use memory efficient configuration
    config = MEMORY_EFFICIENT_CONFIG
    pipeline = VideoPipeline(config)
    
    video_path = "samples/large_video.mp4"
    model_path = "checkpoints/wasb_model.pth"
    output_path = "output/memory_efficient.mp4"
    
    detector_config = {
        "model_type": "wasb_sbdt",
        "model_path": model_path,
        "device": "cuda"
    }
    
    try:
        result = pipeline.process_video(
            video_path=video_path,
            detector_config=detector_config,
            output_path=output_path
        )
        
        print(f"💚 Memory efficient processing completed!")
        print(f"🔋 Memory optimized for large videos")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_realtime_processing():
    """Real-time processing example."""
    print("⚡ Real-time Processing Example")
    print("=" * 50)
    
    # Use real-time configuration
    config = REALTIME_CONFIG
    pipeline = VideoPipeline(config)
    
    video_path = "samples/live_stream.mp4"
    model_path = "checkpoints/lite_tracknet.ckpt"
    output_path = "output/realtime_stream.mp4"
    
    detector_config = {
        "model_type": "lite_tracknet",
        "model_path": model_path,
        "device": "cuda"
    }
    
    try:
        result = pipeline.process_video(
            video_path=video_path,
            detector_config=detector_config,
            output_path=output_path
        )
        
        print(f"⚡ Real-time processing completed!")
        print(f"🎮 Optimized for low latency")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_async_processing():
    """Asynchronous processing example."""
    print("🔄 Asynchronous Processing Example")
    print("=" * 50)
    
    pipeline = VideoPipeline()
    
    video_path = "samples/tennis_match.mp4"
    model_path = "checkpoints/wasb_model.pth"
    output_path = "output/async_match.mp4"
    
    detector_config = {
        "model_type": "wasb_sbdt",
        "model_path": model_path,
        "device": "auto"
    }
    
    try:
        # Start async processing
        async_result = pipeline.process_video_async(
            video_path=video_path,
            detector_config=detector_config,
            output_path=output_path
        )
        
        print("🔄 Processing started asynchronously...")
        
        # Monitor progress
        while not async_result.is_completed():
            progress = async_result.get_progress()
            print(f"Progress: {progress*100:.1f}%", end='\r')
            import time
            time.sleep(0.5)
        
        # Get final result
        result = async_result.get_result()
        
        print(f"\n✅ Async processing completed!")
        print(f"📁 Output: {result['output_path']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_batch_frame_processing():
    """Batch frame processing example."""
    print("📦 Batch Frame Processing Example")
    print("=" * 50)
    
    pipeline = VideoPipeline()
    
    # Simulate frame batch
    import numpy as np
    frames = []
    for i in range(10):
        # Create dummy frames (replace with actual frame loading)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metadata = {'frame_id': f'frame_{i:06d}', 'timestamp': i * 0.033}
        frames.append({'frame': frame, 'metadata': metadata})
    
    detector_config = {
        "model_type": "lite_tracknet",
        "model_path": "checkpoints/lite_tracknet.ckpt",
        "device": "auto"
    }
    
    try:
        detections = pipeline.process_frame_batch(frames, detector_config)
        
        print(f"📦 Batch processing completed!")
        print(f"🎯 Detected frames: {len(detections)}")
        
        for frame_id, detection_list in detections.items():
            if detection_list:
                print(f"  {frame_id}: {len(detection_list)} detections")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_performance_benchmark():
    """Performance benchmark example."""
    print("🏁 Performance Benchmark Example")
    print("=" * 50)
    
    pipeline = VideoPipeline()
    
    video_path = "samples/test_video.mp4"
    detector_config = {
        "model_type": "wasb_sbdt",
        "model_path": "checkpoints/wasb_model.pth",
        "device": "cuda"
    }
    
    try:
        # Benchmark different configurations
        results = pipeline.benchmark_performance(
            video_path=video_path,
            detector_config=detector_config
        )
        
        print("🏁 Benchmark Results:")
        print("-" * 50)
        
        for config_name, result in results.items():
            if 'error' in result:
                print(f"❌ {config_name}: {result['error']}")
            else:
                print(f"✅ {config_name}:")
                print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
                print(f"   🎞️  FPS: {result['average_fps']:.2f}")
                print(f"   🎯 Frames: {result['frames_processed']}")
                print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_custom_configuration():
    """Custom configuration example."""
    print("⚙️ Custom Configuration Example")
    print("=" * 50)
    
    # Create custom configuration
    custom_config = PipelineConfig(
        frame_buffer_size=40,
        tensor_buffer_size=20,
        num_preprocessing_threads=3,
        gpu_batch_size=6,
        enable_visualization=True,
        enable_memory_optimization=True,
        enable_profiling=True,
        log_queue_sizes=True
    )
    
    pipeline = VideoPipeline(custom_config)
    
    video_path = "samples/tennis_match.mp4"
    detector_config = {
        "model_type": "wasb_sbdt",
        "model_path": "checkpoints/wasb_model.pth",
        "device": "cuda"
    }
    
    try:
        result = pipeline.process_video(
            video_path=video_path,
            detector_config=detector_config,
            output_path="output/custom_config.mp4"
        )
        
        print(f"⚙️ Custom configuration processing completed!")
        print(f"📊 Detailed stats: {result['stats']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run all examples (comment out as needed)
    
    print("🎾 Video Processing Pipeline Examples")
    print("=" * 60)
    
    # Uncomment the examples you want to run:
    
    # example_basic_processing()
    # example_high_performance_processing() 
    # example_memory_efficient_processing()
    # example_realtime_processing()
    # example_async_processing()
    # example_batch_frame_processing()
    # example_performance_benchmark()
    # example_custom_configuration()
    
    print("\n✨ All examples completed!")
    print("💡 Uncomment the functions above to run specific examples.") 