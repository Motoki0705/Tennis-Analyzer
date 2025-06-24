"""
Main video processing pipeline with parallel execution.

This module provides the main VideoPipeline class that serves as the
unified interface for parallel ball detection and video overlay processing.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from .config import PipelineConfig, HIGH_PERFORMANCE_CONFIG, MEMORY_EFFICIENT_CONFIG, REALTIME_CONFIG
from .async_processor import AsyncVideoProcessor
from ..ball.factory import create_ball_detector
from ..visualization import VideoOverlay, VisualizationConfig

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Main video processing pipeline with parallel execution.
    
    This class provides a unified interface for parallel ball detection
    and video overlay generation with optimized multi-threading for
    maximum GPU utilization and minimal waiting time.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize video pipeline.
        
        Args:
            config: Pipeline configuration (uses default if None)
        """
        self.config = config or PipelineConfig()
        self.async_processor = AsyncVideoProcessor(self.config)
        
        logger.info("VideoPipeline initialized")
    
    def process_video(
        self,
        video_path: str,
        detector_config: Dict[str, Any],
        output_path: Optional[str] = None,
        visualization_config: Optional[VisualizationConfig] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Process video with ball detection and overlay generation.
        
        Args:
            video_path: Path to input video file
            detector_config: Configuration for ball detector
                - model_type: Type of detector ('lite_tracknet', 'wasb_sbdt', 'auto')
                - model_path: Path to model weights file
                - config_path: Path to model config file (optional)
                - device: Device for inference ('cuda', 'cpu', 'auto')
            output_path: Path for output video with overlays (optional)
            visualization_config: Visualization settings (optional)
            progress_callback: Progress callback function (optional)
            
        Returns:
            Dictionary containing processing results:
            - output_path: Path to output video (if generated)
            - processing_time: Total processing time in seconds
            - frames_processed: Number of frames processed
            - average_fps: Average processing FPS
            - stats: Detailed processing statistics
            - detections: Ball detection results
            - error_count: Number of errors encountered
            
        Raises:
            FileNotFoundError: If input video file doesn't exist
            RuntimeError: If video processing fails
            ValueError: If detector configuration is invalid
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting video pipeline processing: {video_path}")
        
        # Validate detector configuration
        self._validate_detector_config(detector_config)
        
        try:
            # Process video with async processor
            results = self.async_processor.process_video(
                video_path=video_path,
                detector_config=detector_config,
                output_path=output_path,
                visualization_config=visualization_config,
                progress_callback=progress_callback
            )
            
            logger.info(
                f"Pipeline completed successfully: "
                f"{results['frames_processed']} frames in {results['processing_time']:.2f}s "
                f"({results['average_fps']:.2f} FPS)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise
    
    def process_video_async(
        self,
        video_path: str,
        detector_config: Dict[str, Any],
        output_path: Optional[str] = None,
        visualization_config: Optional[VisualizationConfig] = None
    ) -> 'AsyncVideoPipelineResult':
        """Start asynchronous video processing.
        
        Args:
            video_path: Path to input video file
            detector_config: Detector configuration
            output_path: Path for output video (optional)
            visualization_config: Visualization settings (optional)
            
        Returns:
            AsyncVideoPipelineResult object for monitoring progress
        """
        return AsyncVideoPipelineResult(
            self, video_path, detector_config, output_path, visualization_config
        )
    
    def process_frame_batch(
        self,
        frames: List[Dict[str, Any]],
        detector_config: Dict[str, Any]
    ) -> Dict[str, List[List[float]]]:
        """Process a batch of frames for ball detection.
        
        Args:
            frames: List of frame dictionaries with 'frame' and 'metadata' keys
            detector_config: Detector configuration
            
        Returns:
            Dictionary mapping frame_id to detection results
        """
        # Initialize detector
        detector = create_ball_detector(**detector_config)
        
        # Convert to expected format
        frame_data = [(frame['frame'], frame['metadata']) for frame in frames]
        
        # Process frames
        detections = detector.detect_balls(frame_data)
        
        return detections
    
    def benchmark_performance(
        self,
        video_path: str,
        detector_config: Dict[str, Any],
        configs_to_test: Optional[List[PipelineConfig]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark different pipeline configurations.
        
        Args:
            video_path: Path to test video
            detector_config: Detector configuration
            configs_to_test: List of configurations to benchmark (optional)
            
        Returns:
            Dictionary mapping config names to performance results
        """
        if configs_to_test is None:
            configs_to_test = [
                ("high_performance", HIGH_PERFORMANCE_CONFIG),
                ("memory_efficient", MEMORY_EFFICIENT_CONFIG),
                ("realtime", REALTIME_CONFIG),
                ("default", PipelineConfig())
            ]
        else:
            configs_to_test = [(f"config_{i}", config) for i, config in enumerate(configs_to_test)]
        
        results = {}
        
        for config_name, config in configs_to_test:
            logger.info(f"Benchmarking configuration: {config_name}")
            
            # Create pipeline with specific config
            pipeline = VideoPipeline(config)
            
            try:
                # Process first 100 frames only for benchmarking
                result = pipeline.process_video(
                    video_path=video_path,
                    detector_config=detector_config,
                    output_path=None  # No output for benchmarking
                )
                
                results[config_name] = {
                    'processing_time': result['processing_time'],
                    'average_fps': result['average_fps'],
                    'frames_processed': result['frames_processed'],
                    'stats': result['stats'],
                    'config': config.to_dict()
                }
                
            except Exception as e:
                logger.error(f"Benchmark failed for {config_name}: {e}")
                results[config_name] = {'error': str(e)}
        
        return results
    
    def _validate_detector_config(self, detector_config: Dict[str, Any]):
        """Validate detector configuration."""
        required_keys = ['model_path']
        
        for key in required_keys:
            if key not in detector_config:
                raise ValueError(f"Missing required detector config key: {key}")
        
        # Check model file exists
        model_path = detector_config['model_path']
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.async_processor.get_processing_stats()
    
    def update_config(self, **kwargs):
        """Update pipeline configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(**kwargs)
        self.async_processor.config = self.config


class AsyncVideoPipelineResult:
    """Represents an asynchronous video processing operation.
    
    This class provides methods to monitor progress and retrieve
    results of an ongoing video processing operation.
    """
    
    def __init__(
        self,
        pipeline: VideoPipeline,
        video_path: str,
        detector_config: Dict[str, Any],
        output_path: Optional[str],
        visualization_config: Optional[VisualizationConfig]
    ):
        """Initialize async result handler.
        
        Args:
            pipeline: VideoPipeline instance
            video_path: Path to input video
            detector_config: Detector configuration
            output_path: Output path (optional)
            visualization_config: Visualization settings (optional)
        """
        self.pipeline = pipeline
        self.video_path = video_path
        self.detector_config = detector_config
        self.output_path = output_path
        self.visualization_config = visualization_config
        
        self._result = None
        self._error = None
        self._progress = 0.0
        self._completed = False
        
        # Start processing in background
        import threading
        self._thread = threading.Thread(target=self._run_processing)
        self._thread.daemon = True
        self._thread.start()
    
    def _run_processing(self):
        """Run processing in background thread."""
        try:
            def progress_callback(progress: float):
                self._progress = progress
            
            self._result = self.pipeline.process_video(
                video_path=self.video_path,
                detector_config=self.detector_config,
                output_path=self.output_path,
                visualization_config=self.visualization_config,
                progress_callback=progress_callback
            )
            
            self._completed = True
            
        except Exception as e:
            self._error = e
            self._completed = True
    
    def get_progress(self) -> float:
        """Get current processing progress.
        
        Returns:
            Progress as float between 0.0 and 1.0
        """
        return self._progress
    
    def is_completed(self) -> bool:
        """Check if processing is completed.
        
        Returns:
            True if processing is finished (success or error)
        """
        return self._completed
    
    def get_result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get processing result (blocks until completion).
        
        Args:
            timeout: Maximum time to wait in seconds (optional)
            
        Returns:
            Processing result dictionary
            
        Raises:
            RuntimeError: If processing failed
            TimeoutError: If timeout exceeded
        """
        if timeout:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("Processing timeout exceeded")
        else:
            self._thread.join()
        
        if self._error:
            raise RuntimeError(f"Processing failed: {self._error}")
        
        return self._result
    
    def cancel(self):
        """Cancel processing operation."""
        # This would require more complex implementation with proper cancellation
        logger.warning("Cancellation not fully implemented - processing will continue") 