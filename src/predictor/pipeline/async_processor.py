"""
Async video processor for parallel pipeline execution.

This module provides the core async processing engine that manages
multiple threads for efficient video processing with minimal GPU waiting time.
"""

import threading
import queue
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from .config import PipelineConfig
from ..base.detector import BaseBallDetector
from ..ball.factory import create_ball_detector
from ..visualization import VideoOverlay, VisualizationConfig

logger = logging.getLogger(__name__)


class AsyncVideoProcessor:
    """Asynchronous video processor with parallel threading.
    
    This class manages the complete parallel processing pipeline with
    separate threads for frame reading, preprocessing, inference, 
    postprocessing, and visualization to maximize GPU utilization.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize async video processor.
        
        Args:
            config: Pipeline configuration (uses default if None)
        """
        self.config = config or PipelineConfig()
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.stop_event = threading.Event()
        
        # Processing queues (thread-safe)
        self.frame_queue = queue.Queue(maxsize=self.config.frame_buffer_size)
        self.tensor_queue = queue.Queue(maxsize=self.config.tensor_buffer_size)
        self.result_queue = queue.Queue(maxsize=self.config.result_buffer_size)
        self.render_queue = queue.Queue(maxsize=self.config.render_buffer_size)
        
        # State management
        self.detector: Optional[BaseBallDetector] = None
        self.video_overlay: Optional[VideoOverlay] = None
        self.error_count = 0
        self.frame_count = 0
        self.processed_frames = 0
        
        # Performance monitoring
        self.start_time = None
        self.processing_stats = {
            'frames_read': 0,
            'frames_preprocessed': 0,
            'frames_inferred': 0,
            'frames_postprocessed': 0,
            'frames_rendered': 0,
            'total_inference_time': 0.0,
            'total_processing_time': 0.0
        }
        
        logger.info("AsyncVideoProcessor initialized")
    
    def process_video(
        self,
        video_path: str,
        detector_config: Dict[str, Any],
        output_path: Optional[str] = None,
        visualization_config: Optional[VisualizationConfig] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Process video with parallel pipeline.
        
        Args:
            video_path: Path to input video file
            detector_config: Detector configuration
            output_path: Path for output video (optional)
            visualization_config: Visualization settings (optional)
            progress_callback: Progress callback function
            
        Returns:
            Dictionary containing processing results and statistics
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Initialize components
        self._initialize_detector(detector_config)
        if output_path and self.config.enable_visualization:
            self._initialize_visualization(visualization_config)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.frame_count = total_frames
            self.start_time = time.time()
            
            logger.info(f"Processing video: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            # Initialize video writer if needed
            video_writer = None
            if output_path and self.config.enable_visualization:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Start parallel processing threads
            futures = self._start_processing_threads(
                cap, video_writer, total_frames, progress_callback
            )
            
            # Wait for completion
            results = self._wait_for_completion(futures)
            
            # Collect final statistics
            processing_time = time.time() - self.start_time
            self.processing_stats['total_processing_time'] = processing_time
            
            logger.info(f"Video processing completed in {processing_time:.2f}s")
            logger.info(f"Average FPS: {total_frames/processing_time:.2f}")
            
            return {
                'output_path': output_path,
                'processing_time': processing_time,
                'frames_processed': self.processed_frames,
                'average_fps': total_frames/processing_time,
                'stats': self.processing_stats,
                'detections': results.get('detections', {}),
                'error_count': self.error_count
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            self._cleanup()
    
    def _initialize_detector(self, detector_config: Dict[str, Any]):
        """Initialize ball detector."""
        try:
            self.detector = create_ball_detector(**detector_config)
            logger.info(f"Detector initialized: {self.detector.model_info}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize detector: {e}")
    
    def _initialize_visualization(self, visualization_config: Optional[VisualizationConfig]):
        """Initialize visualization components."""
        try:
            vis_config = visualization_config or VisualizationConfig()
            self.video_overlay = VideoOverlay(vis_config)
            logger.info("Visualization initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize visualization: {e}")
    
    def _start_processing_threads(
        self,
        cap: cv2.VideoCapture,
        video_writer: Optional[cv2.VideoWriter],
        total_frames: int,
        progress_callback: Optional[Callable[[float], None]]
    ) -> Dict[str, Future]:
        """Start all processing threads."""
        futures = {}
        
        # Frame reader thread
        futures['reader'] = self.executor.submit(
            self._frame_reader_worker, cap, total_frames
        )
        
        # Preprocessing threads
        for i in range(self.config.num_preprocessing_threads):
            futures[f'preprocessor_{i}'] = self.executor.submit(
                self._preprocessing_worker, i
            )
        
        # GPU inference thread
        if self.config.enable_gpu_threading:
            futures['inference'] = self.executor.submit(
                self._inference_worker
            )
        
        # Postprocessing threads
        for i in range(self.config.num_postprocessing_threads):
            futures[f'postprocessor_{i}'] = self.executor.submit(
                self._postprocessing_worker, i
            )
        
        # Visualization thread
        if video_writer and self.config.enable_visualization:
            futures['visualization'] = self.executor.submit(
                self._visualization_worker, video_writer, total_frames, progress_callback
            )
        
        # Monitoring thread
        if self.config.log_queue_sizes:
            futures['monitor'] = self.executor.submit(
                self._monitoring_worker
            )
        
        logger.info(f"Started {len(futures)} processing threads")
        return futures
    
    def _frame_reader_worker(self, cap: cv2.VideoCapture, total_frames: int):
        """Worker thread for reading video frames."""
        frame_idx = 0
        
        try:
            while frame_idx < total_frames and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if (self.config.enable_frame_skipping and 
                    frame_idx % self.config.frame_skip_interval != 0):
                    frame_idx += 1
                    continue
                
                frame_data = (frame, {'frame_id': f'frame_{frame_idx:06d}', 'frame_number': frame_idx})
                
                try:
                    self.frame_queue.put(frame_data, timeout=self.config.queue_timeout)
                    self.processing_stats['frames_read'] += 1
                    frame_idx += 1
                except queue.Full:
                    logger.warning("Frame queue full, dropping frame")
                    continue
            
            # Signal end of frames
            for _ in range(self.config.num_preprocessing_threads):
                self.frame_queue.put(None, timeout=self.config.queue_timeout)
                
        except Exception as e:
            logger.error(f"Frame reader error: {e}")
            self._handle_thread_error()
    
    def _preprocessing_worker(self, worker_id: int):
        """Worker thread for frame preprocessing."""
        buffer = []
        frames_required = self.detector.frames_required
        
        try:
            while not self.stop_event.is_set():
                try:
                    frame_data = self.frame_queue.get(timeout=self.config.queue_timeout)
                    if frame_data is None:  # End signal
                        break
                    
                    buffer.append(frame_data)
                    
                    # Process when we have enough frames
                    if len(buffer) >= frames_required:
                        # Preprocess batch
                        processed_batch = self.detector.preprocess(buffer[-frames_required:])
                        
                        for processed_item in processed_batch:
                            self.tensor_queue.put(processed_item, timeout=self.config.queue_timeout)
                            self.processing_stats['frames_preprocessed'] += 1
                        
                        # Keep overlap for temporal models
                        if len(buffer) > frames_required:
                            buffer = buffer[-(frames_required-1):]
                    
                except queue.Empty:
                    continue
                except queue.Full:
                    logger.warning(f"Tensor queue full in preprocessor {worker_id}")
                    continue
                    
        except Exception as e:
            logger.error(f"Preprocessing worker {worker_id} error: {e}")
            self._handle_thread_error()
        
        # Signal end for inference
        self.tensor_queue.put(None, timeout=self.config.queue_timeout)
    
    def _inference_worker(self):
        """Worker thread for GPU inference."""
        batch_buffer = []
        
        try:
            while not self.stop_event.is_set():
                try:
                    tensor_data = self.tensor_queue.get(timeout=self.config.queue_timeout)
                    if tensor_data is None:  # End signal
                        break
                    
                    batch_buffer.append(tensor_data)
                    
                    # Process batch when full or timeout
                    if len(batch_buffer) >= self.config.gpu_batch_size:
                        self._process_inference_batch(batch_buffer)
                        batch_buffer = []
                        
                except queue.Empty:
                    # Process remaining batch on timeout
                    if batch_buffer:
                        self._process_inference_batch(batch_buffer)
                        batch_buffer = []
                    continue
                    
            # Process final batch
            if batch_buffer:
                self._process_inference_batch(batch_buffer)
                
        except Exception as e:
            logger.error(f"Inference worker error: {e}")
            self._handle_thread_error()
        
        # Signal end for postprocessing
        for _ in range(self.config.num_postprocessing_threads):
            self.result_queue.put(None, timeout=self.config.queue_timeout)
    
    def _process_inference_batch(self, batch_buffer: List[Tuple[Any, dict]]):
        """Process a batch of tensors through inference."""
        inference_start = time.time()
        
        try:
            # Perform batch inference
            inference_results = self.detector.infer(batch_buffer)
            
            # Add results to queue
            for result in inference_results:
                self.result_queue.put(result, timeout=self.config.queue_timeout)
                self.processing_stats['frames_inferred'] += 1
            
            inference_time = time.time() - inference_start
            self.processing_stats['total_inference_time'] += inference_time
            
        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else "Unknown error"
            logger.error(f"Batch inference error: {error_msg}")
            logger.error(f"Batch inference traceback: {traceback.format_exc()}")
            self._handle_thread_error()
    
    def _postprocessing_worker(self, worker_id: int):
        """Worker thread for postprocessing inference results."""
        try:
            while not self.stop_event.is_set():
                try:
                    result_data = self.result_queue.get(timeout=self.config.queue_timeout)
                    if result_data is None:  # End signal
                        break
                    
                    # Postprocess individual result
                    detections = self.detector.postprocess([result_data])
                    
                    # Add to render queue
                    render_data = (result_data[1], detections)  # (metadata, detections)
                    self.render_queue.put(render_data, timeout=self.config.queue_timeout)
                    self.processing_stats['frames_postprocessed'] += 1
                    
                except queue.Empty:
                    continue
                except queue.Full:
                    logger.warning(f"Render queue full in postprocessor {worker_id}")
                    continue
                    
        except Exception as e:
            logger.error(f"Postprocessing worker {worker_id} error: {e}")
            self._handle_thread_error()
        
        # Signal end for visualization
        self.render_queue.put(None, timeout=self.config.queue_timeout)
    
    def _visualization_worker(
        self,
        video_writer: cv2.VideoWriter,
        total_frames: int,
        progress_callback: Optional[Callable[[float], None]]
    ):
        """Worker thread for video visualization."""
        frame_cache = {}  # Cache for out-of-order frames
        next_frame_id = 0
        
        try:
            while not self.stop_event.is_set():
                try:
                    render_data = self.render_queue.get(timeout=self.config.queue_timeout)
                    if render_data is None:  # End signal
                        break
                    
                    metadata, detections = render_data
                    
                    # Handle missing frame_number key (extract from frame_id if needed)
                    frame_number = metadata.get('frame_number')
                    if frame_number is None:
                        # Extract from frame_id if frame_number not available
                        frame_id = metadata.get('frame_id', '')
                        if frame_id.startswith('frame_'):
                            try:
                                frame_number = int(frame_id.replace('frame_', '').lstrip('0') or '0')
                            except ValueError:
                                frame_number = 0
                        else:
                            frame_number = 0
                    
                    # Cache frame if out of order
                    frame_cache[frame_number] = (metadata, detections)
                    
                    # Process frames in order
                    while next_frame_id in frame_cache:
                        frame_metadata, frame_detections = frame_cache.pop(next_frame_id)
                        
                        # Render frame (simplified - would need original frame)
                        # This would require additional frame caching or re-reading
                        
                        self.processing_stats['frames_rendered'] += 1
                        self.processed_frames += 1
                        next_frame_id += 1
                        
                        # Progress callback
                        if progress_callback and self.config.enable_progress_callback:
                            if self.processed_frames % self.config.progress_update_interval == 0:
                                progress = self.processed_frames / total_frames
                                progress_callback(progress)
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error(f"Visualization worker error: {e}")
            self._handle_thread_error()
    
    def _monitoring_worker(self):
        """Worker thread for monitoring queue sizes and performance."""
        try:
            while not self.stop_event.is_set():
                if self.config.log_queue_sizes:
                    logger.info(
                        f"Queue sizes - Frame: {self.frame_queue.qsize()}, "
                        f"Tensor: {self.tensor_queue.qsize()}, "
                        f"Result: {self.result_queue.qsize()}, "
                        f"Render: {self.render_queue.qsize()}"
                    )
                
                time.sleep(1.0)  # Monitor every second
                
        except Exception as e:
            logger.error(f"Monitoring worker error: {e}")
    
    def _handle_thread_error(self):
        """Handle thread errors with safety limits."""
        self.error_count += 1
        if self.error_count >= self.config.max_error_count:
            logger.error(f"Too many errors ({self.error_count}), stopping processing")
            self.stop_event.set()
    
    def _wait_for_completion(self, futures: Dict[str, Future]) -> Dict[str, Any]:
        """Wait for all threads to complete and collect results."""
        results = {}
        
        try:
            for name, future in futures.items():
                try:
                    result = future.result(timeout=self.config.thread_timeout)
                    results[name] = result
                except Exception as e:
                    logger.error(f"Thread {name} failed: {e}")
                    results[name] = None
            
        except Exception as e:
            logger.error(f"Error waiting for threads: {e}")
            self.stop_event.set()
        
        return results
    
    def _cleanup(self):
        """Clean up resources and threads."""
        self.stop_event.set()
        
        # Clear queues
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.tensor_queue)
        self._clear_queue(self.result_queue)
        self._clear_queue(self.render_queue)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("AsyncVideoProcessor cleanup completed")
    
    def _clear_queue(self, q: queue.Queue):
        """Clear all items from a queue."""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
    
    def _postprocessing_worker(self, worker_id: int):
        """Worker thread for postprocessing inference results."""
        try:
            while not self.stop_event.is_set():
                try:
                    result_data = self.result_queue.get(timeout=self.config.queue_timeout)
                    if result_data is None:  # End signal
                        break
                    
                    # Postprocess individual result
                    detections = self.detector.postprocess([result_data])
                    
                    # Add to render queue
                    render_data = (result_data[1], detections)  # (metadata, detections)
                    self.render_queue.put(render_data, timeout=self.config.queue_timeout)
                    self.processing_stats['frames_postprocessed'] += 1
                    
                except queue.Empty:
                    continue
                except queue.Full:
                    logger.warning(f"Render queue full in postprocessor {worker_id}")
                    continue
                    
        except Exception as e:
            logger.error(f"Postprocessing worker {worker_id} error: {e}")
            self._handle_thread_error()
        
        # Signal end for visualization
        self.render_queue.put(None, timeout=self.config.queue_timeout)
    
    def _visualization_worker(
        self,
        video_writer: cv2.VideoWriter,
        total_frames: int,
        progress_callback: Optional[Callable[[float], None]]
    ):
        """Worker thread for video visualization."""
        frame_cache = {}  # Cache for out-of-order frames
        next_frame_id = 0
        
        try:
            while not self.stop_event.is_set():
                try:
                    render_data = self.render_queue.get(timeout=self.config.queue_timeout)
                    if render_data is None:  # End signal
                        break
                    
                    metadata, detections = render_data
                    
                    # Handle missing frame_number key (extract from frame_id if needed)
                    frame_number = metadata.get('frame_number')
                    if frame_number is None:
                        # Extract from frame_id if frame_number not available
                        frame_id = metadata.get('frame_id', '')
                        if frame_id.startswith('frame_'):
                            try:
                                frame_number = int(frame_id.replace('frame_', '').lstrip('0') or '0')
                            except ValueError:
                                frame_number = 0
                        else:
                            frame_number = 0
                    
                    # Cache frame if out of order
                    frame_cache[frame_number] = (metadata, detections)
                    
                    # Process frames in order
                    while next_frame_id in frame_cache:
                        frame_metadata, frame_detections = frame_cache.pop(next_frame_id)
                        
                        # Note: In a full implementation, we would need to cache
                        # original frames for visualization here
                        
                        self.processing_stats['frames_rendered'] += 1
                        self.processed_frames += 1
                        next_frame_id += 1
                        
                        # Progress callback
                        if progress_callback and self.config.enable_progress_callback:
                            if self.processed_frames % self.config.progress_update_interval == 0:
                                progress = self.processed_frames / total_frames
                                progress_callback(progress)
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error(f"Visualization worker error: {e}")
            self._handle_thread_error()
    
    def _monitoring_worker(self):
        """Worker thread for monitoring queue sizes and performance."""
        try:
            while not self.stop_event.is_set():
                if self.config.log_queue_sizes:
                    logger.info(
                        f"Queue sizes - Frame: {self.frame_queue.qsize()}, "
                        f"Tensor: {self.tensor_queue.qsize()}, "
                        f"Result: {self.result_queue.qsize()}, "
                        f"Render: {self.render_queue.qsize()}"
                    )
                
                time.sleep(1.0)  # Monitor every second
                
        except Exception as e:
            logger.error(f"Monitoring worker error: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.processing_stats.copy() 