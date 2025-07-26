"""
Unified tennis analysis pipeline with standardized task management.
Refactored integration system supporting all model types with flexible configuration.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from .task_registry import TaskRegistry
from .pipeline_manager import PipelineManager
from .video_processor import VideoProcessor
from .result_manager import ResultManager

log = logging.getLogger(__name__)


class UnifiedPipeline:
    """
    Unified tennis analysis pipeline with modular task architecture.
    Supports all model types (ball, court, player, pose, event) with flexible configuration.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = self._initialize_device()
        
        # Core components
        self.task_registry = TaskRegistry()
        self.pipeline_manager = PipelineManager(cfg, self.device)
        self.video_processor = VideoProcessor(cfg)
        self.result_manager = ResultManager(cfg)
        
        # Performance tracking
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'start_time': None,
            'end_time': None,
            'task_times': {},
            'errors': []
        }
        
        # Initialize pipeline
        self._initialize_tasks()
        self._validate_configuration()
        
    def _initialize_device(self) -> torch.device:
        """Initialize computing device."""
        if self.cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.cfg.device)
        
        log.info(f"Using device: {device}")
        return device
    
    def _initialize_tasks(self) -> None:
        """Initialize all configured tasks."""
        log.info("Initializing pipeline tasks...")
        
        enabled_tasks = []
        for task_name, task_config in self.cfg.tasks.items():
            if task_config.get('enabled', False):
                try:
                    task = self.task_registry.create_task(task_name, task_config, self.device)
                    self.pipeline_manager.register_task(task)
                    enabled_tasks.append(task_name)
                    log.info(f"✓ {task_name} task initialized")
                    
                except Exception as e:
                    error_msg = f"Failed to initialize {task_name} task: {e}"
                    log.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    
                    if not self.cfg.error_handling.get('continue_on_task_failure', True):
                        raise
        
        if not enabled_tasks:
            raise ValueError("No tasks were successfully initialized")
            
        log.info(f"Successfully initialized {len(enabled_tasks)} tasks: {', '.join(enabled_tasks)}")
    
    def _validate_configuration(self) -> None:
        """Validate pipeline configuration."""
        # Check required I/O paths
        if not self.cfg.io.input_video:
            raise ValueError("Input video path is required")
        
        if not os.path.exists(self.cfg.io.input_video):
            raise FileNotFoundError(f"Input video not found: {self.cfg.io.input_video}")
        
        # Validate task dependencies
        self.pipeline_manager.validate_task_dependencies(self.cfg.get('dependencies', {}))
        
        # Create output directory
        if hasattr(self.cfg.io, 'output_video') and self.cfg.io.output_video:
            output_dir = Path(self.cfg.io.output_video).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("Configuration validation completed")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dictionary containing execution statistics and results
        """
        log.info("Starting unified tennis analysis pipeline...")
        self.stats['start_time'] = time.time()
        
        try:
            # Initialize video processing
            self.video_processor.initialize(self.cfg.io.input_video)
            self.stats['total_frames'] = self.video_processor.total_frames
            
            log.info(f"Processing {self.stats['total_frames']} frames...")
            log.info(f"Video: {self.video_processor.width}x{self.video_processor.height} "
                    f"@ {self.video_processor.fps:.2f} fps")
            
            # Process video
            if self.cfg.threading.get('mode', 'single') == 'multi':
                results = self._run_multithreaded()
            else:
                results = self._run_single_threaded()
            
            # Save results
            if hasattr(self.cfg.io, 'output_csv') and self.cfg.io.output_csv:
                self.result_manager.save_results(results, self.cfg.io.output_csv)
            
            self.stats['end_time'] = time.time()
            self._generate_report()
            
            return {
                'success': True,
                'stats': self.stats,
                'results': results if len(results) < 1000 else f"{len(results)} frames processed"
            }
            
        except Exception as e:
            self.stats['end_time'] = time.time()
            error_msg = f"Pipeline execution failed: {e}"
            log.error(error_msg, exc_info=True)
            self.stats['errors'].append(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'stats': self.stats
            }
        
        finally:
            self._cleanup()
    
    def _run_single_threaded(self) -> List[Dict[str, Any]]:
        """Run pipeline in single-threaded mode."""
        log.info("Running pipeline in single-threaded mode")
        
        all_results = []
        batch_size = self.cfg.get('batch_size', 1)
        
        with tqdm(total=self.stats['total_frames'], desc="Processing frames") as pbar:
            frame_batch = []
            frame_indices = []
            
            for frame_idx, frame in enumerate(self.video_processor.read_frames()):
                frame_batch.append(frame)
                frame_indices.append(frame_idx)
                
                # Process batch when full or at end
                if len(frame_batch) >= batch_size or frame_idx == self.stats['total_frames'] - 1:
                    batch_results = self._process_frame_batch(frame_batch, frame_indices)
                    all_results.extend(batch_results)
                    
                    # Write output frames if configured
                    if self.cfg.visualization.get('enabled', True):
                        self._write_output_frames(frame_batch, batch_results, frame_indices)
                    
                    self.stats['processed_frames'] += len(frame_batch)
                    pbar.update(len(frame_batch))
                    
                    frame_batch.clear()
                    frame_indices.clear()
        
        return all_results
    
    def _run_multithreaded(self) -> List[Dict[str, Any]]:
        """Run pipeline in multi-threaded mode."""
        log.info("Running pipeline in multi-threaded mode")
        # TODO: Implement multi-threaded execution
        log.warning("Multi-threaded mode not yet implemented, falling back to single-threaded")
        return self._run_single_threaded()
    
    def _process_frame_batch(self, frames: List[np.ndarray], frame_indices: List[int]) -> List[Dict[str, Any]]:
        """Process a batch of frames through all enabled tasks."""
        batch_results = []
        
        for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            frame_result = {'frame_idx': frame_idx, 'timestamp': frame_idx / self.video_processor.fps}
            
            # Process frame through each task
            for task_name, task in self.pipeline_manager.get_tasks().items():
                task_start_time = time.time()
                
                try:
                    # Execute task
                    task_result = task.process_frame(frame, frame_result)
                    frame_result[task_name] = task_result
                    
                    # Track timing
                    task_time = time.time() - task_start_time
                    if task_name not in self.stats['task_times']:
                        self.stats['task_times'][task_name] = []
                    self.stats['task_times'][task_name].append(task_time)
                    
                except Exception as e:
                    error_msg = f"Task {task_name} failed on frame {frame_idx}: {e}"
                    log.warning(error_msg)
                    self.stats['errors'].append(error_msg)
                    frame_result[task_name] = {'error': str(e)}
                    
                    if not self.cfg.error_handling.get('continue_on_task_failure', True):
                        raise
            
            batch_results.append(frame_result)
        
        return batch_results
    
    def _write_output_frames(self, frames: List[np.ndarray], results: List[Dict[str, Any]], frame_indices: List[int]) -> None:
        """Write visualized output frames to video."""
        if not hasattr(self.cfg.io, 'output_video') or not self.cfg.io.output_video:
            return
        
        # Initialize video writer if not already done
        if not hasattr(self, '_video_writer'):
            fourcc = cv2.VideoWriter_fourcc(*self.cfg.io.get('codec', 'mp4v'))
            self._video_writer = cv2.VideoWriter(
                self.cfg.io.output_video,
                fourcc,
                self.cfg.visualization.get('output_fps', self.video_processor.fps),
                (self.video_processor.width, self.video_processor.height)
            )
        
        # Visualize and write each frame
        for frame, result in zip(frames, results):
            vis_frame = self._visualize_frame(frame, result)
            self._video_writer.write(vis_frame)
    
    def _visualize_frame(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Apply visualizations from all tasks to a frame."""
        vis_frame = frame.copy()
        
        # Apply task-specific visualizations
        for task_name, task in self.pipeline_manager.get_tasks().items():
            if task_name in result and 'error' not in result[task_name]:
                try:
                    vis_config = self.cfg.tasks[task_name].get('visualization', {})
                    if vis_config.get('enabled', True):
                        vis_frame = task.visualize(vis_frame, result[task_name], vis_config)
                except Exception as e:
                    log.warning(f"Visualization failed for task {task_name}: {e}")
        
        # Add frame info if enabled
        if self.cfg.visualization.get('show_frame_info', False):
            self._add_frame_info(vis_frame, result)
        
        return vis_frame
    
    def _add_frame_info(self, frame: np.ndarray, result: Dict[str, Any]) -> None:
        """Add frame information overlay."""
        info_text = f"Frame: {result['frame_idx']}, Time: {result['timestamp']:.2f}s"
        
        pos = self.cfg.visualization.get('info_position', [10, 30])
        font_scale = self.cfg.visualization.get('font_scale', 0.6)
        thickness = self.cfg.visualization.get('font_thickness', 1)
        
        cv2.putText(frame, info_text, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness)
    
    def _generate_report(self) -> None:
        """Generate and log execution report."""
        total_time = self.stats['end_time'] - self.stats['start_time']
        fps = self.stats['processed_frames'] / total_time if total_time > 0 else 0
        
        log.info("=" * 60)
        log.info("UNIFIED PIPELINE EXECUTION REPORT")
        log.info("=" * 60)
        log.info(f"Total frames: {self.stats['total_frames']}")
        log.info(f"Processed frames: {self.stats['processed_frames']}")
        log.info(f"Execution time: {total_time:.2f} seconds")
        log.info(f"Processing FPS: {fps:.2f}")
        log.info(f"Input video: {self.cfg.io.input_video}")
        
        if hasattr(self.cfg.io, 'output_video') and self.cfg.io.output_video:
            log.info(f"Output video: {self.cfg.io.output_video}")
        
        # Task-specific timing statistics
        if self.stats['task_times']:
            log.info("\nTask Performance:")
            for task_name, times in self.stats['task_times'].items():
                avg_time = np.mean(times)
                total_task_time = np.sum(times)
                task_fps = len(times) / total_task_time if total_task_time > 0 else 0
                log.info(f"  {task_name}: {avg_time:.4f}s avg, {task_fps:.1f} fps")
        
        # Error summary
        if self.stats['errors']:
            log.warning(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][-5:]:  # Show last 5 errors
                log.warning(f"  {error}")
        
        log.info("=" * 60)
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, '_video_writer'):
            self._video_writer.release()
        
        self.video_processor.cleanup()
        self.pipeline_manager.cleanup()


@hydra.main(config_path="../../../configs/infer/integrate", config_name="pipeline_unified", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for the unified pipeline."""
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.get('level', 'INFO').upper()),
        format=cfg.logging.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    )
    
    log.info("Unified Tennis Analysis Pipeline")
    log.info(f"Configuration: {cfg}")
    
    try:
        pipeline = UnifiedPipeline(cfg)
        result = pipeline.run()
        
        if result['success']:
            log.info("Pipeline execution completed successfully")
        else:
            log.error(f"Pipeline execution failed: {result['error']}")
            exit(1)
            
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        
    except Exception as e:
        log.error(f"Pipeline execution failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()