"""
Flexible tennis analysis pipeline with modular task architecture.
"""
import hydra
from omegaconf import DictConfig
import cv2
import torch
import logging
import threading
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os

from .task_manager import TaskManager
from .data_flow import ThreadSafeDataFlow, DataPacket, DataStage
from .video_io import VideoReader, VideoWriter

log = logging.getLogger(__name__)


class FlexiblePipeline:
    """
    Flexible, modular tennis analysis pipeline with configurable tasks and threading.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = self._initialize_device()
        
        # Core components
        self.task_manager = TaskManager()
        self.data_flow = ThreadSafeDataFlow(
            buffer_size=cfg.get('buffer_size', 100),
            queue_size=cfg.threading.get('queue_size', 50)
        )
        
        # Video I/O
        self.video_reader: Optional[VideoReader] = None
        self.video_writer: Optional[VideoWriter] = None
        
        # Threading
        self.threads: List[threading.Thread] = []
        self.is_running = threading.Event()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Initialize pipeline
        self._initialize_tasks()
        self._initialize_video_io()
        
    def _initialize_device(self) -> torch.device:
        """Initialize PyTorch device."""
        if self.cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.cfg.device)
        
        log.info(f"Using device: {device}")
        return device
    
    def _initialize_tasks(self) -> None:
        """Initialize all configured tasks."""
        log.info("Initializing pipeline tasks...\n")       
        if 'tasks' not in self.cfg or not self.cfg.tasks:
            log.warning("No tasks configured in pipeline")
            return
        
        for task_config in self.cfg.tasks:
            # Convert OmegaConf to dict for easier manipulation
            from omegaconf import OmegaConf
            task_config_dict = OmegaConf.to_container(task_config, resolve=True)
            
            if task_config_dict.get('enabled', True):
                try:
                    self.task_manager.register_task_from_config(task_config_dict, self.device)
                except Exception as e:
                    log.error(f"Failed to initialize task '{task_config_dict.get('name', 'unknown')}': {e}")
                    if task_config_dict.get('critical', True):
                        raise
        
        # Validate pipeline
        if not self.task_manager.validate_dependencies():
            raise RuntimeError("Task dependency validation failed")
        
        log.info(f"Initialized {len(self.task_manager.get_enabled_tasks())} tasks")
        log.info("\\n" + self.task_manager.visualize_dependency_graph())
    
    def _initialize_video_io(self) -> None:
        """Initialize video input/output."""
        # Initialize video reader
        self.video_reader = VideoReader(self.cfg.io.input_video)
        
        # Initialize video writer if output path is specified
        if hasattr(self.cfg.io, 'output_video') and self.cfg.io.output_video:
            self.video_writer = VideoWriter(
                self.cfg.io.output_video,
                fps=self.video_reader.fps,
                frame_size=(self.video_reader.width, self.video_reader.height),
                codec=self.cfg.io.get('codec', 'mp4v')
            )
        
        self.stats['total_frames'] = self.video_reader.frame_count
        log.info(f"Video: {self.video_reader.width}x{self.video_reader.height}, "
                f"{self.video_reader.frame_count} frames, {self.video_reader.fps:.2f} fps")
    
    def run(self) -> None:
        """Run the complete pipeline."""
        log.info("Starting flexible tennis analysis pipeline...")
        
        self.stats['start_time'] = time.time()
        self.is_running.set()
        
        # Determine threading model based on configuration
        if self.cfg.threading.get('mode', 'single') == 'multi':
            self._run_multithreaded()
        else:
            self._run_single_threaded()
        
        self.stats['end_time'] = time.time()
        self._finalize()
    
    def _run_single_threaded(self) -> None:
        """Run pipeline in single-threaded mode."""
        log.info("Running pipeline in single-threaded mode")
        
        batch_size = self.cfg.get('batch_size', 1)
        pbar = tqdm(total=self.stats['total_frames'], desc="Processing frames")
        
        frames_batch = []
        frame_indices_batch = []
        
        for frame_idx, frame in enumerate(self.video_reader):
            if not self.is_running.is_set():
                break
                
            frames_batch.append(frame)
            frame_indices_batch.append(frame_idx)
            
            # Process batch when full or at end of video
            if len(frames_batch) >= batch_size or frame_idx == self.stats['total_frames'] - 1:
                # Execute pipeline on batch
                results = self.task_manager.execute_pipeline(frames_batch)
                
                # Visualize and write output
                if self.video_writer and self.cfg.visualization.get('enabled', True):
                    for i, frame in enumerate(frames_batch):
                        vis_frame = self._visualize_batch_frame(frame, results, i)
                        self.video_writer.write(vis_frame)
                
                # Save results if configured
                if hasattr(self.cfg.io, 'output_csv') and self.cfg.io.output_csv:
                    self._save_batch_results(frame_indices_batch, results)
                
                self.stats['processed_frames'] += len(frames_batch)
                pbar.update(len(frames_batch))
                
                frames_batch.clear()
                frame_indices_batch.clear()
        
        pbar.close()
    
    def _run_multithreaded(self) -> None:
        """Run pipeline in multi-threaded mode."""
        log.info("Running pipeline in multi-threaded mode")
        
        # Create processing queues for each stage
        self.data_flow.create_processing_queue('preprocessing')
        self.data_flow.create_processing_queue('inference')
        self.data_flow.create_processing_queue('postprocessing')
        
        # Start worker threads
        self.threads = [
            threading.Thread(target=self._worker_video_input, name="VideoInput"),
            threading.Thread(target=self._worker_preprocessing, name="Preprocessing"),
            threading.Thread(target=self._worker_inference, name="Inference"),
            threading.Thread(target=self._worker_postprocessing, name="Postprocessing"),
            threading.Thread(target=self._worker_video_output, name="VideoOutput")
        ]
        
        log.info(f"Starting {len(self.threads)} worker threads...")
        for thread in self.threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()
        
        log.info("All worker threads completed")
    
    def _worker_video_input(self) -> None:
        """Worker thread for video input and batching."""
        log.info("Video input worker started")
        
        batch_size = self.cfg.get('batch_size', 1)
        frames_batch = []
        frame_indices_batch = []
        
        try:
            for frame_idx, frame in enumerate(self.video_reader):
                if not self.is_running.is_set():
                    break
                
                frames_batch.append(frame)
                frame_indices_batch.append(frame_idx)
                
                if len(frames_batch) >= batch_size or frame_idx == self.stats['total_frames'] - 1:
                    packet = self.data_flow.create_packet(frame_indices_batch, frames_batch)
                    
                    if not self.data_flow.put_processing('preprocessing', packet, timeout=5.0):
                        log.warning("Failed to queue packet for preprocessing")
                    
                    frames_batch.clear()
                    frame_indices_batch.clear()
        
        except Exception as e:
            log.error(f"Video input worker error: {e}")
        
        finally:
            # Signal end of input
            self.data_flow.put_processing('preprocessing', None)
            log.info("Video input worker finished")
    
    def _worker_preprocessing(self) -> None:
        """Worker thread for preprocessing tasks."""
        log.info("Preprocessing worker started")
        
        try:
            while self.is_running.is_set():
                packet = self.data_flow.get_processing('preprocessing', timeout=1.0)
                
                if packet is None:  # Shutdown signal or timeout
                    if self.data_flow.is_shutdown_requested():
                        break
                    continue
                
                # Execute preprocessing for all tasks
                for task in self.task_manager.get_enabled_tasks():
                    start_time = time.time()
                    try:
                        preprocessed_data, meta = task.preprocess(packet.frames)
                        packet.add_task_data(task.name, DataStage.PREPROCESSED, preprocessed_data, meta)
                        
                        processing_time = time.time() - start_time
                        self.data_flow.record_processing_time(task.name, 'preprocess', processing_time)
                        
                    except Exception as e:
                        log.error(f"Preprocessing failed for task '{task.name}': {e}")
                
                if not self.data_flow.put_processing('inference', packet, timeout=5.0):
                    log.warning("Failed to queue packet for inference")
        
        except Exception as e:
            log.error(f"Preprocessing worker error: {e}")
        
        finally:
            self.data_flow.put_processing('inference', None)
            log.info("Preprocessing worker finished")
    
    def _worker_inference(self) -> None:
        """Worker thread for inference tasks."""
        log.info("Inference worker started")
        
        try:
            while self.is_running.is_set():
                packet = self.data_flow.get_processing('inference', timeout=1.0)
                
                if packet is None:
                    if self.data_flow.is_shutdown_requested():
                        break
                    continue
                
                # Execute inference for all tasks
                for task in self.task_manager.get_enabled_tasks():
                    if not packet.has_task_data(task.name, DataStage.PREPROCESSED):
                        continue
                    
                    start_time = time.time()
                    try:
                        preprocessed_data = packet.get_task_data(task.name, DataStage.PREPROCESSED)
                        meta = packet.get_task_metadata(task.name, DataStage.PREPROCESSED)
                        
                        raw_outputs = task.inference(preprocessed_data, meta)
                        packet.add_task_data(task.name, DataStage.RAW_OUTPUT, raw_outputs)
                        
                        processing_time = time.time() - start_time
                        self.data_flow.record_processing_time(task.name, 'inference', processing_time)
                        
                    except Exception as e:
                        log.error(f"Inference failed for task '{task.name}': {e}")
                
                if not self.data_flow.put_processing('postprocessing', packet, timeout=5.0):
                    log.warning("Failed to queue packet for postprocessing")
        
        except Exception as e:
            log.error(f"Inference worker error: {e}")
        
        finally:
            self.data_flow.put_processing('postprocessing', None)
            log.info("Inference worker finished")
    
    def _worker_postprocessing(self) -> None:
        """Worker thread for postprocessing tasks."""
        log.info("Postprocessing worker started")
        
        try:
            while self.is_running.is_set():
                packet = self.data_flow.get_processing('postprocessing', timeout=1.0)
                
                if packet is None:
                    if self.data_flow.is_shutdown_requested():
                        break
                    continue
                
                # Execute postprocessing for all tasks
                for task in self.task_manager.get_enabled_tasks():
                    if not packet.has_task_data(task.name, DataStage.RAW_OUTPUT):
                        continue
                    
                    start_time = time.time()
                    try:
                        raw_outputs = packet.get_task_data(task.name, DataStage.RAW_OUTPUT)
                        meta = packet.get_task_metadata(task.name, DataStage.PREPROCESSED)
                        
                        results = task.postprocess(raw_outputs, meta)
                        packet.add_task_data(task.name, DataStage.POSTPROCESSED, results)
                        
                        processing_time = time.time() - start_time
                        self.data_flow.record_processing_time(task.name, 'postprocess', processing_time)
                        
                    except Exception as e:
                        log.error(f"Postprocessing failed for task '{task.name}': {e}")
                
                if not self.data_flow.put_output(packet, timeout=5.0):
                    log.warning("Failed to queue packet for output")
        
        except Exception as e:
            log.error(f"Postprocessing worker error: {e}")
        
        finally:
            self.data_flow.put_output(None)
            log.info("Postprocessing worker finished")
    
    def _worker_video_output(self) -> None:
        """Worker thread for video output and result saving."""
        log.info("Video output worker started")
        
        pbar = tqdm(total=self.stats['total_frames'], desc="Writing output")
        
        try:
            while self.is_running.is_set():
                packet = self.data_flow.get_output(timeout=1.0)
                
                if packet is None:
                    if self.data_flow.is_shutdown_requested():
                        break
                    continue
                
                # Visualize and write frames
                if self.video_writer and self.cfg.visualization.get('enabled', True):
                    for i, frame in enumerate(packet.frames):
                        vis_frame = self._visualize_packet_frame(packet, frame, i)
                        self.video_writer.write(vis_frame)
                
                # Save results
                if hasattr(self.cfg.io, 'output_csv') and self.cfg.io.output_csv:
                    self._save_packet_results(packet)
                
                self.stats['processed_frames'] += len(packet.frames)
                pbar.update(len(packet.frames))
        
        except Exception as e:
            log.error(f"Video output worker error: {e}")
        
        finally:
            pbar.close()
            log.info("Video output worker finished")
    
    def _visualize_batch_frame(self, frame: Any, results: Dict[str, Any], frame_idx: int) -> Any:
        """Visualize results on a single frame from batch results."""
        output_frame = frame.copy()
        
        for task in self.task_manager.get_enabled_tasks():
            if task.name in results:
                task_results = results[task.name]
                
                try:
                    vis_config = self.cfg.visualization.get(task.name.split('_')[0], {})
                    
                    # Extract frame-specific results from batch
                    if 'batch_results' in task_results and len(task_results['batch_results']) > frame_idx:
                        frame_results = {'batch_results': [task_results['batch_results'][frame_idx]]}
                        output_frame = task.visualize(output_frame, frame_results, vis_config)
                except Exception as e:
                    log.warning(f"Visualization failed for task '{task.name}': {e}")
        
        return output_frame
    
    def _visualize_packet_frame(self, packet: DataPacket, frame: Any, frame_idx: int) -> Any:
        """Visualize results on a single frame from a packet."""
        output_frame = frame.copy()
        
        for task in self.task_manager.get_enabled_tasks():
            if packet.has_task_data(task.name, DataStage.POSTPROCESSED):
                results = packet.get_task_data(task.name, DataStage.POSTPROCESSED)
                
                try:
                    vis_config = self.cfg.visualization.get(task.name, {})
                    output_frame = task.visualize(output_frame, results, vis_config)
                except Exception as e:
                    log.warning(f"Visualization failed for task '{task.name}': {e}")
        
        return output_frame
    
    def _save_batch_results(self, frame_indices: List[int], results: Dict[str, Any]) -> None:
        """Save batch results to storage."""
        # This is a placeholder for result saving
        # Results would be accumulated and saved at the end
        pass
    
    def _save_packet_results(self, packet: DataPacket) -> None:
        """Save packet results to CSV or other format."""
        # Implementation depends on specific result format requirements
        # This is a placeholder for the CSV saving logic
        pass
    
    def _finalize(self) -> None:
        """Finalize pipeline execution."""
        # Close video I/O
        if self.video_reader:
            self.video_reader.close()
        if self.video_writer:
            self.video_writer.close()
        
        # Signal shutdown to data flow
        self.data_flow.signal_shutdown()
        
        # Generate final report
        self._generate_report()
    
    def _generate_report(self) -> None:
        """Generate and log final execution report."""
        total_time = self.stats['end_time'] - self.stats['start_time']
        fps = self.stats['processed_frames'] / total_time if total_time > 0 else 0
        
        log.info("=" * 60)
        log.info("PIPELINE EXECUTION REPORT")
        log.info("=" * 60)
        log.info(f"Total frames: {self.stats['total_frames']}")
        log.info(f"Processed frames: {self.stats['processed_frames']}")
        log.info(f"Execution time: {total_time:.2f} seconds")
        log.info(f"Processing FPS: {fps:.2f}")
        
        # Task execution statistics
        task_info = self.task_manager.get_task_info()
        log.info(f"Tasks executed: {task_info['enabled_tasks']}/{task_info['total_tasks']}")
        
        # Data flow statistics
        data_stats = self.data_flow.get_stats()
        log.info(f"Data packets processed: {data_stats['packets_processed']}")
        
        if self.cfg.threading.get('mode') == 'multi':
            queue_stats = self.data_flow.get_queue_stats()
            log.info("Queue utilization:")
            for name, util in queue_stats['processing_utilization'].items():
                log.info(f"  {name}: {util:.1f}%")
        
        log.info("=" * 60)


@hydra.main(config_path="../../../configs/infer/integrate", config_name="flexible_pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for the flexible pipeline."""
    
    # Validate required configuration
    if not hasattr(cfg.io, 'input_video') or not cfg.io.input_video:
        raise ValueError("Input video path is required. Please set io.input_video.")
    
    if not os.path.exists(cfg.io.input_video):
        raise FileNotFoundError(f"Input video not found: {cfg.io.input_video}")
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.get('level', 'INFO').upper()),
        format=cfg.logging.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    )
    
    log.info("Flexible Tennis Analysis Pipeline")
    log.info(f"Input video: {cfg.io.input_video}")
    
    if hasattr(cfg.io, 'output_video') and cfg.io.output_video:
        log.info(f"Output video: {cfg.io.output_video}")
    
    try:
        pipeline = FlexiblePipeline(cfg)
        pipeline.run()
        
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        
    except Exception as e:
        log.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()