"""
Pipeline configuration management.

This module provides configuration classes for managing parallel processing
parameters and optimization settings for the video processing pipeline.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch


@dataclass
class PipelineConfig:
    """Configuration for parallel video processing pipeline.
    
    This class manages all parameters for the multi-threaded pipeline
    including buffer sizes, thread counts, and optimization settings.
    """
    
    # Buffer Configuration
    frame_buffer_size: int = 30        # Raw frame buffer size
    tensor_buffer_size: int = 15       # Preprocessed tensor buffer size  
    result_buffer_size: int = 20       # Inference result buffer size
    render_buffer_size: int = 10       # Render queue buffer size
    
    # Thread Configuration
    num_preprocessing_threads: int = 2  # Number of preprocessing threads
    num_postprocessing_threads: int = 1 # Number of postprocessing threads
    enable_gpu_threading: bool = True   # Use separate GPU thread
    
    # GPU Configuration
    gpu_batch_size: int = 4            # Batch size for GPU inference
    device: str = "auto"               # Device selection ("cuda", "cpu", "auto")
    max_gpu_memory_fraction: float = 0.8  # Maximum GPU memory usage
    
    # Processing Configuration
    enable_visualization: bool = True   # Enable video overlay generation
    enable_frame_skipping: bool = False # Skip frames for faster processing
    frame_skip_interval: int = 1       # Interval for frame skipping
    
    # Memory Management
    enable_memory_optimization: bool = True  # Use memory-efficient processing
    tensor_cache_size: int = 100       # Cache size for tensor preprocessing
    enable_cpu_offload: bool = False   # Offload tensors to CPU when possible
    
    # Quality Settings
    output_quality: int = 95           # Output video quality (0-100)
    resize_frames: bool = False        # Resize frames for processing
    target_resolution: Optional[Tuple[int, int]] = None  # (width, height)
    
    # Debugging and Monitoring
    enable_profiling: bool = False     # Enable performance profiling
    log_queue_sizes: bool = False      # Log queue sizes for monitoring
    enable_progress_callback: bool = True  # Enable progress reporting
    progress_update_interval: int = 100    # Progress update frequency
    
    # Safety and Timeout
    thread_timeout: float = 30.0       # Thread timeout in seconds
    queue_timeout: float = 5.0         # Queue operation timeout
    max_error_count: int = 10          # Maximum allowed errors before abort
    
    def __post_init__(self):
        """Validate and adjust configuration parameters."""
        self._validate_parameters()
        self._optimize_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        if self.frame_buffer_size < 5:
            raise ValueError("frame_buffer_size must be at least 5")
        
        if self.tensor_buffer_size < 3:
            raise ValueError("tensor_buffer_size must be at least 3")
        
        if self.gpu_batch_size < 1:
            raise ValueError("gpu_batch_size must be at least 1")
        
        if not (0 < self.max_gpu_memory_fraction <= 1.0):
            raise ValueError("max_gpu_memory_fraction must be between 0 and 1")
        
        if self.num_preprocessing_threads < 1:
            raise ValueError("num_preprocessing_threads must be at least 1")
    
    def _optimize_parameters(self):
        """Optimize parameters based on system configuration."""
        # Auto-detect device if needed
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adjust buffer sizes based on GPU availability
        if self.device == "cpu":
            # Reduce buffer sizes for CPU processing
            self.tensor_buffer_size = min(self.tensor_buffer_size, 8)
            self.gpu_batch_size = min(self.gpu_batch_size, 2)
        
        # Optimize threading for CPU count
        if self.num_preprocessing_threads > 4:
            self.num_preprocessing_threads = 4  # Diminishing returns beyond 4
    
    def get_device(self) -> torch.device:
        """Get the PyTorch device object.
        
        Returns:
            PyTorch device for tensor operations
        """
        return torch.device(self.device)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'frame_buffer_size': self.frame_buffer_size,
            'tensor_buffer_size': self.tensor_buffer_size,
            'result_buffer_size': self.result_buffer_size,
            'render_buffer_size': self.render_buffer_size,
            'num_preprocessing_threads': self.num_preprocessing_threads,
            'num_postprocessing_threads': self.num_postprocessing_threads,
            'enable_gpu_threading': self.enable_gpu_threading,
            'gpu_batch_size': self.gpu_batch_size,
            'device': self.device,
            'max_gpu_memory_fraction': self.max_gpu_memory_fraction,
            'enable_visualization': self.enable_visualization,
            'enable_frame_skipping': self.enable_frame_skipping,
            'frame_skip_interval': self.frame_skip_interval,
            'enable_memory_optimization': self.enable_memory_optimization,
            'tensor_cache_size': self.tensor_cache_size,
            'enable_cpu_offload': self.enable_cpu_offload,
            'output_quality': self.output_quality,
            'resize_frames': self.resize_frames,
            'target_resolution': self.target_resolution,
            'enable_profiling': self.enable_profiling,
            'log_queue_sizes': self.log_queue_sizes,
            'enable_progress_callback': self.enable_progress_callback,
            'progress_update_interval': self.progress_update_interval,
            'thread_timeout': self.thread_timeout,
            'queue_timeout': self.queue_timeout,
            'max_error_count': self.max_error_count
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            PipelineConfig instance
        """
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-validate after updates
        self._validate_parameters()
        self._optimize_parameters()


# Predefined optimized configurations
HIGH_PERFORMANCE_CONFIG = PipelineConfig(
    frame_buffer_size=50,
    tensor_buffer_size=25,
    result_buffer_size=30,
    num_preprocessing_threads=4,
    gpu_batch_size=8,
    enable_memory_optimization=True,
    enable_cpu_offload=False,
    enable_profiling=False
)

MEMORY_EFFICIENT_CONFIG = PipelineConfig(
    frame_buffer_size=15,
    tensor_buffer_size=8,
    result_buffer_size=10,
    num_preprocessing_threads=2,
    gpu_batch_size=2,
    enable_memory_optimization=True,
    enable_cpu_offload=True,
    tensor_cache_size=50
)

REALTIME_CONFIG = PipelineConfig(
    frame_buffer_size=10,
    tensor_buffer_size=5,
    result_buffer_size=8,
    num_preprocessing_threads=2,
    gpu_batch_size=1,
    enable_frame_skipping=True,
    frame_skip_interval=2,
    output_quality=85
)

DEBUG_CONFIG = PipelineConfig(
    frame_buffer_size=5,
    tensor_buffer_size=3,
    result_buffer_size=5,
    num_preprocessing_threads=1,
    gpu_batch_size=1,
    enable_profiling=True,
    log_queue_sizes=True,
    enable_progress_callback=True,
    progress_update_interval=10
) 