"""
Base task interface for the flexible tennis analysis pipeline.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import logging

log = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    Abstract base class for all pipeline tasks.
    
    Each task represents a specific analysis component (e.g., ball tracking, court detection)
    and follows a standardized interface for preprocessing, inference, postprocessing, and visualization.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], device: torch.device):
        """
        Initialize the task.
        
        Args:
            name: Unique identifier for this task
            config: Task-specific configuration
            device: PyTorch device for computation
        """
        self.name = name
        self.config = config
        self.device = device
        self.enabled = config.get('enabled', True)
        self.dependencies = config.get('dependencies', [])
        self.is_initialized = False
        
        if self.enabled:
            try:
                self.initialize()
                self.is_initialized = True
                log.info(f"Task '{self.name}' initialized successfully")
            except Exception as e:
                log.error(f"Failed to initialize task '{self.name}': {e}")
                raise
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize task-specific resources (models, preprocessors, etc.).
        Called once during pipeline setup.
        """
        pass
    
    @abstractmethod
    def preprocess(self, frames: List[Any], metadata: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Preprocess input frames for this task.
        
        Args:
            frames: List of input frames (batch)
            metadata: Optional metadata from previous stages
            
        Returns:
            Tuple of (processed_data, preprocessing_metadata)
        """
        pass
    
    @abstractmethod
    def inference(self, preprocessed_data: Any, metadata: Dict) -> Any:
        """
        Run model inference on preprocessed data.
        
        Args:
            preprocessed_data: Output from preprocess stage
            metadata: Preprocessing metadata
            
        Returns:
            Raw model outputs
        """
        pass
    
    @abstractmethod
    def postprocess(self, raw_outputs: Any, metadata: Dict) -> Dict[str, Any]:
        """
        Post-process raw model outputs into standardized format.
        
        Args:
            raw_outputs: Raw model outputs from inference
            metadata: Preprocessing metadata
            
        Returns:
            Structured results dictionary
        """
        pass
    
    @abstractmethod
    def visualize(self, frame: Any, results: Dict[str, Any], vis_config: Dict) -> Any:
        """
        Draw visualization on frame based on task results.
        
        Args:
            frame: Input frame to draw on
            results: Task results from postprocess stage
            vis_config: Visualization configuration
            
        Returns:
            Frame with visualizations drawn
        """
        pass
    
    def execute(self, frames: List[Any], metadata: Optional[Dict] = None, 
                dependency_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete task pipeline: preprocess -> inference -> postprocess.
        
        Args:
            frames: Input frames batch
            metadata: Optional input metadata
            dependency_results: Results from dependency tasks
            
        Returns:
            Task execution results including all stages
        """
        if not self.is_initialized:
            raise RuntimeError(f"Task '{self.name}' is not initialized")
        
        try:
            # Merge dependency results into metadata if available
            if dependency_results:
                metadata = metadata or {}
                metadata['dependencies'] = dependency_results
            
            # Execute pipeline stages
            preprocessed_data, preprocess_meta = self.preprocess(frames, metadata)
            raw_outputs = self.inference(preprocessed_data, preprocess_meta)
            results = self.postprocess(raw_outputs, preprocess_meta)
            
            # Add task metadata to results
            results['_task_meta'] = {
                'name': self.name,
                'frame_count': len(frames),
                'preprocessing_meta': preprocess_meta
            }
            
            return results
            
        except Exception as e:
            log.error(f"Error executing task '{self.name}': {e}")
            raise
    
    def get_dependencies(self) -> List[str]:
        """Get list of task dependencies."""
        return self.dependencies.copy()
    
    def requires_batch_processing(self) -> bool:
        """
        Check if this task requires batch processing.
        Override in subclasses if needed.
        """
        return True
    
    def get_output_keys(self) -> List[str]:
        """
        Get list of output keys this task produces.
        Used for dependency resolution and data flow validation.
        """
        return ['results']
    
    def __str__(self) -> str:
        return f"Task(name='{self.name}', enabled={self.enabled}, dependencies={self.dependencies})"
    
    def __repr__(self) -> str:
        return self.__str__()


class TaskExecutionResult:
    """Container for task execution results with metadata."""
    
    def __init__(self, task_name: str, results: Dict[str, Any], 
                 execution_time: float, success: bool = True, error: Optional[str] = None):
        self.task_name = task_name
        self.results = results
        self.execution_time = execution_time
        self.success = success
        self.error = error
        self.timestamp = torch.utils.data.get_worker_info()
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED: {self.error}"
        return f"TaskResult(task='{self.task_name}', status={status}, time={self.execution_time:.3f}s)"