"""
Task registry for managing different analysis tasks.
Provides factory methods for creating task instances with standardized interfaces.
"""

import logging
from typing import Any, Dict, Type

import torch
from omegaconf import DictConfig

from ..tasks.base_task import BaseTask

log = logging.getLogger(__name__)


class TaskRegistry:
    """Registry for creating and managing analysis tasks."""
    
    def __init__(self):
        self._task_classes: Dict[str, Type[BaseTask]] = {}
        self._register_builtin_tasks()
    
    def _register_builtin_tasks(self) -> None:
        """Register built-in task types."""
        try:
            from ..tasks.ball_detection_task import BallDetectionTask
            self.register_task_class('ball', BallDetectionTask)
        except ImportError as e:
            log.warning(f"Could not register ball detection task: {e}")
        
        try:
            from ..tasks.court_detection_task import CourtDetectionTask
            self.register_task_class('court', CourtDetectionTask)
        except ImportError as e:
            log.warning(f"Could not register court detection task: {e}")
        
        try:
            from ..tasks.player_detection_task import PlayerDetectionTask
            self.register_task_class('player', PlayerDetectionTask)
        except ImportError as e:
            log.warning(f"Could not register player detection task: {e}")
        
        try:
            from ..tasks.pose_estimation_task import PoseEstimationTask
            self.register_task_class('pose', PoseEstimationTask)
        except ImportError as e:
            log.warning(f"Could not register pose estimation task: {e}")
        
        try:
            from ..tasks.event_classification_task import EventClassificationTask
            self.register_task_class('event', EventClassificationTask)
        except ImportError as e:
            log.warning(f"Could not register event classification task: {e}")
    
    def register_task_class(self, task_name: str, task_class: Type[BaseTask]) -> None:
        """
        Register a task class.
        
        Args:
            task_name: Name of the task
            task_class: Task class that inherits from BaseTask
        """
        if not issubclass(task_class, BaseTask):
            raise ValueError(f"Task class must inherit from BaseTask: {task_class}")
        
        self._task_classes[task_name] = task_class
        log.debug(f"Registered task class: {task_name} -> {task_class.__name__}")
    
    def create_task(self, task_name: str, config: DictConfig, device: torch.device) -> BaseTask:
        """
        Create a task instance.
        
        Args:
            task_name: Name of the task to create
            config: Task configuration
            device: PyTorch device
            
        Returns:
            Initialized task instance
        """
        if task_name not in self._task_classes:
            available_tasks = list(self._task_classes.keys())
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {available_tasks}")
        
        task_class = self._task_classes[task_name]
        
        try:
            task_instance = task_class(config, device)
            log.info(f"Created task instance: {task_name}")
            return task_instance
            
        except Exception as e:
            log.error(f"Failed to create task {task_name}: {e}")
            raise
    
    def get_available_tasks(self) -> Dict[str, Type[BaseTask]]:
        """Get all available task classes."""
        return self._task_classes.copy()
    
    def is_task_available(self, task_name: str) -> bool:
        """Check if a task is available."""
        return task_name in self._task_classes