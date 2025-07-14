"""
Task management system for the flexible tennis analysis pipeline.
"""
from typing import Dict, List, Any, Optional, Set
import logging
import importlib
from collections import defaultdict, deque
import time

from .base_task import BaseTask, TaskExecutionResult

log = logging.getLogger(__name__)


class TaskManager:
    """
    Manages task registration, dependency resolution, and execution orchestration.
    """
    
    def __init__(self):
        self.tasks: Dict[str, BaseTask] = {}
        self.task_configs: Dict[str, Dict] = {}
        self.execution_order: List[str] = []
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def register_task(self, task: BaseTask) -> None:
        """
        Register a task in the manager.
        
        Args:
            task: Task instance to register
        """
        if task.name in self.tasks:
            log.warning(f"Task '{task.name}' is already registered. Overwriting.")
        
        self.tasks[task.name] = task
        self.task_configs[task.name] = task.config
        
        # Update dependency graphs
        self._dependency_graph[task.name] = set(task.get_dependencies())
        for dep in task.get_dependencies():
            self._reverse_dependency_graph[dep].add(task.name)
        
        # Recalculate execution order
        self._calculate_execution_order()
        
        log.info(f"Registered task: {task}")
    
    def register_task_from_config(self, task_config: Dict[str, Any], device) -> None:
        """
        Register a task from configuration dictionary.
        
        Args:
            task_config: Task configuration including module path and parameters
            device: PyTorch device for task initialization
        """
        try:
            # Import task class dynamically
            module_path = task_config['module']
            class_name = task_config.get('class_name', module_path.split('.')[-1])
            
            module = importlib.import_module(module_path)
            task_class = getattr(module, class_name)
            
            # Create full config dictionary
            full_config = task_config.get('config', {}).copy()
            full_config['dependencies'] = task_config.get('dependencies', [])
            full_config['enabled'] = task_config.get('enabled', True)
            full_config['critical'] = task_config.get('critical', True)
            
            task = task_class(
                name=task_config['name'],
                config=full_config,
                device=device
            )
            
            self.register_task(task)
            
        except Exception as e:
            log.error(f"Failed to register task from config {task_config.get('name', 'unknown')}: {e}")
            raise
    
    def get_task(self, name: str) -> Optional[BaseTask]:
        """Get task by name."""
        return self.tasks.get(name)
    
    def get_enabled_tasks(self) -> List[BaseTask]:
        """Get list of enabled tasks in execution order."""
        return [self.tasks[name] for name in self.execution_order if self.tasks[name].enabled]
    
    def _calculate_execution_order(self) -> None:
        """
        Calculate task execution order using topological sort.
        Ensures dependencies are executed before dependent tasks.
        """
        # Only consider enabled tasks
        enabled_tasks = {name: task for name, task in self.tasks.items() if task.enabled}
        
        if not enabled_tasks:
            self.execution_order = []
            return
        
        # Build dependency graph for enabled tasks only
        graph = {}
        in_degree = {}
        
        for task_name in enabled_tasks:
            graph[task_name] = []
            in_degree[task_name] = 0
        
        for task_name, task in enabled_tasks.items():
            for dep in task.get_dependencies():
                if dep in enabled_tasks:  # Only consider enabled dependencies
                    graph[dep].append(task_name)
                    in_degree[task_name] += 1
                elif dep not in self.tasks:
                    log.warning(f"Task '{task_name}' depends on undefined task '{dep}'. Ignoring dependency.")
                else:
                    log.warning(f"Task '{task_name}' depends on disabled task '{dep}'. Ignoring dependency.")
        
        # Topological sort using Kahn's algorithm
        queue = deque([task for task, degree in in_degree.items() if degree == 0])
        order = []
        
        while queue:
            current = queue.popleft()
            order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(order) != len(enabled_tasks):
            remaining = set(enabled_tasks.keys()) - set(order)
            raise ValueError(f"Circular dependency detected in tasks: {remaining}")
        
        self.execution_order = order
        log.info(f"Task execution order: {' -> '.join(order)}")
    
    def validate_dependencies(self) -> bool:
        """
        Validate that all task dependencies are satisfied.
        
        Returns:
            True if all dependencies are valid, False otherwise
        """
        issues = []
        
        for task_name, task in self.tasks.items():
            if not task.enabled:
                continue
                
            for dep in task.get_dependencies():
                if dep not in self.tasks:
                    issues.append(f"Task '{task_name}' depends on undefined task '{dep}'")
                elif not self.tasks[dep].enabled:
                    issues.append(f"Task '{task_name}' depends on disabled task '{dep}'")
        
        if issues:
            for issue in issues:
                log.error(issue)
            return False
        
        return True
    
    def execute_pipeline(self, frames: List[Any], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute all enabled tasks in dependency order.
        
        Args:
            frames: Input frames batch
            metadata: Optional input metadata
            
        Returns:
            Dictionary containing results from all tasks
        """
        if not self.validate_dependencies():
            raise RuntimeError("Pipeline validation failed. Check task dependencies.")
        
        results = {}
        execution_results = {}
        
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            
            if not task.enabled:
                continue
            
            log.debug(f"Executing task: {task_name}")
            start_time = time.perf_counter()
            
            try:
                # Gather dependency results
                dependency_results = {}
                for dep in task.get_dependencies():
                    if dep in results:
                        dependency_results[dep] = results[dep]
                
                # Execute task
                task_results = task.execute(frames, metadata, dependency_results)
                results[task_name] = task_results
                
                execution_time = time.perf_counter() - start_time
                execution_results[task_name] = TaskExecutionResult(
                    task_name=task_name,
                    results=task_results,
                    execution_time=execution_time,
                    success=True
                )
                
                log.debug(f"Task '{task_name}' completed in {execution_time:.3f}s")
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                execution_results[task_name] = TaskExecutionResult(
                    task_name=task_name,
                    results={},
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
                
                log.error(f"Task '{task_name}' failed: {e}")
                
                # Decide whether to continue or abort based on task criticality
                if task.config.get('critical', True):
                    raise RuntimeError(f"Critical task '{task_name}' failed: {e}")
                else:
                    log.warning(f"Non-critical task '{task_name}' failed, continuing pipeline")
        
        # Add execution metadata
        results['_execution_meta'] = {
            'execution_results': execution_results,
            'total_tasks': len(self.execution_order),
            'successful_tasks': sum(1 for r in execution_results.values() if r.success),
            'failed_tasks': sum(1 for r in execution_results.values() if not r.success),
            'total_execution_time': sum(r.execution_time for r in execution_results.values())
        }
        
        return results
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about registered tasks."""
        return {
            'total_tasks': len(self.tasks),
            'enabled_tasks': len([t for t in self.tasks.values() if t.enabled]),
            'execution_order': self.execution_order,
            'dependency_graph': dict(self._dependency_graph),
            'tasks': {name: {
                'enabled': task.enabled,
                'dependencies': task.get_dependencies(),
                'output_keys': task.get_output_keys()
            } for name, task in self.tasks.items()}
        }
    
    def visualize_dependency_graph(self) -> str:
        """
        Create a text representation of the dependency graph.
        Useful for debugging and documentation.
        """
        lines = ["Task Dependency Graph:", "=" * 50]
        
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            status = "✓" if task.enabled else "✗"
            deps = task.get_dependencies()
            dep_str = " <- " + ", ".join(deps) if deps else ""
            lines.append(f"{status} {task_name}{dep_str}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all registered tasks."""
        self.tasks.clear()
        self.task_configs.clear()
        self.execution_order.clear()
        self._dependency_graph.clear()
        self._reverse_dependency_graph.clear()
        log.info("Cleared all registered tasks")