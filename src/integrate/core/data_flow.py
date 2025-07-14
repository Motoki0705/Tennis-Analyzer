"""
Data flow management for the flexible tennis analysis pipeline.
"""
from typing import Any, Dict, List, Optional, Union
import logging
import threading
import queue
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class DataStage(Enum):
    """Enumeration of pipeline data stages."""
    INPUT = "input"
    PREPROCESSED = "preprocessed"
    RAW_OUTPUT = "raw_output"
    POSTPROCESSED = "postprocessed"
    VISUALIZED = "visualized"


@dataclass
class DataPacket:
    """
    Container for data flowing through the pipeline.
    """
    frame_indices: List[int]
    frames: List[Any]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_data: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    
    def add_task_data(self, task_name: str, stage: DataStage, data: Any, metadata: Optional[Dict] = None) -> None:
        """Add data for a specific task and stage."""
        self.task_data[task_name][stage.value] = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
    
    def get_task_data(self, task_name: str, stage: DataStage) -> Optional[Any]:
        """Get data for a specific task and stage."""
        return self.task_data.get(task_name, {}).get(stage.value, {}).get('data')
    
    def get_task_metadata(self, task_name: str, stage: DataStage) -> Dict[str, Any]:
        """Get metadata for a specific task and stage."""
        return self.task_data.get(task_name, {}).get(stage.value, {}).get('metadata', {})
    
    def has_task_data(self, task_name: str, stage: DataStage) -> bool:
        """Check if data exists for a specific task and stage."""
        return stage.value in self.task_data.get(task_name, {})
    
    def get_dependency_data(self, dependencies: List[str], stage: DataStage) -> Dict[str, Any]:
        """Get data from dependency tasks at a specific stage."""
        result = {}
        for dep in dependencies:
            data = self.get_task_data(dep, stage)
            if data is not None:
                result[dep] = data
        return result
    
    def __len__(self) -> int:
        """Return the number of frames in this packet."""
        return len(self.frames)


class DataFlow:
    """
    Manages data flow through the pipeline with buffering and synchronization.
    """
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self._data_history: deque = deque(maxlen=buffer_size)
        self._lock = threading.RLock()
        self._stats = {
            'packets_processed': 0,
            'total_frames': 0,
            'processing_times': defaultdict(list)
        }
    
    def create_packet(self, frame_indices: List[int], frames: List[Any], 
                     metadata: Optional[Dict] = None) -> DataPacket:
        """Create a new data packet."""
        packet = DataPacket(
            frame_indices=frame_indices,
            frames=frames,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._data_history.append(packet)
            self._stats['packets_processed'] += 1
            self._stats['total_frames'] += len(frames)
        
        return packet
    
    def get_packet_by_frame_index(self, frame_index: int) -> Optional[DataPacket]:
        """Find packet containing a specific frame index."""
        with self._lock:
            for packet in reversed(self._data_history):
                if frame_index in packet.frame_indices:
                    return packet
        return None
    
    def get_recent_packets(self, count: int) -> List[DataPacket]:
        """Get the most recent N packets."""
        with self._lock:
            return list(self._data_history)[-count:] if count <= len(self._data_history) else list(self._data_history)
    
    def record_processing_time(self, task_name: str, stage: str, processing_time: float) -> None:
        """Record processing time for performance tracking."""
        with self._lock:
            self._stats['processing_times'][f"{task_name}_{stage}"].append(processing_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data flow statistics."""
        with self._lock:
            stats = self._stats.copy()
            
            # Calculate average processing times
            avg_times = {}
            for key, times in self._stats['processing_times'].items():
                if times:
                    avg_times[key] = {
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'count': len(times)
                    }
            
            stats['avg_processing_times'] = avg_times
            stats['buffer_utilization'] = len(self._data_history) / self.buffer_size
            
            return stats
    
    def clear(self) -> None:
        """Clear all stored data and reset statistics."""
        with self._lock:
            self._data_history.clear()
            self._stats = {
                'packets_processed': 0,
                'total_frames': 0,
                'processing_times': defaultdict(list)
            }


class ThreadSafeDataFlow(DataFlow):
    """
    Thread-safe data flow manager with queues for multi-threaded pipeline execution.
    """
    
    def __init__(self, buffer_size: int = 100, queue_size: int = 50):
        super().__init__(buffer_size)
        self.queue_size = queue_size
        
        # Thread-safe queues for different pipeline stages
        self.input_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.processing_queues: Dict[str, queue.Queue] = {}
        self.output_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        
        # Event for pipeline coordination
        self.shutdown_event = threading.Event()
    
    def create_processing_queue(self, name: str) -> queue.Queue:
        """Create a named processing queue."""
        self.processing_queues[name] = queue.Queue(maxsize=self.queue_size)
        return self.processing_queues[name]
    
    def put_input(self, packet: DataPacket, timeout: Optional[float] = None) -> bool:
        """Put packet into input queue."""
        try:
            self.input_queue.put(packet, timeout=timeout)
            return True
        except queue.Full:
            log.warning("Input queue is full, dropping packet")
            return False
    
    def get_input(self, timeout: Optional[float] = None) -> Optional[DataPacket]:
        """Get packet from input queue."""
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def put_processing(self, queue_name: str, packet: DataPacket, 
                      timeout: Optional[float] = None) -> bool:
        """Put packet into named processing queue."""
        if queue_name not in self.processing_queues:
            raise ValueError(f"Processing queue '{queue_name}' does not exist")
        
        try:
            self.processing_queues[queue_name].put(packet, timeout=timeout)
            return True
        except queue.Full:
            log.warning(f"Processing queue '{queue_name}' is full, dropping packet")
            return False
    
    def get_processing(self, queue_name: str, timeout: Optional[float] = None) -> Optional[DataPacket]:
        """Get packet from named processing queue."""
        if queue_name not in self.processing_queues:
            raise ValueError(f"Processing queue '{queue_name}' does not exist")
        
        try:
            return self.processing_queues[queue_name].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def put_output(self, packet: DataPacket, timeout: Optional[float] = None) -> bool:
        """Put packet into output queue."""
        try:
            self.output_queue.put(packet, timeout=timeout)
            return True
        except queue.Full:
            log.warning("Output queue is full, dropping packet")
            return False
    
    def get_output(self, timeout: Optional[float] = None) -> Optional[DataPacket]:
        """Get packet from output queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def signal_shutdown(self) -> None:
        """Signal all workers to shutdown."""
        self.shutdown_event.set()
        
        # Put sentinel values to unblock waiting workers
        try:
            self.input_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        for q in self.processing_queues.values():
            try:
                q.put(None, timeout=1.0)
            except queue.Full:
                pass
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_event.is_set()
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about queue utilization."""
        stats = {
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'processing_queues': {
                name: q.qsize() for name, q in self.processing_queues.items()
            },
            'queue_capacity': self.queue_size,
            'shutdown_requested': self.is_shutdown_requested()
        }
        
        # Add utilization percentages
        stats['input_utilization'] = stats['input_queue_size'] / self.queue_size * 100
        stats['output_utilization'] = stats['output_queue_size'] / self.queue_size * 100
        stats['processing_utilization'] = {
            name: size / self.queue_size * 100 
            for name, size in stats['processing_queues'].items()
        }
        
        return stats