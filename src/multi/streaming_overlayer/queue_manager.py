"""
拡張可能なキューシステムを管理するQueueManagerクラス
"""
import queue
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class QueueConfig:
    """キュー設定を定義するクラス"""
    name: str
    maxsize: int = 16
    queue_type: str = "Queue"  # "Queue", "PriorityQueue", "LifoQueue"
    description: str = ""


@dataclass
class WorkerQueueSet:
    """ワーカー用キューセットを定義するクラス"""
    worker_name: str
    base_queues: Dict[str, queue.Queue] = field(default_factory=dict)
    extended_queues: Dict[str, queue.Queue] = field(default_factory=dict)
    
    def get_queue(self, queue_name: str) -> Optional[queue.Queue]:
        """指定されたキューを取得"""
        if queue_name in self.base_queues:
            return self.base_queues[queue_name]
        elif queue_name in self.extended_queues:
            return self.extended_queues[queue_name]
        return None
    
    def add_queue(self, queue_name: str, q: queue.Queue, is_extended: bool = False):
        """キューを追加"""
        if is_extended:
            self.extended_queues[queue_name] = q
        else:
            self.base_queues[queue_name] = q


class QueueManager:
    """
    拡張可能なキューシステムを管理するクラス
    
    各ワーカーに対して基本キューセット（preprocess, inference, postprocess）
    および拡張キューセット（custom pipeline用）を提供します。
    """
    
    def __init__(self):
        self.worker_queue_sets: Dict[str, WorkerQueueSet] = {}
        self.results_queue: Optional[queue.Queue] = None
        self.queue_configs: Dict[str, QueueConfig] = {}
        
        # デフォルトキュー設定
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """デフォルトキュー設定を初期化"""
        default_configs = [
            QueueConfig("preprocess", 16, "Queue", "前処理タスクキュー"),
            QueueConfig("inference", 16, "Queue", "推論タスクキュー"),
            QueueConfig("postprocess", 16, "Queue", "後処理タスクキュー"),
            QueueConfig("results", 100, "PriorityQueue", "結果集約キュー"),
            # 拡張キュー用設定
            QueueConfig("detection_inference", 32, "Queue", "Detection推論専用キュー"),
            QueueConfig("detection_postprocess", 32, "Queue", "Detection後処理専用キュー"),
            QueueConfig("pose_inference", 32, "Queue", "Pose推論専用キュー"),
            QueueConfig("pose_postprocess", 32, "Queue", "Pose後処理専用キュー"),
            QueueConfig("ball_inference", 32, "Queue", "Ball推論専用キュー"),
            QueueConfig("court_inference", 32, "Queue", "Court推論専用キュー"),
        ]
        
        for config in default_configs:
            self.queue_configs[config.name] = config
    
    def add_queue_config(self, config: QueueConfig):
        """新しいキュー設定を追加"""
        self.queue_configs[config.name] = config
    
    def create_queue_from_config(self, config_name: str) -> queue.Queue:
        """設定からキューを作成"""
        if config_name not in self.queue_configs:
            raise ValueError(f"Queue configuration '{config_name}' not found")
        
        config = self.queue_configs[config_name]
        
        if config.queue_type == "PriorityQueue":
            return queue.PriorityQueue(maxsize=config.maxsize)
        elif config.queue_type == "LifoQueue":
            return queue.LifoQueue(maxsize=config.maxsize)
        else:  # Default to Queue
            return queue.Queue(maxsize=config.maxsize)
    
    def initialize_worker_queues(self, worker_name: str, extended_queue_names: Optional[List[str]] = None):
        """
        ワーカー用キューセットを初期化
        
        Args:
            worker_name: ワーカー名
            extended_queue_names: 拡張キュー名のリスト
        """
        if worker_name in self.worker_queue_sets:
            raise ValueError(f"Worker '{worker_name}' already has queue set initialized")
        
        queue_set = WorkerQueueSet(worker_name)
        
        # 基本キューセットの作成
        base_queue_names = ["preprocess", "inference", "postprocess"]
        for queue_name in base_queue_names:
            q = self.create_queue_from_config(queue_name)
            queue_set.add_queue(queue_name, q, is_extended=False)
        
        # 拡張キューセットの作成
        if extended_queue_names:
            for queue_name in extended_queue_names:
                q = self.create_queue_from_config(queue_name)
                queue_set.add_queue(queue_name, q, is_extended=True)
        
        self.worker_queue_sets[worker_name] = queue_set
    
    def initialize_results_queue(self):
        """結果キューを初期化"""
        self.results_queue = self.create_queue_from_config("results")
    
    def get_worker_queue_set(self, worker_name: str) -> Optional[WorkerQueueSet]:
        """ワーカーのキューセットを取得"""
        return self.worker_queue_sets.get(worker_name)
    
    def get_queue(self, worker_name: str, queue_name: str) -> Optional[queue.Queue]:
        """指定されたワーカーの指定されたキューを取得"""
        queue_set = self.get_worker_queue_set(worker_name)
        if queue_set:
            return queue_set.get_queue(queue_name)
        return None
    
    def get_results_queue(self) -> Optional[queue.Queue]:
        """結果キューを取得"""
        return self.results_queue
    
    def get_queue_status(self) -> Dict[str, Any]:
        """全キューの状態を取得"""
        status = {
            "workers": {},
            "results_queue_size": self.results_queue.qsize() if self.results_queue else 0
        }
        
        for worker_name, queue_set in self.worker_queue_sets.items():
            worker_status = {
                "base_queues": {},
                "extended_queues": {}
            }
            
            for queue_name, q in queue_set.base_queues.items():
                worker_status["base_queues"][queue_name] = q.qsize()
            
            for queue_name, q in queue_set.extended_queues.items():
                worker_status["extended_queues"][queue_name] = q.qsize()
            
            status["workers"][worker_name] = worker_status
        
        return status
    
    def clear_all_queues(self):
        """全キューをクリア"""
        for queue_set in self.worker_queue_sets.values():
            for q in queue_set.base_queues.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            
            for q in queue_set.extended_queues.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
        
        if self.results_queue:
            while not self.results_queue.empty():
                try:
                    self.results_queue.get_nowait()
                except queue.Empty:
                    break


# 設定ベースのワーカー別キュー設定
WORKER_QUEUE_CONFIGS = {
    "ball": {
        "extended_queues": ["ball_inference"]
    },
    "court": {
        "extended_queues": ["court_inference"]
    },
    "pose": {
        "extended_queues": ["detection_inference", "detection_postprocess", "pose_inference", "pose_postprocess"]
    }
}


def create_queue_manager_for_video_predictor(
    worker_names: List[str], 
    custom_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> QueueManager:
    """
    VideoPredictor用のQueueManagerを作成
    
    Args:
        worker_names: ワーカー名のリスト
        custom_configs: カスタムキュー設定
    
    Returns:
        設定済みのQueueManager
    """
    queue_manager = QueueManager()
    
    # カスタム設定があれば適用
    if custom_configs:
        for config_name, config_data in custom_configs.items():
            config = QueueConfig(
                name=config_name,
                maxsize=config_data.get("maxsize", 16),
                queue_type=config_data.get("queue_type", "Queue"),
                description=config_data.get("description", "")
            )
            queue_manager.add_queue_config(config)
    
    # 結果キューを初期化
    queue_manager.initialize_results_queue()
    
    # 各ワーカーのキューセットを初期化
    for worker_name in worker_names:
        extended_queues = None
        if worker_name in WORKER_QUEUE_CONFIGS:
            extended_queues = WORKER_QUEUE_CONFIGS[worker_name].get("extended_queues")
        
        queue_manager.initialize_worker_queues(worker_name, extended_queues)
    
    return queue_manager 