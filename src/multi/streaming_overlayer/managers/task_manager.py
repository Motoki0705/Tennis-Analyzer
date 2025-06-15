"""
高度拡張性を持つストリーミング処理パイプライン - TaskManager実装

TaskManagerは、ワーカー間の依存関係を管理し、
結果が利用可能になった時点で適切なタスクを発行する責務を持ちます。
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Set

from ..core.interfaces import BaseWorker, ItemId, TaskManagerInterface, TopicName

logger = logging.getLogger(__name__)


class TaskManager(TaskManagerInterface):
    """
    タスクフローの全体管理を行うマネージャー。
    
    設計書に基づき、以下の機能を提供します：
    - ワーカー群からの依存関係グラフ構築
    - 依存関係に基づいたタスクの順次発行
    - バックプレッシャーの管理
    """
    
    def __init__(self, max_pending_items: int = 50, debug: bool = False):
        """
        Args:
            max_pending_items: 同時に処理可能なアイテムの最大数
            debug: デバッグモード
        """
        self.max_pending_items = max_pending_items
        self.debug = debug
        
        # ワーカー管理
        self.workers: List[BaseWorker] = []
        self.workers_by_topic: Dict[TopicName, BaseWorker] = {}
        
        # 依存関係グラフ
        self.dependency_graph: Dict[TopicName, List[TopicName]] = {}
        self.reverse_dependency_graph: Dict[TopicName, List[TopicName]] = {}
        self.upstream_workers: List[BaseWorker] = []  # 依存関係のないワーカー
        
        # アイテム状態管理
        self.pending_items: Dict[ItemId, Dict[str, Any]] = {}
        self.waiting_tasks: Dict[ItemId, Dict[TopicName, Dict[TopicName, bool]]] = {}
        
        # スレッド制御
        self.lock = threading.RLock()
        
        # 統計情報
        self.stats = {
            "total_items": 0,
            "completed_items": 0,
            "current_pending": 0,
            "task_dispatch_count": 0
        }
    
    def setup(self, workers: List[BaseWorker]) -> None:
        """
        ワーカー群から依存関係グラフを構築します。
        
        Args:
            workers: 処理に使用するワーカーのリスト
        """
        with self.lock:
            self.workers = workers
            self._build_dependency_graph()
            self._validate_dependency_graph()
            
            logger.info(f"TaskManager setup completed with {len(workers)} workers")
            if self.debug:
                self._log_dependency_graph()
    
    def submit_item(self, item_id: ItemId, data: Any) -> None:
        """
        新しいアイテムの処理を開始します。
        
        Args:
            item_id: アイテムの一意識別子
            data: 処理対象のデータ
        """
        with self.lock:
            # バックプレッシャー制御
            if len(self.pending_items) >= self.max_pending_items:
                logger.warning(f"TaskManager backpressure: max pending items reached ({self.max_pending_items})")
                return
            
            # アイテム状態を初期化
            self.pending_items[item_id] = {
                "original_data": data,
                "results": {},
                "submit_time": time.time()
            }
            
            # 依存関係待ちタスクを初期化
            self.waiting_tasks[item_id] = {}
            for topic, dependencies in self.dependency_graph.items():
                if dependencies:  # 依存関係があるワーカー
                    self.waiting_tasks[item_id][topic] = {dep: False for dep in dependencies}
            
            # 統計更新
            self.stats["total_items"] += 1
            self.stats["current_pending"] = len(self.pending_items)
            
            # 上流ワーカー（依存関係なし）にタスクを発行
            self._dispatch_upstream_tasks(item_id, data)
    
    def notify_result_available(self, item_id: ItemId, topic: TopicName) -> None:
        """
        結果が利用可能になったことを通知し、依存関係を解決します。
        
        Args:
            item_id: アイテムの一意識別子
            topic: 利用可能になったトピック名
        """
        with self.lock:
            if item_id not in self.pending_items:
                logger.warning(f"Received result for unknown item: {item_id}")
                return
            
            # 依存関係を更新
            downstream_topics = self.reverse_dependency_graph.get(topic, [])
            for downstream_topic in downstream_topics:
                if (item_id in self.waiting_tasks and 
                    downstream_topic in self.waiting_tasks[item_id]):
                    
                    # 依存関係を満たしたとマーク
                    self.waiting_tasks[item_id][downstream_topic][topic] = True
                    
                    # 全ての依存関係が満たされたかチェック
                    if self._all_dependencies_satisfied(item_id, downstream_topic):
                        self._dispatch_task(item_id, downstream_topic)
                        
                        # タスクが発行されたので待ちリストから削除
                        del self.waiting_tasks[item_id][downstream_topic]
    
    def _build_dependency_graph(self) -> None:
        """ワーカー群から依存関係グラフを構築します。"""
        self.dependency_graph.clear()
        self.reverse_dependency_graph.clear()
        self.workers_by_topic.clear()
        self.upstream_workers.clear()
        
        # ワーカーからトピックと依存関係を収集
        for worker in self.workers:
            topic = worker.get_published_topic()
            dependencies = worker.get_dependencies()
            
            self.workers_by_topic[topic] = worker
            self.dependency_graph[topic] = dependencies
            
            # 逆方向のグラフも構築
            for dep in dependencies:
                if dep not in self.reverse_dependency_graph:
                    self.reverse_dependency_graph[dep] = []
                self.reverse_dependency_graph[dep].append(topic)
            
            # 上流ワーカー（依存関係なし）を特定
            if not dependencies:
                self.upstream_workers.append(worker)
    
    def _validate_dependency_graph(self) -> None:
        """依存関係グラフの妥当性をチェックします。"""
        # 循環依存のチェック
        if self._has_circular_dependency():
            raise ValueError("Circular dependency detected in worker dependency graph")
        
        # 存在しない依存関係のチェック
        all_topics = set(self.dependency_graph.keys())
        for topic, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                if dep not in all_topics:
                    raise ValueError(f"Worker '{topic}' depends on unknown topic '{dep}'")
        
        # 上流ワーカーが存在することを確認
        if not self.upstream_workers:
            raise ValueError("No upstream workers found - all workers have dependencies")
    
    def _has_circular_dependency(self) -> bool:
        """循環依存の存在をチェックします（トポロジカルソート）。"""
        # カーンのアルゴリズムを使用
        in_degree = {}
        for topic in self.dependency_graph:
            in_degree[topic] = len(self.dependency_graph[topic])
        
        queue = deque([topic for topic, degree in in_degree.items() if degree == 0])
        processed = 0
        
        while queue:
            current = queue.popleft()
            processed += 1
            
            for dependent in self.reverse_dependency_graph.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return processed != len(self.dependency_graph)
    
    def _dispatch_upstream_tasks(self, item_id: ItemId, data: Any) -> None:
        """上流ワーカー（依存関係なし）にタスクを発行します。"""
        for worker in self.upstream_workers:
            success = worker.submit_task(item_id, data, {})
            if success:
                self.stats["task_dispatch_count"] += 1
                logger.debug(f"Dispatched upstream task to {worker.name} for item {item_id}")
            else:
                logger.warning(f"Failed to dispatch upstream task to {worker.name} for item {item_id}")
    
    def _dispatch_task(self, item_id: ItemId, topic: TopicName) -> None:
        """指定されたトピックのワーカーにタスクを発行します。"""
        if topic not in self.workers_by_topic:
            logger.error(f"No worker found for topic: {topic}")
            return
        
        worker = self.workers_by_topic[topic]
        
        # 依存関係の結果を収集
        dependencies = {}
        item_results = self.pending_items[item_id]["results"]
        
        for dep_topic in self.dependency_graph[topic]:
            if dep_topic in item_results:
                dependencies[dep_topic] = item_results[dep_topic]
            else:
                logger.error(f"Missing dependency {dep_topic} for topic {topic}, item {item_id}")
                return
        
        # タスクを発行
        original_data = self.pending_items[item_id]["original_data"]
        success = worker.submit_task(item_id, original_data, dependencies)
        
        if success:
            self.stats["task_dispatch_count"] += 1
            logger.debug(f"Dispatched task to {worker.name} for item {item_id}")
        else:
            logger.warning(f"Failed to dispatch task to {worker.name} for item {item_id}")
    
    def _all_dependencies_satisfied(self, item_id: ItemId, topic: TopicName) -> bool:
        """指定されたトピックの全依存関係が満たされているかチェックします。"""
        if (item_id not in self.waiting_tasks or 
            topic not in self.waiting_tasks[item_id]):
            return False
        
        dependencies_status = self.waiting_tasks[item_id][topic]
        return all(dependencies_status.values())
    
    def update_result(self, item_id: ItemId, topic: TopicName, result: Any) -> None:
        """アイテムの結果を更新します。"""
        with self.lock:
            if item_id in self.pending_items:
                self.pending_items[item_id]["results"][topic] = result
    
    def remove_completed_item(self, item_id: ItemId) -> None:
        """完了したアイテムを管理対象から削除します。"""
        with self.lock:
            if item_id in self.pending_items:
                del self.pending_items[item_id]
                self.stats["completed_items"] += 1
                self.stats["current_pending"] = len(self.pending_items)
            
            if item_id in self.waiting_tasks:
                del self.waiting_tasks[item_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """TaskManagerの統計情報を返します。"""
        with self.lock:
            stats = self.stats.copy()
            stats["worker_count"] = len(self.workers)
            stats["dependency_graph"] = dict(self.dependency_graph)
            return stats
    
    def _log_dependency_graph(self) -> None:
        """依存関係グラフをログ出力します（デバッグ用）。"""
        logger.info("=== Dependency Graph ===")
        for topic, dependencies in self.dependency_graph.items():
            if dependencies:
                logger.info(f"  {topic} <- {dependencies}")
            else:
                logger.info(f"  {topic} (upstream)")
        
        logger.info("=== Reverse Dependency Graph ===")
        for topic, dependents in self.reverse_dependency_graph.items():
            logger.info(f"  {topic} -> {dependents}") 