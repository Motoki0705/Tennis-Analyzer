# streaming_annotator/workers/base_worker.py
# (前回の回答とほぼ同じなので、要点のみ記載)
import queue
import threading
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..core.interfaces import ItemId, TopicName, TaskData, ResultData

logger = logging.getLogger(__name__)

class BatchTask:
    """バッチ処理用のタスクコンテナ"""
    
    def __init__(self, item_id: ItemId, task_data: TaskData, 
                 dependencies: Dict[TopicName, ResultData], submit_time: float):
        self.item_id = item_id
        self.task_data = task_data
        self.dependencies = dependencies
        self.submit_time = submit_time

class BaseWorker(ABC):
    """
    個別の処理を実行するワーカーの基底クラス。
    
    設計書の通り、依存関係の宣言とトピックベースの
    結果パブリッシュをサポートします。
    バッチ推論機能を含む。
    """
    
    def __init__(self, name: str, results_queue: queue.Queue, 
                 max_concurrent_tasks: int = 3, debug: bool = False,
                 batch_size: int = 1, batch_timeout: float = 0.1,
                 task_queue_maxsize: int = 1000):
        """
        Args:
            name: ワーカーの一意な名前
            results_queue: 結果を出力するキュー
            max_concurrent_tasks: 最大同時実行タスク数
            debug: デバッグモード
            batch_size: バッチサイズ（1の場合は個別処理）
            batch_timeout: バッチ待機タイムアウト（秒）
            task_queue_maxsize: タスクキューの最大サイズ
        """
        self.name = name
        self.results_queue = results_queue
        self.max_concurrent_tasks = max_concurrent_tasks
        self.debug = debug
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # 内部管理用
        self.running = False
        self.threads: List[threading.Thread] = []
        self.task_queue = queue.Queue(maxsize=task_queue_maxsize)
        
        # バッチ処理用
        self.batch_buffer: List[BatchTask] = []
        self.batch_lock = threading.RLock()
        self.last_batch_time = time.time()
        
        # 処理統計
        self.stats = {
            "total_processed": 0,
            "total_errors": 0,
            "start_time": None,
            "average_processing_time": 0.0,
            "last_processing_time": 0.0,
            "batch_count": 0,
            "average_batch_size": 0.0
        }
        
        # スレッド制御
        self.lock = threading.RLock()
    
    @abstractmethod
    def get_published_topic(self) -> TopicName:
        """
        このワーカーが出版する結果のトピック名を返します。
        
        Returns:
            str: トピック名（例: "ball_detection", "court_detection", "event_detection"）
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[TopicName]:
        """
        このワーカーが依存するトピック名のリストを返します。
        
        Returns:
            List[str]: 依存するトピック名のリスト
                空リストの場合は依存関係なし（上流ワーカー）
        """
        pass
    
    def process_task(self, item_id: ItemId, task_data: TaskData, 
                    dependencies: Dict[TopicName, ResultData]) -> ResultData:
        """
        単一タスクの処理（後方互換性のため残す）。
        
        デフォルトではprocess_batchを呼び出します。
        特殊な処理が必要な場合はオーバーライドしてください。
        
        Args:
            item_id: アイテムの一意識別子
            task_data: 処理対象のデータ
            dependencies: 依存するトピックの結果辞書
        
        Returns:
            ResultData: 処理結果
        """
        # 単一タスクをバッチとして処理
        batch_tasks = [BatchTask(item_id, task_data, dependencies, time.time())]
        batch_results = self.process_batch(batch_tasks)
        
        if batch_results and len(batch_results) > 0:
            return batch_results[0]
        else:
            raise RuntimeError(f"No result returned from process_batch for item {item_id}")
    
    @abstractmethod
    def process_batch(self, batch_tasks: List[BatchTask]) -> List[ResultData]:
        """
        バッチタスクの処理を実行します。
        
        Args:
            batch_tasks: バッチタスクのリスト
        
        Returns:
            List[ResultData]: 処理結果のリスト（batch_tasksと同じ順序）
        """
        pass
    
    def start(self) -> None:
        """ワーカーを開始します。"""
        if self.running:
            logger.warning(f"{self.name} worker is already running")
            return
        
        with self.lock:
            self.running = True
            self.stats["start_time"] = time.time()
            
            # ワーカースレッドを起動
            for i in range(self.max_concurrent_tasks):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"{self.name}_worker_{i}",
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
            
            # バッチタイムアウト監視スレッドを起動
            if self.batch_size > 1:
                timeout_thread = threading.Thread(
                    target=self._batch_timeout_loop,
                    name=f"{self.name}_batch_timeout",
                    daemon=True
                )
                timeout_thread.start()
                self.threads.append(timeout_thread)
            
            logger.info(f"Started {self.name} worker with {len(self.threads)} threads (batch_size={self.batch_size})")
    
    def stop(self) -> None:
        """ワーカーを停止します。"""
        if not self.running:
            return
        
        with self.lock:
            self.running = False
            
            # 残りのバッチを処理
            self._flush_batch()
            
            # スレッドの終了を待機
            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            
            self.threads.clear()
            logger.info(f"Stopped {self.name} worker")
    
    def submit_task(self, item_id: ItemId, task_data: TaskData,
                   dependencies: Dict[TopicName, ResultData]) -> bool:
        """
        タスクをワーカーに送信します。
        
        Args:
            item_id: アイテムの一意識別子
            task_data: 処理対象のデータ
            dependencies: 依存するトピックの結果辞書
        
        Returns:
            bool: タスクの送信が成功したかどうか
        """
        try:
            task = {
                "item_id": item_id,
                "task_data": task_data,
                "dependencies": dependencies,
                "submit_time": time.time()
            }
            
            self.task_queue.put(task, timeout=0.1)
            return True
        
        except queue.Full:
            logger.warning(f"{self.name} worker task queue is full")
            return False
        except Exception as e:
            logger.error(f"Error submitting task to {self.name}: {e}")
            return False
    
    def _worker_loop(self) -> None:
        """ワーカーの内部処理ループ"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
                
                if self.batch_size <= 1:
                    # 個別処理モード
                    self._process_task_safe(
                        task["item_id"],
                        task["task_data"],
                        task["dependencies"]
                    )
                else:
                    # バッチ処理モード
                    self._add_to_batch(
                        task["item_id"],
                        task["task_data"],
                        task["dependencies"],
                        task["submit_time"]
                    )
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} worker loop error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
    
    def _add_to_batch(self, item_id: ItemId, task_data: TaskData,
                     dependencies: Dict[TopicName, ResultData], submit_time: float) -> None:
        """タスクをバッチバッファに追加します"""
        with self.batch_lock:
            batch_task = BatchTask(item_id, task_data, dependencies, submit_time)
            self.batch_buffer.append(batch_task)
            
            # バッチサイズに達したら処理
            if len(self.batch_buffer) >= self.batch_size:
                self._process_batch_safe()
    
    def _batch_timeout_loop(self) -> None:
        """バッチタイムアウト監視ループ"""
        while self.running:
            try:
                time.sleep(self.batch_timeout / 2)  # タイムアウトの半分の間隔でチェック
                
                with self.batch_lock:
                    if (self.batch_buffer and 
                        time.time() - self.last_batch_time > self.batch_timeout):
                        self._process_batch_safe()
                        
            except Exception as e:
                logger.error(f"{self.name} batch timeout loop error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
    
    def _process_batch_safe(self) -> None:
        """バッチを安全に処理します"""
        if not self.batch_buffer:
            return
        
        # バッファからバッチを取得
        current_batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_batch_time = time.time()
        
        start_time = time.time()
        
        try:
            # バッチ処理を実行
            results = self.process_batch(current_batch)
            
            if len(results) != len(current_batch):
                raise ValueError(f"Batch result count mismatch: expected {len(current_batch)}, got {len(results)}")
            
            # 処理時間を計算
            processing_time = time.time() - start_time
            
            # 各結果を個別に出力
            for batch_task, result in zip(current_batch, results):
                self._publish_result(batch_task.item_id, result, processing_time / len(current_batch))
            
            # 統計を更新
            with self.lock:
                self.stats["total_processed"] += len(current_batch)
                self.stats["batch_count"] += 1
                self.stats["last_processing_time"] = processing_time
                
                # 平均処理時間を更新
                total = self.stats["total_processed"]
                current_avg = self.stats["average_processing_time"]
                avg_per_item = processing_time / len(current_batch)
                self.stats["average_processing_time"] = \
                    (current_avg * (total - len(current_batch)) + avg_per_item * len(current_batch)) / total
                
                # 平均バッチサイズを更新
                batch_count = self.stats["batch_count"]
                current_batch_avg = self.stats["average_batch_size"]
                self.stats["average_batch_size"] = \
                    (current_batch_avg * (batch_count - 1) + len(current_batch)) / batch_count
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self.lock:
                self.stats["total_errors"] += len(current_batch)
            
            logger.error(f"{self.name} batch processing error: {e}")
            
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー結果を各タスクに対して出力
            for batch_task in current_batch:
                self._publish_error_result(batch_task.item_id, e, processing_time / len(current_batch))
    
    def _flush_batch(self) -> None:
        """残りのバッチを強制的に処理します"""
        with self.batch_lock:
            if self.batch_buffer:
                logger.info(f"{self.name} flushing remaining {len(self.batch_buffer)} tasks")
                self._process_batch_safe()
    
    def _process_task_safe(self, item_id: ItemId, task_data: TaskData,
                          dependencies: Dict[TopicName, ResultData]) -> None:
        """
        タスクを安全に処理し、結果を出力します（個別処理モード用）。
        
        Args:
            item_id: アイテムの一意識別子
            task_data: 処理対象のデータ
            dependencies: 依存するトピックの結果辞書
        """
        start_time = time.time()
        
        try:
            # 実際の処理を実行
            result = self.process_task(item_id, task_data, dependencies)
            
            # 処理時間を計算
            processing_time = time.time() - start_time
            
            # 結果を出力
            self._publish_result(item_id, result, processing_time)
            
            # 統計を更新
            with self.lock:
                self.stats["total_processed"] += 1
                self.stats["last_processing_time"] = processing_time
                
                # 平均処理時間を更新
                total = self.stats["total_processed"]
                current_avg = self.stats["average_processing_time"]
                self.stats["average_processing_time"] = \
                    (current_avg * (total - 1) + processing_time) / total
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self.lock:
                self.stats["total_errors"] += 1
            
            logger.error(f"{self.name} processing error for item {item_id}: {e}")
            
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー結果を出力（オプション）
            self._publish_error_result(item_id, e, processing_time)
    
    def _publish_result(self, item_id: ItemId, result: ResultData, 
                       processing_time: float) -> None:
        """
        処理結果をResultManagerに送信します。
        
        Args:
            item_id: アイテムの一意識別子
            result: 処理結果
            processing_time: 処理時間（秒）
        """
        try:
            result_message = {
                "item_id": item_id,
                "topic": self.get_published_topic(),
                "data": result,
                "worker_name": self.name,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            
            self.results_queue.put(result_message)
            
            if self.debug:
                logger.debug(f"{self.name} published result for item {item_id} "
                           f"(processing time: {processing_time:.3f}s)")
        
        except Exception as e:
            logger.error(f"Error publishing result from {self.name}: {e}")
    
    def _publish_error_result(self, item_id: ItemId, error: Exception, 
                             processing_time: float) -> None:
        """
        エラー結果をResultManagerに送信します。
        
        Args:
            item_id: アイテムの一意識別子
            error: 発生したエラー
            processing_time: 処理時間（秒）
        """
        try:
            error_result = {
                "error": True,
                "error_message": str(error),
                "error_type": type(error).__name__
            }
            
            result_message = {
                "item_id": item_id,
                "topic": self.get_published_topic(),
                "data": error_result,
                "worker_name": self.name,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            
            self.results_queue.put(result_message)
            
        except Exception as e:
            logger.error(f"Error publishing error result from {self.name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        ワーカーの統計情報を返します。
        
        Returns:
            Dict[str, Any]: 統計情報の辞書
        """
        with self.lock:
            stats = self.stats.copy()
            
            # 実行時間を追加
            if stats["start_time"]:
                stats["uptime"] = time.time() - stats["start_time"]
            else:
                stats["uptime"] = 0.0
            
            # 成功率を追加
            total = stats["total_processed"] + stats["total_errors"]
            if total > 0:
                stats["success_rate"] = stats["total_processed"] / total
            else:
                stats["success_rate"] = 0.0
            
            # キューサイズを追加
            stats["queue_size"] = self.task_queue.qsize()
            stats["max_queue_size"] = self.task_queue.maxsize
            
            # バッチ関連統計を追加
            stats["batch_size"] = self.batch_size
            stats["current_batch_buffer_size"] = len(self.batch_buffer)
            
            return stats
    
    def is_upstream_worker(self) -> bool:
        """
        このワーカーが上流ワーカー（依存関係なし）かどうかを返します。
        
        Returns:
            bool: 上流ワーカーの場合True
        """
        return len(self.get_dependencies()) == 0
    
    def get_worker_info(self) -> Dict[str, Any]:
        """
        ワーカーの基本情報を返します。
        
        Returns:
            Dict[str, Any]: ワーカー情報の辞書
        """
        return {
            "name": self.name,
            "published_topic": self.get_published_topic(),
            "dependencies": self.get_dependencies(),
            "is_upstream": self.is_upstream_worker(),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "batch_size": self.batch_size,
            "batch_timeout": self.batch_timeout,
            "running": self.running,
            "thread_count": len(self.threads)
        }