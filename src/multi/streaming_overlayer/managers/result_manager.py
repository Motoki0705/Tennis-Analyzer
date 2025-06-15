"""
高度拡張性を持つストリーミング処理パイプライン - ResultManager実装

ResultManagerは、ワーカーからの結果を集約し、
最終成果物の完成を判定してPipelineRunnerに通知する責務を持ちます。
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..core.interfaces import ItemId, ResultManagerInterface, TopicName
from .task_manager import TaskManager

logger = logging.getLogger(__name__)


class ResultManager(ResultManagerInterface):
    """
    結果の集約と最終成果物の生成を管理するマネージャー。
    
    設計書に基づき、以下の機能を提供します：
    - ワーカーからの結果受信と保存
    - TaskManagerへの結果利用可能通知
    - 最終成果物の完成判定
    - PipelineRunnerへの完成成果物の送信
    """
    
    def __init__(self, task_manager: TaskManager, results_queue: queue.Queue,
                 completion_topics: Optional[Set[TopicName]] = None,
                 max_cache_size: int = 200, debug: bool = False):
        """
        Args:
            task_manager: TaskManagerのインスタンス（通知先）
            results_queue: ワーカーからの結果を受信するキュー
            completion_topics: 最終成果物の完成に必要なトピックのセット
                              Noneの場合は全トピックが必要
            max_cache_size: 結果キャッシュの最大サイズ
            debug: デバッグモード
        """
        self.task_manager = task_manager
        self.results_queue = results_queue
        self.completion_topics = completion_topics
        self.max_cache_size = max_cache_size
        self.debug = debug
        
        # 結果キャッシュ
        self.frame_cache: Dict[ItemId, Dict[str, Any]] = {}
        
        # 完成した最終成果物のキュー
        self.completed_results_queue = queue.Queue()
        
        # スレッド制御
        self.running = False
        self.process_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # 統計情報
        self.stats = {
            "total_results_received": 0,
            "total_completed_items": 0,
            "current_cache_size": 0,
            "cache_evictions": 0,
            "processing_errors": 0
        }
    
    def start(self) -> None:
        """結果処理ループを開始します。"""
        if self.running:
            logger.warning("ResultManager is already running")
            return
        
        self.running = True
        self.process_thread = threading.Thread(
            target=self._process_loop,
            name="ResultManager_process",
            daemon=True
        )
        self.process_thread.start()
        
        logger.info("ResultManager started")
    
    def stop(self) -> None:
        """結果処理ループを停止します。"""
        if not self.running:
            return
        
        self.running = False
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        logger.info("ResultManager stopped")
    
    def get_completed_results(self) -> Iterator[Tuple[ItemId, Dict[str, Any], Any]]:
        """
        完成した最終成果物のイテレータを返します。
        
        Yields:
            Tuple[ItemId, Dict[str, Any], Any]: (item_id, final_result, original_data)
        """
        while True:
            try:
                result = self.completed_results_queue.get(timeout=0.1)
                yield result
            except queue.Empty:
                if not self.running:
                    break
                continue
            except Exception as e:
                logger.error(f"Error getting completed result: {e}")
                break
    
    def _process_loop(self) -> None:
        """結果処理の内部ループ"""
        while self.running:
            try:
                # ワーカーからの結果を受信
                result_message = self.results_queue.get(timeout=0.1)
                self._handle_result_message(result_message)
                self.results_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ResultManager process loop error: {e}")
                self.stats["processing_errors"] += 1
                
                if self.debug:
                    import traceback
                    traceback.print_exc()
    
    def _handle_result_message(self, result_message: Dict[str, Any]) -> None:
        """
        ワーカーからの結果メッセージを処理します。
        
        Args:
            result_message: ワーカーからの結果メッセージ
                {
                    "item_id": ItemId,
                    "topic": TopicName, 
                    "data": ResultData,
                    "worker_name": str,
                    "processing_time": float,
                    "timestamp": float
                }
        """
        with self.lock:
            item_id = result_message["item_id"]
            topic = result_message["topic"]
            data = result_message["data"]
            
            # 結果をキャッシュに保存
            self._store_result(item_id, topic, data, result_message)
            
            # TaskManagerに結果利用可能を通知
            self.task_manager.notify_result_available(item_id, topic)
            
            # TaskManagerの結果辞書も更新
            self.task_manager.update_result(item_id, topic, data)
            
            # 最終成果物の完成をチェック
            if self._is_item_completed(item_id):
                self._generate_final_result(item_id)
            
            # キャッシュサイズ管理
            self._manage_cache_size()
            
            # 統計更新
            self.stats["total_results_received"] += 1
            self.stats["current_cache_size"] = len(self.frame_cache)
    
    def _store_result(self, item_id: ItemId, topic: TopicName, data: Any,
                     metadata: Dict[str, Any]) -> None:
        """結果をキャッシュに保存します。"""
        if item_id not in self.frame_cache:
            self.frame_cache[item_id] = {
                "results": {},
                "metadata": {},
                "original_data": None,
                "completion_time": None,
                "first_result_time": time.time()
            }
        
        # 結果とメタデータを保存
        self.frame_cache[item_id]["results"][topic] = data
        self.frame_cache[item_id]["metadata"][topic] = {
            "worker_name": metadata.get("worker_name"),
            "processing_time": metadata.get("processing_time"),
            "timestamp": metadata.get("timestamp")
        }
        
        # 元データの取得（TaskManagerから）
        if item_id in self.task_manager.pending_items:
            self.frame_cache[item_id]["original_data"] = \
                self.task_manager.pending_items[item_id]["original_data"]
    
    def _is_item_completed(self, item_id: ItemId) -> bool:
        """
        アイテムの処理が完了したかどうかを判定します。
        
        Args:
            item_id: アイテムの一意識別子
        
        Returns:
            bool: 完了した場合True
        """
        if item_id not in self.frame_cache:
            return False
        
        item_results = self.frame_cache[item_id]["results"]
        
        # 完成に必要なトピックが指定されている場合
        if self.completion_topics:
            return self.completion_topics.issubset(set(item_results.keys()))
        
        # 全トピックが必要な場合（デフォルト）
        # TaskManagerから全トピックリストを取得
        all_topics = set(self.task_manager.dependency_graph.keys())
        return all_topics.issubset(set(item_results.keys()))
    
    def _generate_final_result(self, item_id: ItemId) -> None:
        """
        最終成果物を生成し、完成キューに送信します。
        
        Args:
            item_id: アイテムの一意識別子
        """
        if item_id not in self.frame_cache:
            logger.error(f"Cannot generate final result for unknown item: {item_id}")
            return
        
        item_data = self.frame_cache[item_id]
        
        # 完成時刻を記録
        item_data["completion_time"] = time.time()
        
        # 最終成果物を構築
        final_result = {
            "results": item_data["results"].copy(),
            "metadata": {
                "processing_metadata": item_data["metadata"],
                "total_processing_time": item_data["completion_time"] - item_data["first_result_time"],
                "completion_time": item_data["completion_time"],
                "item_id": item_id
            }
        }
        
        original_data = item_data["original_data"]
        
        # 完成キューに送信
        try:
            completed_item = (item_id, final_result, original_data)
            self.completed_results_queue.put(completed_item, timeout=0.1)
            
            # TaskManagerから完成したアイテムを削除
            self.task_manager.remove_completed_item(item_id)
            
            # 統計更新
            self.stats["total_completed_items"] += 1
            
            logger.debug(f"Generated final result for item {item_id}")
            
        except queue.Full:
            logger.warning(f"Completed results queue is full for item {item_id}")
        except Exception as e:
            logger.error(f"Error generating final result for item {item_id}: {e}")
    
    def _manage_cache_size(self) -> None:
        """キャッシュサイズを管理し、必要に応じて古い結果を削除します。"""
        while len(self.frame_cache) > self.max_cache_size:
            # 最も古いアイテムを削除（first_result_timeが最も小さいもの）
            oldest_item_id = min(
                self.frame_cache.keys(),
                key=lambda x: self.frame_cache[x]["first_result_time"]
            )
            
            # 完成していないアイテムの削除をログに記録
            if not self._is_item_completed(oldest_item_id):
                logger.warning(f"Evicting incomplete item from cache: {oldest_item_id}")
            
            del self.frame_cache[oldest_item_id]
            self.stats["cache_evictions"] += 1
    
    def set_completion_topics(self, topics: Set[TopicName]) -> None:
        """
        最終成果物の完成に必要なトピックを設定します。
        
        Args:
            topics: 必要なトピックのセット
        """
        with self.lock:
            self.completion_topics = topics
            logger.info(f"Set completion topics: {topics}")
    
    def get_stats(self) -> Dict[str, Any]:
        """ResultManagerの統計情報を返します。"""
        with self.lock:
            stats = self.stats.copy()
            stats["completion_topics"] = list(self.completion_topics) if self.completion_topics else None
            stats["completed_queue_size"] = self.completed_results_queue.qsize()
            return stats
    
    def get_cache_status(self) -> Dict[str, Any]:
        """キャッシュの詳細状況を返します（デバッグ用）。"""
        with self.lock:
            cache_status = {}
            
            for item_id, item_data in self.frame_cache.items():
                cache_status[str(item_id)] = {
                    "available_topics": list(item_data["results"].keys()),
                    "is_completed": self._is_item_completed(item_id),
                    "first_result_time": item_data["first_result_time"],
                    "completion_time": item_data["completion_time"]
                }
            
            return cache_status 