"""
高度拡張性を持つストリーミング処理パイプライン - インターフェース定義

このモジュールは、パイプライン内の各コンポーネントの
抽象的なインターフェースを定義します。
"""

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# 型エイリアス
ItemId = Union[str, int]
TopicName = str
TaskData = Any
ResultData = Any


class InputHandler(ABC):
    """
    データソースから処理単位（データアイテム）を取り出すハンドラの基底クラス。
    
    様々な入力形式（動画ファイル、フレームリスト、ディレクトリなど）を
    統一的なインターフェースで扱えるよう抽象化します。
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[ItemId, Any]]:
        """
        データアイテムのイテレータを返します。
        
        Returns:
            Iterator[Tuple[ItemId, Any]]: (item_id, data) のタプルのイテレータ
                item_id: 一意な識別子（例: フレームインデックス）
                data: 処理対象データ（例: np.ndarrayフレーム）
        """
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        データソースのメタ情報を返します。
        
        Returns:
            Dict[str, Any]: データソースのプロパティ辞書
                例: {"fps": 30, "width": 1920, "height": 1080, "total_frames": 1000}
        """
        pass
    
    def close(self) -> None:
        """
        リソースを解放します。デフォルト実装では何もしません。
        必要に応じて継承クラスでオーバーライドしてください。
        """
        pass


class OutputHandler(ABC):
    """
    パイプラインの最終成果物を処理するハンドラの基底クラス。
    
    様々な出力形式（動画ファイル、JSON、リアルタイム表示など）を
    統一的なインターフェースで扱えるよう抽象化します。
    """
    
    @abstractmethod
    def setup(self, properties: Dict[str, Any]) -> None:
        """
        InputHandlerのプロパティを元に初期設定を行います。
        
        Args:
            properties: InputHandler.get_properties()から取得したプロパティ
        """
        pass
    
    @abstractmethod
    def handle_result(self, item_id: ItemId, final_result: Dict[str, Any], 
                     original_data: Any) -> None:
        """
        最終成果物を処理します。
        
        Args:
            item_id: アイテムの一意識別子
            final_result: パイプラインで生成された最終結果
            original_data: 元のデータ（必要に応じて）
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        リソースを解放します（例: ファイルを閉じる）。
        """
        pass


class BaseWorker(ABC):
    """
    個別の処理を実行するワーカーの基底クラス。
    
    設計書の通り、依存関係の宣言とトピックベースの
    結果パブリッシュをサポートします。
    """
    
    def __init__(self, name: str, results_queue: queue.Queue, 
                 max_concurrent_tasks: int = 3, debug: bool = False):
        """
        Args:
            name: ワーカーの一意な名前
            results_queue: 結果を出力するキュー
            max_concurrent_tasks: 最大同時実行タスク数
            debug: デバッグモード
        """
        self.name = name
        self.results_queue = results_queue
        self.max_concurrent_tasks = max_concurrent_tasks
        self.debug = debug
        
        # 内部管理用
        self.running = False
        self.threads: List[threading.Thread] = []
        self.task_queue = queue.Queue(maxsize=max_concurrent_tasks * 2)
        
        # 処理統計
        self.stats = {
            "total_processed": 0,
            "total_errors": 0,
            "start_time": None
        }
    
    @abstractmethod
    def get_published_topic(self) -> TopicName:
        """
        このワーカーが出版する結果のトピック名を返します。
        
        Returns:
            str: トピック名（例: "ball_result", "court_result", "event_result"）
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
    
    @abstractmethod
    def process_task(self, item_id: ItemId, task_data: TaskData, 
                    dependencies: Dict[TopicName, ResultData]) -> ResultData:
        """
        実際の処理を実行します。
        
        Args:
            item_id: アイテムの一意識別子
            task_data: 処理対象のデータ
            dependencies: 依存するトピックの結果辞書
        
        Returns:
            ResultData: 処理結果
        """
        pass
    
    def start(self) -> None:
        """ワーカーを開始します。"""
        if self.running:
            logger.warning(f"{self.name} worker is already running")
            return
        
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
        
        logger.info(f"Started {self.name} worker with {len(self.threads)} threads")
    
    def stop(self) -> None:
        """ワーカーを停止します。"""
        if not self.running:
            return
        
        self.running = False
        
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
            bool: タスクが正常に送信されたかどうか
        """
        try:
            self.task_queue.put((item_id, task_data, dependencies), timeout=0.1)
            return True
        except queue.Full:
            logger.warning(f"{self.name} worker task queue is full")
            return False
    
    def _worker_loop(self) -> None:
        """ワーカーの処理ループ"""
        while self.running:
            try:
                item_id, task_data, dependencies = self.task_queue.get(timeout=0.1)
                self._process_task_safe(item_id, task_data, dependencies)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} worker loop error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
    
    def _process_task_safe(self, item_id: ItemId, task_data: TaskData,
                          dependencies: Dict[TopicName, ResultData]) -> None:
        """
        タスクを安全に処理し、結果をパブリッシュします。
        """
        try:
            start_time = time.time()
            result = self.process_task(item_id, task_data, dependencies)
            processing_time = time.time() - start_time
            
            # 結果をパブリッシュ
            self._publish_result(item_id, result, processing_time)
            
            self.stats["total_processed"] += 1
            
        except Exception as e:
            logger.error(f"{self.name} task processing error for item {item_id}: {e}")
            self.stats["total_errors"] += 1
            
            if self.debug:
                import traceback
                traceback.print_exc()
    
    def _publish_result(self, item_id: ItemId, result: ResultData, 
                       processing_time: float) -> None:
        """
        結果をResultManagerに送信します。
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
            
            self.results_queue.put(result_message, timeout=0.1)
            
        except queue.Full:
            logger.warning(f"{self.name} results queue is full")
        except Exception as e:
            logger.error(f"{self.name} result publishing error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """ワーカーの統計情報を返します。"""
        stats = self.stats.copy()
        if stats["start_time"]:
            stats["uptime"] = time.time() - stats["start_time"]
        stats["queue_size"] = self.task_queue.qsize()
        return stats


class TaskManagerInterface(ABC):
    """タスクマネージャーのインターフェース"""
    
    @abstractmethod
    def setup(self, workers: List[BaseWorker]) -> None:
        """ワーカー群から依存関係グラフを構築します。"""
        pass
    
    @abstractmethod
    def submit_item(self, item_id: ItemId, data: Any) -> None:
        """新しいアイテムの処理を開始します。"""
        pass
    
    @abstractmethod
    def notify_result_available(self, item_id: ItemId, topic: TopicName) -> None:
        """結果が利用可能になったことを通知します。"""
        pass


class ResultManagerInterface(ABC):
    """結果マネージャーのインターフェース"""
    
    @abstractmethod
    def start(self) -> None:
        """結果処理ループを開始します。"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """結果処理ループを停止します。"""
        pass
    
    @abstractmethod
    def get_completed_results(self) -> Iterator[Tuple[ItemId, Dict[str, Any], Any]]:
        """完成した最終成果物のイテレータを返します。"""
        pass 