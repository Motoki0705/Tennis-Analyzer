"""
高度拡張性を持つストリーミング処理パイプライン - PipelineRunner実装

PipelineRunnerは、パイプライン全体のライフサイクルを管理する
オーケストレーターとして機能します。
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from .interfaces import (
    BaseWorker, 
    InputHandler, 
    OutputHandler, 
    ResultManagerInterface, 
    TaskManagerInterface
)
from ..managers.result_manager import ResultManager
from ..managers.task_manager import TaskManager

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    パイプライン全体のライフサイクルを管理するオーケストレーター。
    
    設計書に基づき、以下の責務を持ちます：
    - 各コンポーネントの初期化と起動
    - 入力データの読み込みとTaskManagerへの送信
    - ResultManagerからの完成結果の受信
    - OutputHandlerを通じた結果の出力
    - 全コンポーネントの安全な終了
    """
    
    def __init__(self, 
                 input_handler: InputHandler,
                 output_handler: OutputHandler,
                 workers: List[BaseWorker],
                 max_pending_items: int = 50,
                 max_cache_size: int = 200,
                 completion_topics: Optional[set] = None,
                 debug: bool = False):
        """
        Args:
            input_handler: 入力データを提供するハンドラ
            output_handler: 結果を出力するハンドラ
            workers: 処理を実行するワーカーのリスト
            max_pending_items: TaskManagerの最大同時処理アイテム数
            max_cache_size: ResultManagerの最大キャッシュサイズ
            completion_topics: 最終成果物の完成に必要なトピックのセット
            debug: デバッグモード
        """
        self.input_handler = input_handler
        self.output_handler = output_handler
        self.workers = workers
        self.debug = debug
        
        # 結果キューを作成
        self.results_queue = queue.Queue()
        
        # コアマネージャーを初期化
        self.task_manager = TaskManager(
            max_pending_items=max_pending_items,
            debug=debug
        )
        
        self.result_manager = ResultManager(
            task_manager=self.task_manager,
            results_queue=self.results_queue,
            completion_topics=completion_topics,
            max_cache_size=max_cache_size,
            debug=debug
        )
        
        # 統計情報
        self.stats = {
            "total_items_processed": 0,
            "total_items_output": 0,
            "pipeline_start_time": None,
            "pipeline_end_time": None,
            "processing_errors": 0
        }
        
        # 状態管理
        self.running = False
        self.setup_completed = False
    
    def run(self) -> None:
        """
        パイプラインの実行を開始します。
        
        このメソッドは、全ての処理が完了するまでブロックします。
        """
        try:
            logger.info("Starting pipeline execution")
            self.stats["pipeline_start_time"] = time.time()
            
            # パイプラインのセットアップ
            self._setup_pipeline()
            
            # 実行状態に移行
            self.running = True
            
            # 主処理ループ
            self._main_processing_loop()
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise
        
        finally:
            # 安全にシャットダウン
            self._shutdown_pipeline()
            
            self.stats["pipeline_end_time"] = time.time()
            logger.info("Pipeline execution completed")
            
            if self.debug:
                self._log_final_stats()
    
    def _setup_pipeline(self) -> None:
        """パイプライン全体のセットアップを実行します。"""
        logger.info("Setting up pipeline components")
        
        # 1. OutputHandlerの初期化
        input_properties = self.input_handler.get_properties()
        self.output_handler.setup(input_properties)
        
        # 2. ワーカーの結果キューを設定
        for worker in self.workers:
            worker.results_queue = self.results_queue
        
        # 3. TaskManagerのセットアップ
        self.task_manager.setup(self.workers)
        
        # 4. 全コンポーネントの開始
        self._start_all_components()
        
        self.setup_completed = True
        logger.info("Pipeline setup completed")
    
    def _start_all_components(self) -> None:
        """全コンポーネントを開始します。"""
        # ResultManagerを最初に開始（結果受信の準備）
        self.result_manager.start()
        
        # ワーカーを開始
        for worker in self.workers:
            worker.start()
        
        logger.info(f"Started {len(self.workers)} workers and result manager")
    
    def _main_processing_loop(self) -> None:
        """メインの処理ループを実行します。"""
        logger.info("Starting main processing loop")
        
        # 入力処理とResultManager監視を並行実行
        input_thread = threading.Thread(
            target=self._input_processing_loop,
            name="InputProcessingLoop",
            daemon=True
        )
        
        input_thread.start()
        
        # 完成した結果の処理
        self._output_processing_loop()
        
        # 入力処理の完了を待機
        input_thread.join()
        
        logger.info("Main processing loop completed")
    
    def _input_processing_loop(self) -> None:
        """入力データの処理ループ"""
        try:
            for item_id, data in self.input_handler:
                if not self.running:
                    break
                
                # TaskManagerにアイテムを送信
                self.task_manager.submit_item(item_id, data)
                self.stats["total_items_processed"] += 1
                
                if self.debug and self.stats["total_items_processed"] % 100 == 0:
                    logger.info(f"Processed {self.stats['total_items_processed']} items")
        
        except Exception as e:
            logger.error(f"Input processing error: {e}")
            self.stats["processing_errors"] += 1
            
            if self.debug:
                import traceback
                traceback.print_exc()
        
        finally:
            logger.info("Input processing completed")
    
    def _output_processing_loop(self) -> None:
        """出力処理のループ"""
        logger.info("Starting output processing loop")
        
        try:
            # 完成した結果を処理
            for item_id, final_result, original_data in self.result_manager.get_completed_results():
                if not self.running and self.result_manager.completed_results_queue.empty():
                    # 実行が停止され、キューが空になったら終了
                    break
                
                # OutputHandlerに結果を送信
                self._handle_output_safe(item_id, final_result, original_data)
                self.stats["total_items_output"] += 1
                
                if self.debug and self.stats["total_items_output"] % 100 == 0:
                    logger.info(f"Output {self.stats['total_items_output']} items")
        
        except Exception as e:
            logger.error(f"Output processing error: {e}")
            self.stats["processing_errors"] += 1
            
            if self.debug:
                import traceback
                traceback.print_exc()
        
        finally:
            logger.info("Output processing completed")
    
    def _handle_output_safe(self, item_id, final_result, original_data) -> None:
        """出力処理を安全に実行します。"""
        try:
            self.output_handler.handle_result(item_id, final_result, original_data)
        except Exception as e:
            logger.error(f"Output handling error for item {item_id}: {e}")
            self.stats["processing_errors"] += 1
            
            if self.debug:
                import traceback
                traceback.print_exc()
    
    def _shutdown_pipeline(self) -> None:
        """パイプライン全体を安全にシャットダウンします。"""
        logger.info("Shutting down pipeline")
        
        # 実行状態をクリア
        self.running = False
        
        # 全ワーカーを停止
        for worker in self.workers:
            try:
                worker.stop()
            except Exception as e:
                logger.warning(f"Error stopping worker {worker.name}: {e}")
        
        # ResultManagerを停止
        try:
            self.result_manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping result manager: {e}")
        
        # I/Oハンドラーを閉じる
        try:
            self.input_handler.close()
        except Exception as e:
            logger.warning(f"Error closing input handler: {e}")
        
        try:
            self.output_handler.close()
        except Exception as e:
            logger.warning(f"Error closing output handler: {e}")
        
        logger.info("Pipeline shutdown completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """パイプライン全体の統計情報を返します。"""
        stats = self.stats.copy()
        
        # 実行時間を計算
        if self.stats["pipeline_start_time"]:
            end_time = self.stats["pipeline_end_time"] or time.time()
            stats["total_runtime"] = end_time - self.stats["pipeline_start_time"]
        
        # 各コンポーネントの統計を追加
        if self.setup_completed:
            stats["task_manager"] = self.task_manager.get_stats()
            stats["result_manager"] = self.result_manager.get_stats()
            stats["workers"] = {
                worker.name: worker.get_stats() for worker in self.workers
            }
        
        return stats
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """詳細なステータス情報を返します（デバッグ用）。"""
        status = {
            "running": self.running,
            "setup_completed": self.setup_completed,
            "stats": self.get_stats()
        }
        
        if self.setup_completed:
            status["result_manager_cache"] = self.result_manager.get_cache_status()
        
        return status
    
    def _log_final_stats(self) -> None:
        """最終統計情報をログ出力します（デバッグ用）。"""
        stats = self.get_stats()
        
        logger.info("=== Final Pipeline Statistics ===")
        logger.info(f"Total items processed: {stats['total_items_processed']}")
        logger.info(f"Total items output: {stats['total_items_output']}")
        logger.info(f"Total runtime: {stats.get('total_runtime', 'N/A')} seconds")
        logger.info(f"Processing errors: {stats['processing_errors']}")
        
        if "task_manager" in stats:
            tm_stats = stats["task_manager"]
            logger.info(f"Task manager - Total dispatched: {tm_stats['task_dispatch_count']}")
            logger.info(f"Task manager - Completed items: {tm_stats['completed_items']}")
        
        if "result_manager" in stats:
            rm_stats = stats["result_manager"]
            logger.info(f"Result manager - Results received: {rm_stats['total_results_received']}")
            logger.info(f"Result manager - Cache evictions: {rm_stats['cache_evictions']}")
        
        # ワーカー統計
        if "workers" in stats:
            logger.info("=== Worker Statistics ===")
            for worker_name, worker_stats in stats["workers"].items():
                logger.info(f"{worker_name}: {worker_stats['total_processed']} processed, "
                          f"{worker_stats['total_errors']} errors")
    
    def stop(self) -> None:
        """
        パイプラインの実行を停止します。
        
        非同期で停止信号を送信し、run()メソッドの完了を待つ必要があります。
        """
        logger.info("Stop signal received")
        self.running = False 