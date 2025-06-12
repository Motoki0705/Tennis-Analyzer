# streaming_annotator/workers/court_worker.py

import torch
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Dict, Optional, Tuple

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

logger = logging.getLogger(__name__)


class CourtWorker(BaseWorker):
    """
    コート検出のためのマルチフローパイプラインワーカー。
    
    前処理→推論→後処理の3段階を完全に分離し、
    効率的なバッチ処理を実装します。
    """
    
    def __init__(self, name, predictor, queue_set, results_q, debug=False):
        # QueueManagerから基本キューを取得
        preprocess_q = queue_set.get_queue("preprocess")
        inference_q = queue_set.get_queue("inference")
        postprocess_q = queue_set.get_queue("postprocess")
        
        super().__init__(name, predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)
        
        # スレッドプール（前処理・後処理用）
        self.preprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_preprocess")
        self.postprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_postprocess")
        
        # パフォーマンス監視
        self.preprocess_count = 0
        self.inference_count = 0
        self.postprocess_count = 0
        
    def stop(self):
        """ワーカーとスレッドプールを停止します。"""
        # スレッドプールをシャットダウン
        if hasattr(self, 'preprocess_pool'):
            self.preprocess_pool.shutdown(wait=False)
        if hasattr(self, 'postprocess_pool'):
            self.postprocess_pool.shutdown(wait=False)
        
        # ベースクラスの停止処理
        super().stop()

    def _process_preprocess_task(self, task):
        """
        前処理タスクを処理します。
        
        コート検出用の前処理を並列実行します。
        
        Args:
            task: PreprocessTask - フレームデータとメタデータを含む
        """
        try:
            # 前処理をスレッドプールで並列実行
            future = self.preprocess_pool.submit(self._execute_preprocess, task)
            
            # 結果を取得（ノンブロッキング）
            try:
                processed_data, original_shapes = future.result(timeout=5.0)  # 5秒でタイムアウト
                
                if processed_data is not None:
                    # 推論タスクをキューに送信
                    inference_task = InferenceTask(
                        task.task_id, 
                        processed_data, 
                        task.meta_data,
                        original_shapes=original_shapes
                    )
                    self.inference_queue.put(inference_task)
                    
                    self.preprocess_count += 1
                    
                    if self.debug:
                        logger.debug(f"CourtWorker前処理完了: {task.task_id}, frames={len(task.frames)}")
                else:
                    if self.debug:
                        logger.debug(f"CourtWorker前処理失敗: {task.task_id}")
                        
            except TimeoutError:
                logger.error(f"CourtWorker前処理タイムアウト: {task.task_id}")
            except Exception as e:
                logger.error(f"CourtWorker前処理実行エラー: {task.task_id}, {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"CourtWorker前処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _execute_preprocess(self, task) -> Tuple[Any, List[Tuple[int, int]]]:
        """
        実際の前処理を実行します。
        
        Args:
            task: PreprocessTask
            
        Returns:
            (processed_data, original_shapes) のタプル
        """
        try:
            # CourtPredictorの前処理を実行
            processed_data, original_shapes = self.predictor.preprocess(task.frames)
            
            return processed_data, original_shapes
            
        except Exception as e:
            logger.error(f"CourtWorker前処理実行中にエラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None, []

    def _process_inference_task(self, task):
        """
        推論タスクを処理します。
        
        Args:
            task: InferenceTask - 前処理済みデータを含む
        """
        try:
            # GPU推論を実行
            with torch.no_grad():
                court_predictions = self.predictor.inference(task.tensor_data)
            
            # 後処理タスクをキューに送信
            postprocess_task = PostprocessTask(
                task.task_id,
                court_predictions,
                task.meta_data,
                original_shapes=getattr(task, 'original_shapes', None)
            )
            self.postprocess_queue.put(postprocess_task)
            
            self.inference_count += 1
            
            if self.debug:
                logger.debug(f"CourtWorker推論完了: {task.task_id}")
                
        except Exception as e:
            logger.error(f"CourtWorker推論エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_postprocess_task(self, task):
        """
        後処理タスクを処理します。
        
        Args:
            task: PostprocessTask - 推論結果とメタデータを含む
        """
        try:
            # 後処理をスレッドプールで並列実行
            future = self.postprocess_pool.submit(self._execute_postprocess, task)
            
            # 結果を取得（ノンブロッキング）
            try:
                results = future.result(timeout=3.0)  # 3秒でタイムアウト
                
                if results:
                    # 結果をフレームごとに分解し、結果キューに送信
                    for i, result in enumerate(results):
                        if task.meta_data and i < len(task.meta_data):
                            frame_idx = task.meta_data[i][0]
                        else:
                            frame_idx = i
                        
                        # (フレームインデックス, "タスク名", 結果) のタプルを格納
                        self.results_queue.put((frame_idx, "court", result))
                    
                    self.postprocess_count += 1
                    
                    if self.debug:
                        logger.debug(f"CourtWorker後処理完了: {task.task_id}, results={len(results)}")
                else:
                    if self.debug:
                        logger.debug(f"CourtWorker後処理結果なし: {task.task_id}")
                        
            except TimeoutError:
                logger.error(f"CourtWorker後処理タイムアウト: {task.task_id}")
            except Exception as e:
                logger.error(f"CourtWorker後処理実行エラー: {task.task_id}, {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"CourtWorker後処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _execute_postprocess(self, task) -> Optional[List[Any]]:
        """
        実際の後処理を実行します。
        
        Args:
            task: PostprocessTask
            
        Returns:
            処理結果のリスト
        """
        try:
            # メタデータからオリジナルフレームの形状情報を復元
            original_shapes = getattr(task, 'original_shapes', None)
            if original_shapes is None:
                # フォールバック: メタデータから形状を復元
                original_shapes = [(meta[1], meta[2]) for meta in task.meta_data]
            
            # CourtPredictorの後処理を実行
            kps_list_batch, _ = self.predictor.postprocess(task.inference_output, original_shapes)
            
            return kps_list_batch
            
        except Exception as e:
            logger.error(f"CourtWorker後処理実行中にエラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def get_performance_stats(self) -> Dict[str, int]:
        """パフォーマンス統計を取得します。"""
        return {
            "preprocess_count": self.preprocess_count,
            "inference_count": self.inference_count,
            "postprocess_count": self.postprocess_count
        }