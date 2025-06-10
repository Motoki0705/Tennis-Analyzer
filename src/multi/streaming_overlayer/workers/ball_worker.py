# streaming_annotator/workers/ball_worker.py

import torch
import logging

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

logger = logging.getLogger(__name__)


class BallWorker(BaseWorker):
    """
    ボール検知のためのパイプラインワーカー。
    
    ボール位置と信頼度を検知し、結果をキューに送信します。
    """

    def _process_preprocess_task(self, task):
        """
        前処理タスクを処理します。
        
        Args:
            task: PreprocessTask - フレームデータとメタデータを含む
        """
        try:
            # 予測器の前処理メソッドを呼び出し
            processed_data, meta_info = self.predictor.preprocess(task.frames)
            
            # 推論タスクをキューに送信
            self.inference_queue.put(InferenceTask(task.task_id, processed_data, task.meta_data))
            
            if self.debug:
                logger.debug(f"BallWorker前処理完了: {task.task_id}, frames={len(task.frames)}")
                
        except Exception as e:
            logger.error(f"BallWorker前処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_inference_task(self, task):
        """
        推論タスクを処理します。
        
        Args:
            task: InferenceTask - 前処理済みデータを含む
        """
        try:
            with torch.no_grad():
                # 予測器の推論メソッドを呼び出し
                preds = self.predictor.inference(task.tensor_data)
            
            # 後処理タスクをキューに送信
            self.postprocess_queue.put(PostprocessTask(task.task_id, preds, task.meta_data))
            
            if self.debug:
                logger.debug(f"BallWorker推論完了: {task.task_id}")
                
        except Exception as e:
            logger.error(f"BallWorker推論エラー: {e}")
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
            # メタデータからオリジナルフレームの形状情報を復元
            original_shapes = [(meta[1], meta[2]) for meta in task.meta_data] if task.meta_data else None
            
            # 予測器の後処理メソッドを呼び出し
            ball_results = self.predictor.postprocess(task.inference_output, original_shapes)
            
            # 結果をフレームごとに分解し、フレームインデックスを付けて結果キューに追加
            if isinstance(ball_results, (list, tuple)):
                # バッチ処理の場合
                for i, ball_result_per_frame in enumerate(ball_results):
                    if task.meta_data and i < len(task.meta_data):
                        frame_idx = task.meta_data[i][0]
                    else:
                        frame_idx = i
                    
                    # (フレームインデックス, "タスク名", 結果) のタプルを格納
                    self.results_queue.put((frame_idx, "ball", ball_result_per_frame))
            else:
                # 単一フレームの場合
                frame_idx = task.meta_data[0][0] if task.meta_data else 0
                self.results_queue.put((frame_idx, "ball", ball_results))
            
            if self.debug:
                logger.debug(f"BallWorker後処理完了: {task.task_id}")
                
        except Exception as e:
            logger.error(f"BallWorker後処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
