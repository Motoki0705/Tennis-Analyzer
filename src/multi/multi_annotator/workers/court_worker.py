# multi_flow_annotator/workers/court_worker.py

import torch
from typing import List

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

class CourtWorker(BaseWorker):
    """コート検出のためのワーカークラス"""
    
    def __init__(
        self, 
        predictor, 
        preprocess_q, 
        inference_q, 
        postprocess_q, 
        results_q, 
        debug=False
    ):
        """初期化

        Args:
            predictor: コート検出モデルの予測器
            preprocess_q: 前処理タスクキュー
            inference_q: 推論タスクキュー
            postprocess_q: 後処理タスクキュー
            results_q: 結果を書き込む共有キュー
            debug: デバッグモードフラグ
        """
        super().__init__("court", predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)
    
    def _process_preprocess_task(self, task):
        """前処理タスクを処理
        
        フレームをモデル入力形式に変換し、推論キューに送る
        """
        try:
            # 前処理を実行
            processed_data, original_shapes = self.predictor.preprocess(task.frames)
            
            # 推論タスクを作成
            inference_task = InferenceTask(
                task_id=task.task_id,
                tensor_data=processed_data,
                meta_data=task.meta_data,
                original_frames=task.frames
            )
            self.inference_queue.put(inference_task)
            
            if self.debug:
                print(f"[COURT] 推論キューに追加: {task.task_id}, フレーム数: {len(task.frames)}")
        
        except Exception as e:
            print(f"[COURT] 前処理エラー: {str(e)}")
    
    def _process_inference_task(self, task):
        """推論タスクを処理
        
        モデルによる推論を実行し、結果を後処理キューに送る
        """
        try:
            # 推論実行
            with torch.no_grad():
                preds = self.predictor.inference(task.tensor_data)
            
            # 後処理タスクを作成
            post_task = PostprocessTask(
                task_id=task.task_id,
                inference_output=preds,
                meta_data=task.meta_data,
                original_frames=task.original_frames
            )
            self.postprocess_queue.put(post_task)
            
            if self.debug:
                print(f"[COURT] 後処理キューに追加: {task.task_id}")
        
        except Exception as e:
            print(f"[COURT] 推論エラー: {str(e)}")
    
    def _process_postprocess_task(self, task):
        """後処理タスクを処理
        
        推論結果を処理してアノテーションを生成し、結果キューに送る
        """
        try:
            # 画像のサイズ情報を抽出
            shapes = [frame.shape[:2] for frame in task.original_frames]
            
            # 後処理を実行
            kps_list, confidences = self.predictor.postprocess(task.inference_output, shapes)
            
            # 結果キューに追加
            for (img_id, _), kps in zip(task.meta_data, kps_list, strict=False):
                if kps:  # 有効なキーポイントがある場合
                    self.results_queue.put({
                        "type": "court",
                        "image_id": img_id,
                        "result": kps
                    })
            
            if self.debug:
                print(f"[COURT] 後処理完了: {task.task_id}, 結果数: {len(kps_list)}")
        
        except Exception as e:
            print(f"[COURT] 後処理エラー: {str(e)}")