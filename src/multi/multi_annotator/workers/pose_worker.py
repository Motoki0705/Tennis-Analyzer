# multi_flow_annotator/workers/pose_worker.py

import torch
from typing import List, Dict, Any

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

class PoseWorker(BaseWorker):
    """プレーヤー検出とポーズ推定のためのワーカークラス
    
    プレーヤーの検出とポーズ推定を2段階で処理する
    """
    
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
            predictor: ポーズ推定モデルの予測器（プレーヤー検出機能も含む）
            preprocess_q: 前処理タスクキュー
            inference_q: 推論タスクキュー
            postprocess_q: 後処理タスクキュー
            results_q: 結果を書き込む共有キュー
            debug: デバッグモードフラグ
        """
        super().__init__("pose", predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)
    
    def _process_preprocess_task(self, task):
        """前処理タスクを処理
        
        フレームを物体検出モデルの入力形式に変換し、推論キューに送る
        """
        try:
            # プレーヤー検出の前処理
            processed_data = self.predictor.preprocess_detection(task.frames)
            
            # 推論タスクを作成
            inference_task = InferenceTask(
                task_id=task.task_id,
                tensor_data=processed_data,
                meta_data=task.meta_data,
                original_frames=task.frames
            )
            self.inference_queue.put(inference_task)
            
            if self.debug:
                print(f"[POSE] 検出推論キューに追加: {task.task_id}, フレーム数: {len(task.frames)}")
        
        except Exception as e:
            print(f"[POSE] 前処理エラー: {str(e)}")
    
    def _process_inference_task(self, task):
        """推論タスクを処理
        
        プレーヤー検出の推論を実行し、結果を後処理キューに送る
        """
        try:
            # プレーヤー検出の推論
            with torch.no_grad():
                preds = self.predictor.inference_detection(task.tensor_data)
            
            # 後処理タスクを作成
            post_task = PostprocessTask(
                task_id=task.task_id,
                inference_output=preds,  # 検出結果
                meta_data=task.meta_data,
                original_frames=task.original_frames
            )
            self.postprocess_queue.put(post_task)
            
            if self.debug:
                print(f"[POSE] 後処理キューに追加: {task.task_id}")
        
        except Exception as e:
            print(f"[POSE] 検出推論エラー: {str(e)}")
    
    def _process_postprocess_task(self, task):
        """後処理タスクを処理
        
        プレーヤー検出の後処理を行い、検出された人物に対してポーズ推定を実行し、
        最終結果を結果キューに送る
        """
        try:
            # 1. プレーヤー検出の後処理
            det_outputs = task.inference_output
            batch_boxes, batch_scores, batch_valid, images_for_pose = self.predictor.postprocess_detection(
                det_outputs, task.original_frames
            )
            
            if not images_for_pose:
                # プレーヤーが検出されなかった場合は空の結果を登録
                for img_id, _ in task.meta_data:
                    self.results_queue.put({
                        "type": "pose",
                        "image_id": img_id,
                        "result": []
                    })
                
                if self.debug:
                    print(f"[POSE] プレーヤー非検出: {task.task_id}")
                return
            
            # 2. ポーズ推定の実行
            try:
                # ポーズ推定の前処理
                pose_inputs = self.predictor.preprocess_pose(images_for_pose, batch_boxes)
                
                # ポーズ推定の推論
                with torch.no_grad():
                    pose_outputs = self.predictor.inference_pose(pose_inputs)
                
                # ポーズ推定の後処理
                pose_results = self.predictor.postprocess_pose(
                    pose_outputs, batch_boxes, batch_scores, batch_valid, len(task.original_frames)
                )
                
                # 3. 結果キューに追加
                for (img_id, _), poses in zip(task.meta_data, pose_results, strict=False):
                    self.results_queue.put({
                        "type": "pose",
                        "image_id": img_id,
                        "result": poses
                    })
                
                if self.debug:
                    print(f"[POSE] 後処理完了: {task.task_id}, 画像数: {len(task.meta_data)}")
                    
            except Exception as e:
                print(f"[POSE] ポーズ推定エラー: {str(e)}")
                # エラー時も空の結果を登録して処理を継続
                for img_id, _ in task.meta_data:
                    self.results_queue.put({
                        "type": "pose",
                        "image_id": img_id,
                        "result": []
                    })
        
        except Exception as e:
            print(f"[POSE] 後処理エラー: {str(e)}")