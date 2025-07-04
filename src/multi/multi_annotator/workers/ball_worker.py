# multi_flow_annotator/workers/ball_worker.py

import queue
from typing import Any, List
import numpy as np
import torch

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

class BallWorker(BaseWorker):
    """ボール検出のためのパイプラインワーカー。

    スライディングウィンドウを使用して時系列情報を考慮したボール検出を行います。

    Attributes:
        sliding_window (List[np.ndarray]): ボール検出用のフレームを保持するスライディングウィンドウ。
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
            predictor: ボール検出モデルの予測器
            preprocess_q: 前処理タスクキュー
            inference_q: 推論タスクキュー
            postprocess_q: 後処理タスクキュー
            results_q: 結果を書き込む共有キュー
            debug: デバッグモードフラグ
        """
        super().__init__("ball", predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)
        self.sliding_window: List[np.ndarray] = []

    def reset_state(self):
        """新しいクリップの処理のために内部状態（スライディングウィンドウ）をリセットします。"""
        self.sliding_window.clear()

    def _process_preprocess_task(self, task):
        """ボール検出の前処理を実行します。

        フレームをスライディングウィンドウに追加し、ウィンドウサイズが満たされたら
        モデルへの入力テンソルを生成して推論キューに追加します。

        Args:
            task (PreprocessTask): 処理する前処理タスク。
        """
        clips_for_inference = []
        meta_for_inference = []

        for i, frame in enumerate(task.frames):
            self.sliding_window.append(frame.copy())
            if len(self.sliding_window) > self.predictor.num_frames:
                self.sliding_window.pop(0)

            if len(self.sliding_window) == self.predictor.num_frames:
                clips_for_inference.append(list(self.sliding_window))
                # メタデータはウィンドウの最後のフレームに対応させる
                meta_for_inference.append(task.meta_data[i])

        if clips_for_inference:
            processed_data = self.predictor.preprocess(clips_for_inference)
            inference_task = InferenceTask(
                task_id=task.task_id,
                tensor_data=processed_data,
                meta_data=meta_for_inference,
                original_frames=clips_for_inference
            )
            self.inference_queue.put(inference_task)
            if self.debug: print(f"[BALL] 推論キューに追加: {task.task_id}, クリップ数: {len(clips_for_inference)}")

    def _process_inference_task(self, task):
        """ボール検出の推論を実行します。

        Args:
            task (InferenceTask): 処理する推論タスク。
        """
        try:
            with torch.no_grad():
                preds = self.predictor.inference(task.tensor_data)
            
            post_task = PostprocessTask(
                task_id=task.task_id,
                inference_output=preds,
                meta_data=task.meta_data,
                original_frames=task.original_frames
            )
            self.postprocess_queue.put(post_task)
            if self.debug: print(f"[BALL] 後処理キューに追加: {task.task_id}")
        
        except Exception as e:
            print(f"[BALL] 推論エラー: {str(e)}")

    def _process_postprocess_task(self, task):
        """ボール検出の後処理を行い、アノテーションを登録します。

        Args:
            task (PostprocessTask): 処理する後処理タスク。
        """
        try:
            results = self.predictor.postprocess(task.inference_output, task.original_frames)
            
            for (img_id, _), result in zip(task.meta_data, results, strict=False):
                if result:  # 有効な結果がある場合
                    self.results_queue.put({
                        "type": "ball",
                        "image_id": img_id,
                        "result": result
                    })
            
            self.processed_counter += len(results)
            if self.debug: print(f"[BALL] 後処理完了: {task.task_id}, 結果数: {len(results)}")
        
        except Exception as e:
            print(f"[BALL] 後処理エラー: {str(e)}")