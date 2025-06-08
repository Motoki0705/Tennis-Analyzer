# multi_flow_annotator/workers/pose_worker.py

import torch

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

class PoseWorker(BaseWorker):
    """プレーヤー検出とポーズ推定のためのパイプラインワーカー。"""

    def __init__(self, *args, **kwargs):
        """PoseWorkerのインスタンスを初期化します。"""
        super().__init__("pose", *args, **kwargs)

    def _process_preprocess_task(self, task):
        """プレーヤー検出（ポーズ推定の第一段階）の前処理を実行します。

        Args:
            task (PreprocessTask): 処理する前処理タスク。
        """
        # pose_predictorには物体検出の機能が含まれている想定
        processed_data = self.predictor.preprocess_detection(task.frames)
        inference_task = InferenceTask(
            task_id=task.task_id,
            tensor_data=processed_data,
            meta_data=task.meta_data,
            original_frames=task.frames
        )
        self.inference_queue.put(inference_task)
        if self.debug: print(f"[POSE] 検出推論キューに追加: {task.task_id}")

    def _process_inference_task(self, task):
        """プレーヤー検出の推論を実行します。

        Args:
            task (InferenceTask): 処理する推論タスク。
        """
        with torch.no_grad():
            preds = self.predictor.inference_detection(task.tensor_data)
        
        post_task = PostprocessTask(
            task_id=task.task_id,
            inference_output=preds, # ここでは検出結果を渡す
            meta_data=task.meta_data,
            original_frames=task.original_frames
        )
        self.postprocess_queue.put(post_task)
        if self.debug: print(f"[POSE] 後処理キューに追加: {task.task_id}")

    def _process_postprocess_task(self, task):
        """プレーヤー検出結果を後処理し、ポーズ推定を行い、アノテーションを登録します。

        この後処理は、検出とポーズ推定の2段階のパイプラインになっています。

        Args:
            task (PostprocessTask): 処理する後処理タスク。
        """
        # 1. プレーヤー検出の後処理
        det_outputs = task.inference_output
        batch_boxes, batch_scores, batch_valid, images_for_pose = self.predictor.postprocess_detection(
            det_outputs, task.original_frames
        )
        
        if not images_for_pose:
            # 検出されたプレーヤーがいない場合、アノテーションを追加せず終了
            for img_id, _ in task.meta_data:
                self.coco_manager.add_pose_annotations(img_id, [], self.vis_thresh)
            self.processed_counter += len(task.meta_data)
            if self.debug: print(f"[POSE] プレーヤー非検出: {task.task_id}")
            return

        # 2. ポーズ推定のパイプライン
        pose_inputs = self.predictor.preprocess_pose(images_for_pose, batch_boxes)
        with torch.no_grad():
            pose_outputs = self.predictor.inference_pose(pose_inputs)
        
        pose_results = self.predictor.postprocess_pose(
            pose_outputs, batch_boxes, batch_scores, batch_valid, len(task.original_frames)
        )
        
        # 3. アノテーション登録
        for (img_id, _), poses in zip(task.meta_data, pose_results, strict=False):
            self.coco_manager.add_pose_annotations(img_id, poses, self.vis_thresh)

        self.processed_counter += len(task.meta_data)
        if self.debug: print(f"[POSE] 後処理完了: {task.task_id}, {len(task.meta_data)}件")