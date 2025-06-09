# common/workers/pose_worker.py

import torch

from .base_worker import BaseWorker
from common.definitions import InferenceTask, PostprocessTask

class PoseWorker(BaseWorker):
    """プレーヤー検出とポーズ推定のための汎用パイプラインワーカー。"""
    def __init__(self, *args, **kwargs):
        super().__init__("pose", *args, **kwargs)

    def _process_preprocess_task(self, task):
        processed_data = self.predictor.preprocess_detection(task.frames)
        # 後処理で元のフレームが必要なため、meta_dataに含める
        meta_with_frames = list(zip(task.meta_data, task.frames))
        self.inference_queue.put(InferenceTask(task.task_id, processed_data, meta_with_frames))

    def _process_inference_task(self, task):
        with torch.no_grad():
            preds = self.predictor.inference_detection(task.tensor_data)
        self.postprocess_queue.put(PostprocessTask(task.task_id, preds, task.meta_data))

    def _process_postprocess_task(self, task):
        meta_data, original_frames = zip(*task.meta_data)
        det_outputs = task.inference_output
        batch_boxes, batch_scores, batch_valid, images_for_pose = self.predictor.postprocess_detection(
            det_outputs, list(original_frames)
        )

        if not images_for_pose:
            results = [[] for _ in original_frames]
        else:
            pose_inputs = self.predictor.preprocess_pose(images_for_pose, batch_boxes)
            with torch.no_grad():
                pose_outputs = self.predictor.inference_pose(pose_inputs)
            results = self.predictor.postprocess_pose(
                pose_outputs, batch_boxes, batch_scores, batch_valid, len(original_frames)
            )
        
        self.postprocess_handler(results=results, meta_data=list(meta_data))