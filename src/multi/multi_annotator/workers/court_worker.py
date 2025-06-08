# multi_flow_annotator/workers/court_worker.py

import torch

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

class CourtWorker(BaseWorker):
    """コート検出のためのパイプラインワーカー。"""

    def __init__(self, *args, **kwargs):
        """CourtWorkerのインスタンスを初期化します。"""
        super().__init__("court", *args, **kwargs)

    def _process_preprocess_task(self, task):
        """コート検出の前処理を実行します。

        Args:
            task (PreprocessTask): 処理する前処理タスク。
        """
        processed_data, _ = self.predictor.preprocess(task.frames)
        inference_task = InferenceTask(
            task_id=task.task_id,
            tensor_data=processed_data,
            meta_data=task.meta_data,
            original_frames=task.frames
        )
        self.inference_queue.put(inference_task)
        if self.debug: print(f"[COURT] 推論キューに追加: {task.task_id}")

    def _process_inference_task(self, task):
        """コート検出の推論を実行します。

        Args:
            task (InferenceTask): 処理する推論タスク。
        """
        with torch.no_grad():
            preds = self.predictor.inference(task.tensor_data)
        
        post_task = PostprocessTask(
            task_id=task.task_id,
            inference_output=preds,
            meta_data=task.meta_data,
            original_frames=task.original_frames
        )
        self.postprocess_queue.put(post_task)
        if self.debug: print(f"[COURT] 後処理キューに追加: {task.task_id}")

    def _process_postprocess_task(self, task):
        """コート検出の後処理を行い、アノテーションを登録します。

        Args:
            task (PostprocessTask): 処理する後処理タスク。
        """
        shapes = [frame.shape[:2] for frame in task.original_frames]
        kps_list, _ = self.predictor.postprocess(task.inference_output, shapes)
        
        for (img_id, _), kps in zip(task.meta_data, kps_list, strict=False):
            self.coco_manager.add_court_annotation(img_id, kps, self.vis_thresh)
            
        self.processed_counter += len(kps_list)
        if self.debug: print(f"[COURT] 後処理完了: {task.task_id}, {len(kps_list)}件")