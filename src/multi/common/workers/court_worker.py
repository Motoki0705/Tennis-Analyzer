# common/workers/court_worker.py

import torch

from .base_worker import BaseWorker
from common.definitions import InferenceTask, PostprocessTask

class CourtWorker(BaseWorker):
    """コート検出のための汎用パイプラインワーカー。"""
    def __init__(self, *args, **kwargs):
        super().__init__("court", *args, **kwargs)

    def _process_preprocess_task(self, task):
        processed_data, _ = self.predictor.preprocess(task.frames)
        self.inference_queue.put(InferenceTask(task.task_id, processed_data, task.meta_data))

    def _process_inference_task(self, task):
        with torch.no_grad():
            preds = self.predictor.inference(task.tensor_data)
        self.postprocess_queue.put(PostprocessTask(task.task_id, preds, task.meta_data))

    def _process_postprocess_task(self, task):
        original_shapes = [(meta[1], meta[2]) for meta in task.meta_data]
        results, _ = self.predictor.postprocess(task.inference_output, original_shapes)
        self.postprocess_handler(results=results, meta_data=task.meta_data)