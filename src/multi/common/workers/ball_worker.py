# common/workers/ball_worker.py

from typing import List
import numpy as np
import torch

from .base_worker import BaseWorker
from common.definitions import InferenceTask, PostprocessTask

class BallWorker(BaseWorker):
    """ボール検出のための汎用パイプラインワーカー。"""
    def __init__(self, *args, **kwargs):
        super().__init__("ball", *args, **kwargs)
        self.sliding_window: List[np.ndarray] = []

    def reset_state(self):
        """新しいクリップ処理のために内部状態をリセットします。"""
        self.sliding_window.clear()

    def _process_preprocess_task(self, task):
        clips, meta = [], []
        for i, frame in enumerate(task.frames):
            self.sliding_window.append(frame.copy())
            if len(self.sliding_window) > self.predictor.num_frames:
                self.sliding_window.pop(0)

            if len(self.sliding_window) == self.predictor.num_frames:
                clips.append(list(self.sliding_window))
                meta.append(task.meta_data[i])

        if clips:
            processed_data = self.predictor.preprocess(clips)
            self.inference_queue.put(InferenceTask(task.task_id, processed_data, meta))

    def _process_inference_task(self, task):
        with torch.no_grad():
            preds = self.predictor.inference(task.tensor_data)
        self.postprocess_queue.put(PostprocessTask(task.task_id, preds, task.meta_data))

    def _process_postprocess_task(self, task):
        results = self.predictor.postprocess(task.inference_output)
        self.postprocess_handler(results=results, meta_data=task.meta_data)