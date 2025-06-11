# streaming_annotator/workers/court_worker.py

import torch

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

class CourtWorker(BaseWorker):
    """コート検出のためのパイプラインワーカー。"""
    
    def __init__(self, name, predictor, queue_set, results_q, debug=False):
        # QueueManagerから基本キューを取得
        preprocess_q = queue_set.get_queue("preprocess")
        inference_q = queue_set.get_queue("inference")
        postprocess_q = queue_set.get_queue("postprocess")
        
        super().__init__(name, predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)

    def _process_preprocess_task(self, task):
        processed_data, _ = self.predictor.preprocess(task.frames)
        self.inference_queue.put(InferenceTask(task.task_id, processed_data, task.meta_data))

    def _process_inference_task(self, task):
        with torch.no_grad():
            preds = self.predictor.inference(task.tensor_data)
        self.postprocess_queue.put(PostprocessTask(task.task_id, preds, task.meta_data))

    def _process_postprocess_task(self, task):
        # メタデータからオリジナルフレームの形状情報を復元
        original_shapes = [(meta[1], meta[2]) for meta in task.meta_data]
        kps_list_batch, _ = self.predictor.postprocess(task.inference_output, original_shapes)
        
        # 結果をフレームごとに分解し、フレームインデックスを付けて結果キューに追加
        for i, kps_list_per_frame in enumerate(kps_list_batch):
            frame_idx = task.meta_data[i][0]
            # (フレームインデックス, "タスク名", 結果) のタプルを格納
            self.results_queue.put((frame_idx, "court", kps_list_per_frame))