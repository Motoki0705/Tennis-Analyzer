# common/definitions.py

import time
from typing import Any, List, Tuple, Union

import numpy as np

TaskId = Union[str, int]

class PreprocessTask:
    """前処理タスクを表すクラス。"""
    def __init__(self, task_id: TaskId, frames: List[np.ndarray], meta_data: List[Any]):
        self.task_id = task_id
        self.frames = frames
        self.meta_data = meta_data
        self.timestamp = time.time()

class InferenceTask:
    """推論タスクを表すクラス。"""
    def __init__(self, task_id: TaskId, tensor_data: Any, meta_data: List[Any]):
        self.task_id = task_id
        self.tensor_data = tensor_data
        self.meta_data = meta_data
        self.timestamp = time.time()

class PostprocessTask:
    """後処理タスクを表すクラス。"""
    def __init__(self, task_id: TaskId, inference_output: Any, meta_data: List[Any]):
        self.task_id = task_id
        self.inference_output = inference_output
        self.meta_data = meta_data
        self.timestamp = time.time()