# multi_flow_annotator/definitions.py

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np

# --- カテゴリ定義 ---

PLAYER_CATEGORY = {
    "id": 2, "name": "player", "supercategory": "person",
    "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ],
    "skeleton": [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6],
    ],
}

COURT_CATEGORY = {
    "id": 3, "name": "court", "supercategory": "field",
    "keypoints": [f"pt{i}" for i in range(15)],
    "skeleton": [],
}

BALL_CATEGORY = {
    "id": 1, "name": "ball", "supercategory": "sports",
    "keypoints": ["center"],
    "skeleton": [],
}

# --- 型エイリアス ---

TaskId = Union[str, int]

# --- タスククラス ---

class PreprocessTask:
    """前処理タスクを表すクラス。

    Attributes:
        task_id (TaskId): タスクの一意なID。
        frames (List[np.ndarray]): 処理対象の画像フレームのリスト。
        meta_data (List[Tuple[int, Path]]): 各フレームのメタデータ（画像ID, 画像パス）のリスト。
        timestamp (float): タスクが作成されたタイムスタンプ。
    """
    def __init__(
        self,
        task_id: TaskId,
        frames: List[np.ndarray],
        meta_data: List[Tuple[int, Path]]
    ):
        self.task_id = task_id
        self.frames = frames
        self.meta_data = meta_data
        self.timestamp = time.time()

class InferenceTask:
    """推論タスクを表すクラス。

    Attributes:
        task_id (TaskId): タスクの一意なID。
        tensor_data (Any): モデルへの入力となる前処理済みテンソルデータ。
        meta_data (List[Tuple[int, Path]]): 各フレームのメタデータ（画像ID, 画像パス）のリスト。
        original_frames (Optional[List[np.ndarray]]): 後処理で必要な場合がある元の画像フレーム。
        timestamp (float): タスクが作成されたタイムスタンプ。
    """
    def __init__(
        self,
        task_id: TaskId,
        tensor_data: Any,
        meta_data: List[Tuple[int, Path]],
        original_frames: Optional[List[np.ndarray]] = None
    ):
        self.task_id = task_id
        self.tensor_data = tensor_data
        self.meta_data = meta_data
        self.original_frames = original_frames
        self.timestamp = time.time()

class PostprocessTask:
    """後処理タスクを表すクラス。

    Attributes:
        task_id (TaskId): タスクの一意なID。
        inference_output (Any): モデルからの推論出力。
        meta_data (List[Tuple[int, Path]]): 各フレームのメタデータ（画像ID, 画像パス）のリスト。
        original_frames (Optional[List[np.ndarray]]): 元の画像フレーム。
        timestamp (float): タスクが作成されたタイムスタンプ。
    """
    def __init__(
        self,
        task_id: TaskId,
        inference_output: Any,
        meta_data: List[Tuple[int, Path]],
        original_frames: Optional[List[np.ndarray]] = None
    ):
        self.task_id = task_id
        self.inference_output = inference_output
        self.meta_data = meta_data
        self.original_frames = original_frames
        self.timestamp = time.time()

# --- カスタム例外 ---

class PredictionTimeoutError(Exception):
    """モデル予測がタイムアウトした場合に発生する例外。"""
    pass