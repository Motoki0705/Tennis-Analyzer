import cv2
import numpy as np
from typing import List, Tuple, Optional

# --- キーポイント描画用の色のリスト (OpenCV BGR形式) ---
KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (75, 25, 230)
]

# --- スケルトン接続情報 ---
# court_reference.py の key_points のインデックスに対応
COURT_SKELETON_EDGES = [
    (0, 1), (2, 3), (0, 2), (1, 3),  # 外枠
    (4, 5), (6, 7),                  # シングルスライン
    (8, 9), (10, 11),                # サービスライン
    (12, 13)                         # センターライン
]

SKELETON_COLOR_BGR = (0, 255, 255)  # 黄色
SKELETON_THICKNESS = 2
POINT_RADIUS = 5

def draw_keypoints_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> None:
    """フレームに検出されたキーポイントを描画する（フレームを直接変更）"""
    for i in range(len(keypoints)):
        if scores[i] > threshold:
            px, py = int(keypoints[i][0]), int(keypoints[i][1])
            color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2.circle(frame, (px, py), POINT_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

def draw_court_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> None:
    """検出されたキーポイントを元にコートのスケルトンを描画する（フレームを直接変更）"""
    for start_idx, end_idx in COURT_SKELETON_EDGES:
        # 接続する両方の点のスコアが閾値を超えているか確認
        if scores[start_idx] > threshold and scores[end_idx] > threshold:
            p1 = keypoints[start_idx]
            p2 = keypoints[end_idx]
            start_point = (int(p1[0]), int(p1[1]))
            end_point = (int(p2[0]), int(p2[1]))
            cv2.line(frame, start_point, end_point, SKELETON_COLOR_BGR, SKELETON_THICKNESS, cv2.LINE_AA)