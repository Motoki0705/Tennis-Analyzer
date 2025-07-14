import logging
from typing import Dict, List
import cv2
import numpy as np
import argparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def draw_results_on_frame(frame: np.ndarray, player_detections: Dict, pose_results: List[Dict], pose_keypoint_threshold: float) -> np.ndarray:
    """フレームにプレイヤーのBBoxと骨格を描画する"""
    # --- Drawing Helpers ---
    SKELETON = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
        [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [0, 5], [0, 6]
    ]
    PLAYER_BBOX_COLOR = (36, 255, 12) # Green
    KEYPOINT_COLOR = (0, 255, 0)  # Green
    SKELETON_COLOR = (255, 128, 0) # Orange
    # 1. プレイヤーのBBoxを描画
    for score, box in zip(player_detections['scores'], player_detections['boxes']):
        box_int = [int(i) for i in box]
        x1, y1, x2, y2 = box_int
        cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_BBOX_COLOR, 2)
        label_text = f"player: {score:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLAYER_BBOX_COLOR, 2)

    # 2. 骨格を描画 (検出された場合)
    if pose_results:
        for person_pose in pose_results:
            keypoints = person_pose['keypoints']
            scores = person_pose['scores']
            for i, (point, score) in enumerate(zip(keypoints, scores)):
                if score > pose_keypoint_threshold:
                    cv2.circle(frame, tuple(map(int, point)), 5, KEYPOINT_COLOR, -1, cv2.LINE_AA)
            
            for joint in SKELETON:
                idx1, idx2 = joint
                if scores[idx1] > pose_keypoint_threshold and scores[idx2] > pose_keypoint_threshold:
                    pt1, pt2 = tuple(map(int, keypoints[idx1])), tuple(map(int, keypoints[idx2]))
                    cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)
    return frame
