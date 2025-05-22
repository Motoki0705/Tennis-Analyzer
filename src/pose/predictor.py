import logging
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# COCO スケルトン定義
COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


class PosePredictor:
    def __init__(
        self,
        det_model: torch.nn.Module,
        det_processor,
        pose_model: torch.nn.Module,
        pose_processor,
        device: Union[str, torch.device] = "cpu",
        player_label_id: int = 0,
        det_score_thresh: float = 0.6,
        pose_score_thresh: float = 0.6,
        use_half: bool = False,
    ):
        from utils.logging_utils import setup_logger
        self.logger = setup_logger(self.__class__)

        self.device = device
        self.det_model = det_model.to(self.device).eval()
        self.det_processor = det_processor
        self.pose_model = pose_model.to(self.device).eval()
        self.pose_processor = pose_processor

        self.player_label_id = player_label_id
        self.det_score_thresh = det_score_thresh
        self.pose_score_thresh = pose_score_thresh
        self.use_half = use_half

    def predict(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        frames: List of BGR np.ndarray
        returns: List (per frame) of List of {"bbox": [x0,y0,w,h], "keypoints": [...], "scores": [...]}
        """
        if not frames:
            return []

        # DETR への入力生成
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        det_inputs = self.det_processor(images=batch_rgb, return_tensors="pt").to(
            self.device
        )

        # DETR 推論
        if self.use_half:
            with (
                torch.no_grad(),
                torch.amp.autocast(device_type=self.device, dtype=torch.float16),
            ):
                det_outputs = self.det_model(pixel_values=det_inputs["pixel_values"])
        else:
            with torch.no_grad():
                det_outputs = self.det_model(pixel_values=det_inputs["pixel_values"])

        # 後処理：バウンディングボックス抽出
        target_sizes = [f.shape[:2] for f in frames]
        det_results = self.det_processor.post_process_object_detection(
            det_outputs, threshold=self.det_score_thresh, target_sizes=target_sizes
        )

        # 各フレームごとの bbox, score, valid index, pose 用画像を蓄積
        batch_boxes = []
        batch_scores = []
        batch_valid = []
        images_for_pose = []

        for idx, (frame, detections) in enumerate(zip(frames, det_results, strict=False)):
            frame_boxes = []
            frame_scores = []  # ← ここを追加

            for score, label, box in zip(
                detections["scores"], detections["labels"], detections["boxes"], strict=False
            ):
                if label.item() == self.player_label_id:
                    x0, y0, x1, y1 = box.int().tolist()
                    w, h = x1 - x0, y1 - y0
                    frame_boxes.append([x0, y0, w, h])
                    frame_scores.append(float(score))  # ← ここを変更

            if frame_boxes:
                batch_boxes.append(frame_boxes)
                batch_scores.append(frame_scores)  # ← ここを追加
                batch_valid.append(idx)
                images_for_pose.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )

        # プレーヤー検出が一件もない場合は全フレーム空リスト
        if not images_for_pose:
            return [[] for _ in frames]

        # ViTPose への入力生成
        pose_inputs = self.pose_processor(
            images=images_for_pose, boxes=batch_boxes, return_tensors="pt"
        ).to(self.device)

        # ViTPose 推論
        with torch.no_grad():
            pose_outputs = self.pose_model(**pose_inputs)

        # 後処理：キーポイント抽出
        pose_results = self.pose_processor.post_process_pose_estimation(
            pose_outputs, boxes=batch_boxes
        )

        # 出力形式に整形
        batch_result: List[List[dict]] = [[] for _ in frames]
        for idx, poses, boxes, det_scores in zip(
            batch_valid, pose_results, batch_boxes, batch_scores, strict=False
        ):
            frame_objs = []
            for pose, bbox, det_score in zip(poses, boxes, det_scores, strict=False):
                keypoints = pose["keypoints"].cpu().numpy()
                scores = pose["scores"].cpu().numpy()
                frame_objs.append(
                    {
                        "bbox": bbox,  # [x0, y0, w, h]
                        "det_score": det_score,
                        "keypoints": [
                            (int(x), int(y)) for x, y in keypoints.astype(int)
                        ],
                        "scores": [float(s) for s in scores],
                    }
                )
            batch_result[idx] = frame_objs

        return batch_result

    def overlay(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        frame: BGR np.ndarray
        detections: predict() の 1 フレーム分の結果
        returns: オーバーレイ画像
        """
        annotated = frame.copy()

        for det in detections:
            x0, y0, w, h = det["bbox"]
            x1, y1 = x0 + w, y0 + h

            # バウンディングボックス
            cv2.rectangle(
                annotated, (x0, y0), (x1, y1), (0, 255, 0), 2, lineType=cv2.LINE_AA
            )
            # 検出スコア
            cv2.putText(
                annotated,
                f"{det['det_score']:.2f}",
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # キーポイント描画
            kps = np.array(det["keypoints"])
            scores = np.array(det["scores"])
            for (x, y), s in zip(kps, scores, strict=False):
                if s >= self.pose_score_thresh:
                    cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
            # スケルトン描画
            for i, j in COCO_SKELETON:
                if (
                    i < len(kps)
                    and j < len(kps)
                    and scores[i] >= self.pose_score_thresh
                    and scores[j] >= self.pose_score_thresh
                ):
                    cv2.line(
                        annotated,
                        tuple(kps[i]),
                        tuple(kps[j]),
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        return annotated

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        batch_size: int = 8,
    ) -> None:
        """
        input_path の動画を読み込み、batch_size フレームずつまとめて
        predict → overlay → 動画出力
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error(f"動画を開けませんでした: {input_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"読み込み完了 → フレーム数: {total}, FPS: {fps:.2f}, 解像度: {width}×{height}"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        batch_frames: List[np.ndarray] = []
        with tqdm(total=total, desc="Pose 推論処理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                if len(batch_frames) == batch_size:
                    poses_batch = self.predict(batch_frames)
                    for frm, poses in zip(batch_frames, poses_batch, strict=False):
                        writer.write(self.overlay(frm, poses))
                        pbar.update(1)
                    batch_frames.clear()

            # 残りフレーム
            if batch_frames:
                poses_batch = self.predict(batch_frames)
                for frm, poses in zip(batch_frames, poses_batch, strict=False):
                    writer.write(self.overlay(frm, poses))
                    pbar.update(1)

        cap.release()
        writer.release()
        self.logger.info(f"処理完了 → 出力動画: {output_path}")
