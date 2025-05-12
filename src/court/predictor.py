import cv2
import torch
import numpy as np
import albumentations as A
import logging
from typing import List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from scipy.special import expit
from PIL import Image

from src.court.models.fpn import CourtDetectorFPN
from src.utils.load_model import load_model_weights


class CourtPredictor:
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",
        input_size: Tuple[int, int] = (256, 256),
        num_keypoints: int = 1,
        threshold: float = 0.5,
        min_distance: int = 10,
        radius: int = 5,
        kp_color: Tuple[int, int, int] = (0, 255, 0),
        use_half: bool = False
    ):
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.device = device
        self.input_size = input_size
        self.threshold = threshold
        self.min_distance = min_distance
        self.radius = radius
        self.kp_color = kp_color
        self.use_half = use_half
        
        # モデルロード
        self.model = self._load_model(model_path, num_keypoints)

        # 変換
        self.transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2()
        ])

    def _load_model(self, model_path: str, num_keypoints: int) -> torch.nn.Module:
        self.logger.info(f"Loading model with num_keypoints={num_keypoints} from {model_path}")
        model = CourtDetectorFPN()
        model = load_model_weights(model, model_path)
        model = model.eval().to(self.device)
        return model

    def predict(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        入力フレーム群に対してコートキーポイント推論を行う。
        returns: List of List of {"x": int, "y": int, "confidence": float}
        """
        tensors = []
        for img in frames:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=rgb)
            tensors.append(aug["image"])
        batch = torch.stack(tensors).to(self.device)  # (B, C, H, W)

        if self.use_half:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                outputs = self.model(batch)
        else:
            with torch.no_grad():
                outputs = self.model(batch)

        if outputs.ndim == 4 and outputs.shape[1] == 1:
            heatmaps = outputs[:, 0]
        else:
            heatmaps = outputs.sum(dim=1)

        heatmaps = expit(heatmaps.cpu().numpy())  # シグモイド正規化

        results = []
        for hm, frame in zip(heatmaps, frames):
            keypoints = self._extract_keypoints(hm, frame.shape[:2])
            results.append(keypoints)

        return results

    def _extract_keypoints(self, heatmap: np.ndarray, orig_shape: Tuple[int, int]) -> List[dict]:
        """
        シグモイド後のヒートマップからキーポイント座標を抽出する。
        returns: [{"x": int, "y": int, "confidence": float}, ...]
        """
        h_heat, w_heat = heatmap.shape
        H, W = orig_shape

        # 局所最大検出
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(heatmap, kernel)
        peaks_mask = (heatmap == dilated) & (heatmap > self.threshold)
        ys, xs = np.where(peaks_mask)

        # NMS
        scores = heatmap[ys, xs]
        order = np.argsort(scores)[::-1]
        selected = []
        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            if all(abs(y - yy) + abs(x - xx) > self.min_distance for yy, xx in selected):
                selected.append((y, x))

        keypoints = []
        for y, x in selected:
            X = int(x * W / w_heat)
            Y = int(y * H / h_heat)
            confidence = float(heatmap[y, x])
            keypoints.append({
                "x": X,
                "y": Y,
                "confidence": confidence
            })

        return keypoints

    def overlay(self, frame: np.ndarray, keypoints: List[dict]) -> np.ndarray:
        """
        入力フレームとキーポイント群を受け取り、描画して返す。
        """
        for kp in keypoints:
            if kp["confidence"] >= self.threshold:
                cv2.circle(
                    frame,
                    (kp["x"], kp["y"]),
                    self.radius,
                    self.kp_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )
        return frame
    
    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        batch_size: int = 8
    ) -> None:
        """
        動画を読み込み、batch_size フレームずつまとめて推論→オーバーレイし、
        出力動画に書き出します。
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error(f"動画ファイルを開けませんでした: {input_path}")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        self.logger.info(
            f"読み込み完了 → フレーム数: {total}, FPS: {fps:.2f}, 解像度: {width}×{height}"
        )

        batch: List[np.ndarray] = []
        with tqdm(total=total, desc="Court 推論処理") as pbar:
            # フレームの読み込み＋バッチ推論ループ
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                batch.append(frame)
                # バッチがたまったらまとめて推論
                if len(batch) == batch_size:
                    kps_batch = self.predict(batch)
                    for frm, kps in zip(batch, kps_batch):
                        overlaid = self.overlay(frm, kps)
                        writer.write(overlaid)
                        pbar.update(1)
                    batch.clear()

            # 残りフレームの処理
            if batch:
                kps_batch = self.predict(batch)
                for frm, kps in zip(batch, kps_batch):
                    overlaid = self.overlay(frm, kps)
                    writer.write(overlaid)
                    pbar.update(1)

        cap.release()
        writer.release()
        self.logger.info(f"処理完了 → 出力ファイル: {output_path}")
