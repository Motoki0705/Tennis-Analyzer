import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from src.utils.logging_utils import setup_logger


class BallPredictor:
    def __init__(
        self,
        model: Callable,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        num_frames: int,
        threshold: float = 0.6,
        device: str = "cuda",
        visualize_mode: str = "overlay",
        feature_layer: int = -1,
        use_half: bool = False,  # ★ 追加: 半精度推論を使うか
    ):
        # ────────── logger 初期化 ──────────
        self.logger = setup_logger(self.__class__)

        # ────────── パラメータ ──────────
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.num_frames = num_frames
        self.threshold = threshold
        self.device = device
        self.visualize_mode = visualize_mode
        self.feature_layer = feature_layer
        self.use_half = use_half  # ★ 追加

        # ────────── モデル ──────────
        self.model = model.eval()

        # ────────── 前処理定義 ──────────
        self.transform = A.Compose(
            [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2(),
            ]
        )

    def _preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        tens_list = []
        for img_bgr in frames:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=img_rgb)
            tens_list.append(aug["image"])
        clip = torch.cat(tens_list, dim=0)
        return clip.unsqueeze(0).to(self.device)

    def predict(self, clips: List[List[np.ndarray]]) -> List[dict]:
        """
        複数のクリップ（各クリップは複数フレーム）を処理し、ボールの位置を予測します。

        Args:
            clips: 各クリップのリスト。各クリップは複数のフレーム（通常3つ）を含む。

        Returns:
            クリップごとの予測結果のリスト。各結果は {"x": int, "y": int, "confidence": float} 形式。
        """
        tensors = [self._preprocess_clip(clip) for clip in clips]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            if self.use_half:
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    preds = self.model(batch)
            else:
                preds = self.model(batch)

        # --- ヒートマップモード ---
        results = []
        
        # 2次元の場合 (B, H, W) - ヒートマップ
        if preds.ndim == 3:
            heatmaps = torch.sigmoid(preds)
            for h, clip in zip(heatmaps, clips, strict=False):
                h_np = h.cpu().numpy()
                xh, yh = self._argmax_coord(h_np)
                xb, yb = self._to_original_scale(
                    (xh, yh), self.heatmap_size, clip[-1].shape[:2]
                )
                conf = float(h_np.max())
                results.append({"x": xb, "y": yb, "confidence": conf})
                
        # 3次元の場合 (B, C, H, W) - ヒートマップ
        elif preds.ndim == 4 and preds.shape[1] == 1:
            heatmaps = torch.sigmoid(preds)
            for h, clip in zip(heatmaps, clips, strict=False):
                h_np = h.squeeze(0).cpu().numpy()
                xh, yh = self._argmax_coord(h_np)
                xb, yb = self._to_original_scale(
                    (xh, yh), self.heatmap_size, clip[-1].shape[:2]
                )
                conf = float(h_np.max())
                results.append({"x": xb, "y": yb, "confidence": conf})

        # --- 座標回帰モード ---
        elif preds.ndim == 2 and preds.shape[1] == 2:
            coords = preds.cpu().numpy()  # [B,2], normalized in [0,1]
            for (x_norm, y_norm), clip in zip(coords, clips, strict=False):
                h_org, w_org = clip[-1].shape[:2]
                xb = int(x_norm * w_org)
                yb = int(y_norm * h_org)
                # 回帰には confidence が無いので None に
                results.append({"x": xb, "y": yb, "confidence": None})

        else:
            raise RuntimeError(f"Unsupported model output shape: {preds.shape}")

        # 外れ値除去 & 欠損補完は heatmap/regression 共通
        if len(results) >= 2:
            results = self.remove_jumps(results)
        if len(results) >= 3:
            results = self.interpolate_track(results)

        return results

    def _extract_feature_sequence(
        self, clips: List[List[np.ndarray]], original_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        tensors = [self._preprocess_clip(clip) for clip in clips]
        batch = torch.cat(tensors, dim=0)

        if self.use_half:
            with (
                torch.no_grad(),
                torch.amp.autocast(device_type=self.device, dtype=torch.float16),
            ):
                feats_nhwc = self.model.backbone(batch)
        else:
            with torch.no_grad():
                feats_nhwc = self.model.backbone(batch)

        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats_nhwc]
        lowfeat = feats[self.feature_layer]
        avg = lowfeat.mean(dim=1)
        arr = avg.cpu().numpy()

        seq = []
        for m in arr:
            norm = (m - m.min()) / (m.max() - m.min() + 1e-6) * 255
            img = norm.astype(np.uint8)
            img = cv2.resize(img, original_size, interpolation=cv2.INTER_CUBIC)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            seq.append(img)
        return seq

    def _extract_heatmap_sequence(
        self, clips: List[List[np.ndarray]], original_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        tensors = [self._preprocess_clip(clip) for clip in clips]
        batch = torch.cat(tensors, dim=0)

        if self.use_half:
            with (
                torch.no_grad(),
                torch.amp.autocast(device_type=self.device, dtype=torch.float16),
            ):
                preds = self.model(batch)
                heatmaps = torch.sigmoid(preds)
        else:
            with torch.no_grad():
                preds = self.model(batch)
                heatmaps = torch.sigmoid(preds)

        arr = heatmaps.cpu().numpy()

        seq = []
        for m in arr:
            norm = (m * 255).astype(np.uint8)
            img = cv2.resize(norm, original_size, interpolation=cv2.INTER_CUBIC)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            seq.append(img)
        return seq

    def overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        if result["x"] is not None and result["confidence"] > 0.5:
            cv2.circle(
                frame,
                (result["x"], result["y"]),
                6,
                (0, 255, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
        return frame

    def _process_and_write(self, clip_buffer, last_frames, writer, w_org, h_org, pbar):
        if self.visualize_mode == "features":
            feature_frames = self._extract_feature_sequence(clip_buffer, (w_org, h_org))
            for vf in feature_frames:
                if writer:
                    writer.write(vf)
                pbar.update(1)
        elif self.visualize_mode == "heatmap":
            heatmap_frames = self._extract_heatmap_sequence(clip_buffer, (w_org, h_org))
            for hm in heatmap_frames:
                if writer:
                    writer.write(hm)
                pbar.update(1)
        else:
            results = self.predict(clip_buffer)
            for result, last_frame in zip(results, last_frames, strict=False):
                out_frame = last_frame.copy()
                if result["confidence"] >= self.threshold:
                    out_frame = self.overlay(out_frame, result)
                if writer:
                    writer.write(out_frame)
                pbar.update(1)

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
    ) -> None:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_org = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_org = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_org, h_org))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open writer: {output_path}")

        buffer_frames: List[np.ndarray] = []
        clip_buffer: List[List[np.ndarray]] = []
        last_frames: List[np.ndarray] = []

        with tqdm(total=total, desc="Ball 推論処理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                buffer_frames.append(frame)
                if len(buffer_frames) >= self.num_frames:
                    clip_buffer.append(buffer_frames[-self.num_frames :])
                    last_frames.append(frame)
                if len(clip_buffer) >= batch_size:
                    self._process_and_write(
                        clip_buffer, last_frames, writer, w_org, h_org, pbar
                    )
                    clip_buffer.clear()
                    last_frames.clear()

            if clip_buffer:
                self._process_and_write(
                    clip_buffer, last_frames, writer, w_org, h_org, pbar
                )

        cap.release()
        if writer:
            writer.release()

        self.logger.info(f"✅ Processing completed. Total frames processed: {pbar.n}")

    def _argmax_coord(self, heat: np.ndarray) -> Tuple[int, int]:
        idx = np.argmax(heat)
        y, x = divmod(idx, heat.shape[1])
        return (int(x), int(y))

    def _to_original_scale(
        self,
        coord: Tuple[int, int],
        from_size: Tuple[int, int],
        to_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        w_from, h_from = from_size
        h_to, w_to = to_size
        x, y = coord
        return int(x * w_to / w_from), int(y * h_to / h_from)

    @staticmethod
    def remove_jumps(track, max_dist=80):
        cleaned = [track[0]]
        for prev, curr in zip(track, track[1:], strict=False):
            if prev["x"] is not None and curr["x"] is not None:
                d = distance.euclidean((prev["x"], prev["y"]), (curr["x"], curr["y"]))
                if d > max_dist:
                    curr = {"x": None, "y": None, "confidence": 0.0}
            cleaned.append(curr)
        return cleaned

    @staticmethod
    def interpolate_track(track):
        xs = np.array([p["x"] if p["x"] is not None else np.nan for p in track])
        ys = np.array([p["y"] if p["y"] is not None else np.nan for p in track])
        confs = np.array([p["confidence"] or 0.0 for p in track])

        def interp_nan(arr):
            nans = np.isnan(arr)
            if nans.all():
                return arr
            arr[nans] = np.interp(
                np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans]
            )
            return arr

        xs = interp_nan(xs)
        ys = interp_nan(ys)

        result = []
        for x, y, c in zip(xs, ys, confs, strict=False):
            if np.isnan(x) or np.isnan(y):
                result.append({"x": None, "y": None, "confidence": 0.0})
            else:
                result.append(
                    {"x": int(round(x)), "y": int(round(y)), "confidence": float(c)}
                )
        return result
