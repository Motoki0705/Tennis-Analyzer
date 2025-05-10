import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Union
from tqdm import tqdm
import albumentations as A

from src.ball.models.single_frame.mobilenet import MobileNetUHeatmapNet
from src.utils.load_model import load_model_weights


class BallPredictor:
    def __init__(
        self,
        ckpt_path: Union[str, Path],
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        num_frames: int,
        threshold: float = 0.5,
        device: str = "cuda",
        visualize_mode: str = "overlay",
        feature_layer: int = -1,
        use_half: bool = False  # ★ 追加: 半精度推論を使うか
    ):
        # ────────── logger 初期化 ──────────
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # ────────── パラメータ ──────────
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.num_frames = num_frames
        self.threshold = threshold
        self.device = torch.device(device)
        self.visualize_mode = visualize_mode
        self.feature_layer = feature_layer
        self.use_half = use_half  # ★ 追加

        # ────────── モデルロード ──────────
        self.model = self._load_model(ckpt_path)
        self.model.eval().to(self.device)

        # ────────── 前処理定義 ──────────
        self.transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2()
        ])

    def _load_model(self, ckpt_path: Union[str, Path]) -> torch.nn.Module:
        self.logger.info(f"Loading model from {ckpt_path}")
        model = MobileNetUHeatmapNet()
        model = load_model_weights(model, ckpt_path)
        sample_param = next(model.parameters()).detach().cpu().numpy().ravel()[:5]
        self.logger.info(f"  → sample weights: {sample_param}")
        return model

    def _preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        tens_list = []
        for img_bgr in frames:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=img_rgb)
            tens_list.append(aug["image"])
        clip = torch.cat(tens_list, dim=0)
        return clip.unsqueeze(0).to(self.device)

    def predict(self, clips: List[List[np.ndarray]]) -> List[dict]:
        tensors = []
        for clip in clips:
            assert len(clip) == self.num_frames, f"Each clip must have {self.num_frames} frames"
            tensors.append(self._preprocess_clip(clip))
        batch = torch.cat(tensors, dim=0)

        if self.use_half:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                preds = self.model(batch)
                heatmaps = torch.sigmoid(preds[:, 0])
        else:
            with torch.no_grad():
                preds = self.model(batch)
                heatmaps = torch.sigmoid(preds[:, 0])

        heatmaps = heatmaps.cpu().numpy()
        results = []
        for heat, clip in zip(heatmaps, clips):
            xh, yh = self._argmax_coord(heat)
            xb, yb = self._to_original_scale((xh, yh), self.heatmap_size, clip[-1].shape[:2][::-1])
            conf = float(np.max(heat))
            results.append({"x": xb, "y": yb, "confidence": conf})
        return results

    def _extract_feature_sequence(self, clips: List[List[np.ndarray]], original_size: Tuple[int, int]) -> List[np.ndarray]:
        tensors = [self._preprocess_clip(clip) for clip in clips]
        batch = torch.cat(tensors, dim=0)
        
        if self.use_half:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
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

    def _extract_heatmap_sequence(self, clips: List[List[np.ndarray]], original_size: Tuple[int, int]) -> List[np.ndarray]:
        tensors = [self._preprocess_clip(clip) for clip in clips]
        batch = torch.cat(tensors, dim=0)
        
        if self.use_half:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                preds = self.model(batch)
                heatmaps = torch.sigmoid(preds[:, 0])
        else:
            with torch.no_grad():
                preds = self.model(batch)
                heatmaps = torch.sigmoid(preds[:, 0])

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
            cv2.circle(frame, (result["x"], result["y"]), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
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
            for result, last_frame in zip(results, last_frames):
                out_frame = last_frame.copy()
                if result["confidence"] >= self.threshold:
                    out_frame = self.overlay(out_frame, result)
                if writer:
                    writer.write(out_frame)
                pbar.update(1)

    def run(self, input_path: Union[str, Path], output_path: Union[str, Path] = None, batch_size: int = 4) -> None:
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
                    clip_buffer.append(buffer_frames[-self.num_frames:])
                    last_frames.append(frame)
                if len(clip_buffer) >= batch_size:
                    self._process_and_write(clip_buffer, last_frames, writer, w_org, h_org, pbar)
                    clip_buffer.clear()
                    last_frames.clear()

            if clip_buffer:
                self._process_and_write(clip_buffer, last_frames, writer, w_org, h_org, pbar)

        cap.release()
        if writer:
            writer.release()

        self.logger.info(f"✅ Processing completed. Total frames processed: {pbar.n}")

    def _argmax_coord(self, heat: np.ndarray) -> Tuple[int, int]:
        idx = np.argmax(heat)
        y, x = divmod(idx, heat.shape[1])
        return x, y

    def _to_original_scale(self, coord: Tuple[int, int], from_size: Tuple[int, int], to_size: Tuple[int, int]) -> Tuple[int, int]:
        w_from, h_from = from_size
        w_to, h_to = to_size
        x, y = coord
        return int(x * w_to / w_from), int(y * h_to / h_from)
