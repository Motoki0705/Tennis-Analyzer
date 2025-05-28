import logging
from pathlib import Path
from typing import List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.ball.models.sequence.mobile_gru_unet import (
    MobileNetUHeatmapNet,
    MobileNetUHeatmapWrapper,
    TemporalHeatmapModel,
)
from src.utils.model_utils import load_model_weights


class BallPredictor:
    """
    Tフレーム入力 → T枚のヒートマップ出力する時系列推論クラス。
    visualize_mode="overlay" / "heatmap" / "features" 対応。
    """

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        base_ch: int = 32,
        num_frames: int = 6,
        threshold: float = 0.5,
        device: str = "cuda",
        visualize_mode: str = "overlay",  # ★ 追加
        feature_layer: int = -1,  # ★ 追加
        use_half: bool = False,  # ★ 追加
    ):
        # ロガー設定
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # パラメータ
        self.H_in, self.W_in = input_size
        self.H_out, self.W_out = heatmap_size
        self.base_ch = base_ch
        self.T = num_frames
        self.thresh = threshold
        self.device = torch.device(device)
        self.visualize_mode = visualize_mode
        self.feature_layer = feature_layer
        self.use_half = use_half

        # モデルロード
        self.model = self._load_model(ckpt_path)
        self.model.eval().to(self.device)

        # 前処理
        self.transform = A.Compose(
            [
                A.Resize(self.H_in, self.W_in),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2(),
            ]
        )

    def _load_model(self, ckpt_path):
        self.logger.info(f"Loading model from {ckpt_path}")
        backbone = MobileNetUHeatmapNet()
        wrapper = MobileNetUHeatmapWrapper(backbone)
        model = TemporalHeatmapModel(wrapper, hidden_dim=self.base_ch * 8)
        model = load_model_weights(model, ckpt_path)
        self.logger.info("  → Model loaded")
        return model

    def _preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        tens_list = []
        for img in frames:
            aug = self.transform(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tens_list.append(aug["image"])
        clip = torch.stack(tens_list, dim=0)  # [T, 3, H, W]
        return clip.unsqueeze(0).to(self.device)  # [1, T, 3, H, W]

    def predict(self, clips: List[List[np.ndarray]]) -> torch.Tensor:
        """
        clips: list of length B, each is list of T BGR frames
        return: heatmaps Tensor shape (B, T, H_out, W_out)
        """
        batch = torch.cat(
            [self._preprocess_clip(c) for c in clips], dim=0
        )  # [B, T, 3, H, W]
        if self.use_half:
            with (
                torch.no_grad(),
                torch.amp.autocast(device_type="cuda", dtype=torch.float16),
            ):
                preds = self.model(batch)  # [B,1,T,h,w]
        else:
            with torch.no_grad():
                preds = self.model(batch)

        heatmaps = torch.sigmoid(preds[:, 0])  # [B, T, h, w]
        return heatmaps

    def _extract_feature_sequence(
        self, batch: torch.Tensor, original_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        batch: [B, T, 3, H, W]
        return: list of (B*T) カラーマップ画像
        """
        if self.use_half:
            with (
                torch.no_grad(),
                torch.amp.autocast(device_type="cuda", dtype=torch.float16),
            ):
                feats = self.model.backbone.encode(batch.view(-1, *batch.shape[2:]))[
                    1
                ]  # skip features
        else:
            with torch.no_grad():
                feats = self.model.backbone.encode(batch.view(-1, *batch.shape[2:]))[1]

        feat_map = feats[self.feature_layer]  # (B*T, C, h, w)
        avg_map = feat_map.mean(dim=1)  # (B*T, h, w)

        seq = []
        arr = avg_map.cpu().numpy()
        for m in arr:
            norm = (m - m.min()) / (m.max() - m.min() + 1e-6) * 255
            img = norm.astype(np.uint8)
            img = cv2.resize(img, original_size, interpolation=cv2.INTER_CUBIC)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            seq.append(img)
        return seq

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        colormap: int = cv2.COLORMAP_JET,
        batch_size: int = 4,  # ★ 追加: バッチサイズ指定
    ) -> None:
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_org = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_org = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (
                (w_org, h_org)
                if self.visualize_mode == "overlay"
                else (self.W_out, self.H_out)
            )
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)

        clip_buffer = []  # 複数のクリップをためる
        frame_buffer = []  # 対応するオリジナルフレームをためる

        buffer = []
        with tqdm(total=total, desc="Ball 推論処理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                buffer.append(frame)
                if len(buffer) == self.T:
                    clip_buffer.append(buffer.copy())
                    frame_buffer.append(buffer.copy())
                    buffer.clear()

                if len(clip_buffer) >= batch_size:
                    self._process_and_write(
                        clip_buffer, frame_buffer, writer, (w_org, h_org), colormap
                    )
                    pbar.update(batch_size * self.T)
                    clip_buffer.clear()
                    frame_buffer.clear()

            # 残り処理
            if buffer:
                while len(buffer) < self.T:
                    buffer.append(buffer[-1].copy())
                clip_buffer.append(buffer.copy())
                frame_buffer.append(buffer.copy())

            if clip_buffer:
                self._process_and_write(
                    clip_buffer, frame_buffer, writer, (w_org, h_org), colormap
                )
                pbar.update(len(clip_buffer) * self.T)

        cap.release()
        if writer:
            writer.release()

    def _process_and_write(self, frames_T, writer, original_size, colormap):
        """
        frames_T: 長さTのBGR画像リスト
        """
        batch = self._preprocess_clip(frames_T)  # [1, T, 3, H, W]

        if self.visualize_mode == "features":
            feature_frames = self._extract_feature_sequence(batch, original_size)
            for img in feature_frames:
                if writer:
                    writer.write(img)

        else:
            heatmaps = self.predict([frames_T])[0]  # [T, h, w]
            for idx, (frame, heat) in enumerate(zip(frames_T, heatmaps.cpu().numpy(), strict=False)):
                if self.visualize_mode == "overlay":
                    flat_ix = heat.argmax()
                    y, x = divmod(flat_ix, heat.shape[1])
                    x0, y0 = self._to_orig(
                        (x, y), (self.W_out, self.H_out), frame.shape[:2][::-1]
                    )
                    if heat.max() >= self.thresh:
                        cv2.circle(frame, (x0, y0), 6, (0, 255, 255), -1, cv2.LINE_AA)
                    out = frame

                else:  # heatmap
                    disp = (heat * 255).astype(np.uint8)
                    disp = cv2.resize(disp, (self.W_out, self.H_out))
                    out = cv2.applyColorMap(disp, colormap)

                if writer:
                    writer.write(out)

    @staticmethod
    def _to_orig(coord, from_size, to_size):
        x, y = coord
        w_f, h_f = from_size
        w_t, h_t = to_size
        return int(x * w_t / w_f), int(y * h_t / h_f)
