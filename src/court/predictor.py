import cv2
import torch
import numpy as np
import albumentations as A
import logging
from typing import List, Tuple, Union, Callable
from pathlib import Path
from tqdm import tqdm
from scipy.special import expit

class CourtPredictor:
    """
    動画フレームに対してコートのキーポイントを推論し、
    ・円だけ重ねる overlay モード
    ・ヒートマップを重ねる heatmap モード
    のいずれかで動画として書き出す。
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        input_size: Tuple[int, int] = (256, 256),
        num_keypoints: int = 15,
        threshold: float = 0.5,
        radius: int = 5,
        kp_color: Tuple[int, int, int] = (0, 255, 0),
        use_half: bool = False,
        visualize_mode: str = "overlay"  # "overlay" | "heatmap" | "heatmap_channels"
    ):
        # ロガー設定
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.device = device
        self.num_keypoints = num_keypoints
        self.threshold = threshold
        self.radius = radius
        self.kp_color = kp_color
        self.use_half = use_half
        self.visualize_mode = visualize_mode

        # モデル
        self.model = model.to(self.device).eval()

        # 変換パイプライン
        self.transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2()
        ])

    def predict(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[List[List[dict]], List[np.ndarray]]:
        """
        B 枚のフレームに対して、
        - keypoints: List[List[{"x","y","confidence"}]]
        - raw_heatmaps: List[np.ndarray]（shape=(1,H,W) か (C,H,W)）
        を返す。
        """
        # 前処理 → バッチテンソル
        tensors = []
        for img in frames:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=rgb)
            tensors.append(aug["image"])
        batch = torch.stack(tensors).to(self.device)

        # 推論
        with torch.no_grad():
            if self.use_half:
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(batch)
            else:
                outputs = self.model(batch)

        # sigmoid → numpy
        heatmaps = expit(outputs.cpu().numpy())

        kps_list, hm_list = [], []
        for hm, frame in zip(heatmaps, frames):
            # hm の shape によって単一 or マルチチャネルを自動判定
            if hm.ndim == 3 and hm.shape[0] == self.num_keypoints:
                keypoints = self._extract_keypoints_multichannel(hm, frame.shape[:2])
            else:
                keypoints = self._extract_keypoints_singlechannel(hm, frame.shape[:2])
            kps_list.append(keypoints)
            hm_list.append(hm)
        return kps_list, hm_list

    def _extract_keypoints_multichannel(
        self,
        heatmaps: np.ndarray,
        orig_shape: Tuple[int, int]
    ) -> List[dict]:
        """
        各チャンネル最大値位置を 1 点ずつ取り出す。
        """
        H, W = orig_shape
        _, h_hm, w_hm = heatmaps.shape
        kps = []
        for c in range(self.num_keypoints):
            hm_c = heatmaps[c]
            y, x = np.unravel_index(np.argmax(hm_c), hm_c.shape)
            conf = float(hm_c[y, x])
            X = int(x * W / w_hm)
            Y = int(y * H / h_hm)
            kps.append({"x": X, "y": Y, "confidence": conf})
        return kps

    def _extract_keypoints_singlechannel(
        self,
        heatmap: np.ndarray,
        orig_shape: Tuple[int, int]
    ) -> List[dict]:
        """
        単一チャネルのヒートマップから局所最大点を NMS で取り出す（従来方式）。
        """
        heatmap = np.squeeze(heatmap)
        h_hm, w_hm = heatmap.shape
        H, W = orig_shape

        # 局所最大
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(heatmap, kernel)
        peaks = np.where((heatmap == dilated) & (heatmap > self.threshold))
        ys, xs = peaks

        # NMS
        scores = heatmap[ys, xs]
        order = np.argsort(scores)[::-1]
        selected = []
        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            if all(abs(y - yy) + abs(x - xx) > 5 for yy, xx in selected):
                selected.append((y, x))

        kps = []
        for y, x in selected:
            X = int(x * W / w_hm)
            Y = int(y * H / h_hm)
            conf = float(heatmap[y, x])
            kps.append({"x": X, "y": Y, "confidence": conf})
        return kps

    def overlay(self, frame: np.ndarray, keypoints: List[dict]) -> np.ndarray:
        """
        円だけを重ねる。
        """
        out = frame.copy()
        for kp in keypoints:
            if kp["confidence"] >= self.threshold:
                cv2.circle(
                    out,
                    (kp["x"], kp["y"]),
                    self.radius,
                    self.kp_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )
        return out

    def overlay_heatmap(self, frame: np.ndarray, hm: np.ndarray) -> np.ndarray:
        """
        ヒートマップカラーを α ブレンドで重ねる。
        マルチチャネル時は最大値合成して可視化。
        """
        if hm.ndim == 3:
            hm_vis = hm.max(axis=0)
        else:
            hm_vis = hm

        if hm_vis.max() > 0:
            hm_uint8 = np.uint8(hm_vis / hm_vis.max() * 255)
            color   = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            color   = cv2.resize(color, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            out = cv2.addWeighted(frame, 0.6, color, 0.4, 0)
        else:
            out = frame.copy()
        return out

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        batch_size: int = 8,
    ) -> None:
        """
        input_path の動画を読み込み、
        ・overlay: 円のみ重ねる
        ・heatmap: 全チャネルを max 合成して重ねる
        ・heatmap_channels: チャネルごとに別動画を出力
        のいずれかで output_path に書き出す。
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error(f"動画を開けません: {input_path}")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 出力ライター準備
        if self.visualize_mode != "heatmap_channels":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        else:
            # チャネル数分のライターを作る
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            stem   = output_path.stem
            parent = output_path.parent
            writers = [
                cv2.VideoWriter(str(parent / f"{stem}_kp{c}.mp4"),
                                fourcc, fps, (width, height))
                for c in range(self.num_keypoints)
            ]

        self.logger.info(
            f"読み込み完了 → フレーム数: {total}, FPS: {fps:.2f}, 解像度: {width}×{height}"
        )

        batch_frames: List[np.ndarray] = []
        with tqdm(total=total, desc="Court 推論") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)

                if len(batch_frames) == batch_size:
                    kps_batch, hm_batch = self.predict(batch_frames)
                    for frm, kps, hm in zip(batch_frames, kps_batch, hm_batch):
                        if self.visualize_mode == "overlay":
                            out = self.overlay(frm, kps)
                            writer.write(out)

                        elif self.visualize_mode == "heatmap":
                            out = self.overlay_heatmap(frm, hm)
                            writer.write(out)

                        else:  # heatmap_channels
                            # 各チャネルごとに 2D ヒートマップを重ねて書き出し
                            for c, w in enumerate(writers):
                                # hm[c] が (h,w) の 2D ヒートマップ
                                out_c = self.overlay_heatmap(frm, hm[c])
                                w.write(out_c)

                        pbar.update(1)
                    batch_frames.clear()

            # 残りフレーム
            if batch_frames:
                kps_batch, hm_batch = self.predict(batch_frames)
                for frm, kps, hm in zip(batch_frames, kps_batch, hm_batch):
                    if self.visualize_mode == "overlay":
                        writer.write(self.overlay(frm, kps))
                    elif self.visualize_mode == "heatmap":
                        writer.write(self.overlay_heatmap(frm, hm))
                    else:
                        for c, w in enumerate(writers):
                            w.write(self.overlay_heatmap(frm, hm[c]))
                    pbar.update(1)

        cap.release()
        if self.visualize_mode != "heatmap_channels":
            writer.release()
            self.logger.info(f"処理完了 → 出力: {output_path}")
        else:
            for w in writers:
                w.release()
            self.logger.info(
                f"処理完了 → チャネルごとの動画を {parent}/ に出力しました"
            )