import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import albumentations as A
import albumentations.pytorch
import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from src.utils.logging_utils import setup_logger


class BallPredictor:
    """
    動画からボールの位置を高速に検出するクラス。

    このクラスは、動画を num_frames ごとのクリップに分割し、
    指定されたバッチサイズで一括してモデル推論を実行します。
    これにより、逐次処理に比べて大幅な高速化を実現します。

    モデルは [B, T, C, H, W] の入力を受け取り、
    [B, T, H, W] のヒートマップを出力することを想定しています。
    (B: バッチサイズ, T: num_frames)
    """

    def __init__(
        self,
        litmodule,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        num_frames: int,
        threshold: float = 0.5,
        device: str = "cuda",
        use_half: bool = False,
    ):
        """
        Args:
            litmodule: PyTorch Lightningモデルモジュール
            input_size: モデルへの入力画像サイズ (H, W)
            heatmap_size: モデルが出力するヒートマップのサイズ (H, W)
            num_frames: 1クリップあたりのフレーム数 (T)
            threshold: ボールとして検出する信頼度の閾値
            device: 推論に使用するデバイス ("cuda" or "cpu")
            use_half: 半精度(FP16)推論を使用するか
        """
        self.logger = setup_logger(self.__class__)

        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.num_frames = num_frames
        self.threshold = threshold
        self.device = device
        self.use_half = use_half

        self.model = litmodule.eval().to(self.device)
        if self.use_half:
            self.model.half()

        self.transform = A.Compose(
            [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2(),
            ]
        )

    def _preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """単一クリップ（複数フレーム）を前処理し、テンソルに変換します。"""
        tens_list = []
        for img_bgr in frames:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=img_rgb)
            tens_list.append(aug["image"])
        # フレームを新しい次元(T)でスタックし、[T, C, H, W] 形式にする
        clip_tensor = torch.stack(tens_list, dim=0)
        return clip_tensor

    def preprocess(self, clips: List[List[np.ndarray]]) -> torch.Tensor:
        """クリップのリストをバッチテンソルに変換します。"""
        batch_tensors = [self._preprocess_clip(clip) for clip in clips]
        # クリップをバッチ次元でスタックし、[B, T, C, H, W] 形式にする
        batch = torch.stack(batch_tensors, dim=0).to(self.device)
        if self.use_half:
            return batch.half()
        return batch

    def inference(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """モデル推論を実行します。"""
        with torch.no_grad():
            if self.use_half:
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    preds = self.model(batch_tensor)
            else:
                preds = self.model(batch_tensor)
        return preds

    def postprocess(self, preds: torch.Tensor, clips: List[List[np.ndarray]]) -> List[dict]:
        """
        モデル出力 [B, T, H, W] を後処理し、フラットな予測結果リストを返します。
        """
        all_results = []
        if not (preds.ndim == 4 and preds.shape[1] == self.num_frames):
             raise RuntimeError(f"Unsupported model output shape: {preds.shape}")

        heatmaps = torch.sigmoid(preds.float())  # floatにキャストして安定化

        for clip_heatmaps, clip_frames in zip(heatmaps, clips, strict=True):
            for frame_heatmap, frame in zip(clip_heatmaps, clip_frames, strict=True):
                h_np = frame_heatmap.cpu().numpy()
                confidence = float(h_np.max())
                
                if confidence < self.threshold:
                    all_results.append({"x": None, "y": None, "confidence": confidence})
                    continue

                xh, yh = self._argmax_coord(h_np)
                original_shape = frame.shape[:2]  # (H, W)
                xb, yb = self._to_original_scale(
                    (xh, yh), self.heatmap_size, (original_shape[1], original_shape[0])
                )
                all_results.append({"x": xb, "y": yb, "confidence": confidence})
        
        return all_results

    def predict(self, clips: List[List[np.ndarray]]) -> List[dict]:
        """
        クリップのバッチに対して、前処理、推論、後処理を一貫して実行します。
        """
        batch_tensor = self.preprocess(clips)
        preds = self.inference(batch_tensor)
        results = self.postprocess(preds, clips)
        return results

    def overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """予測結果を単一フレームに描画します。"""
        if result.get("x") is not None:
            cv2.circle(
                frame,
                (result["x"], result["y"]),
                radius=6,
                color=(0, 255, 255),  # Yellow
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        return frame

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
    ) -> None:
        """
        動画ファイルに対して完全なバッチ予測を実行し、結果を出力します。
        """
        # 1. 動画のセットアップ
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_org = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_org = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 2. 全フレームをメモリに読み込む
        self.logger.info(f"Reading all frames from {input_path}...")
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        total_frames = len(all_frames)
        self.logger.info(f"Read {total_frames} frames.")
        if total_frames == 0:
            self.logger.warning("Video has no frames. Exiting.")
            return

        # 3. フレームをクリップに分割
        self.logger.info(f"Dividing frames into clips of size {self.num_frames}...")
        clips = []
        for i in range(0, total_frames, self.num_frames):
            clip = all_frames[i : i + self.num_frames]
            if len(clip) < self.num_frames:
                padding_needed = self.num_frames - len(clip)
                clip.extend([clip[-1]] * padding_needed)
            clips.append(clip)

        # 4. バッチ処理による推論
        self.logger.info(f"Running inference with batch size {batch_size}...")
        all_predictions = []
        for i in tqdm(range(0, len(clips), batch_size), desc="Inference"):
            batch_clips = clips[i : i + batch_size]
            if not batch_clips:
                continue
            predictions = self.predict(batch_clips)
            all_predictions.extend(predictions)

        # 5. 後処理（軌跡の補間など）を一括適用
        self.logger.info("Applying post-processing to the entire track...")
        all_predictions = all_predictions[:total_frames]
        
        if len(all_predictions) >= 2:
            all_predictions = self.remove_jumps(all_predictions)
        if len(all_predictions) >= 3:
            all_predictions = self.interpolate_track(all_predictions)

        # 6. 結果の描画と動画書き込み
        if output_path:
            self.logger.info(f"Overlaying predictions and writing video to {output_path}...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_org, h_org))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open writer: {output_path}")

            for frame, result in tqdm(zip(all_frames, all_predictions), total=total_frames, desc="Writing video"):
                out_frame = self.overlay(frame.copy(), result)
                writer.write(out_frame)
            writer.release()
            
        self.logger.info("Processing finished.")

    # --------------------------------------------------------------------------
    # Helper and Static Methods
    # --------------------------------------------------------------------------

    def _argmax_coord(self, heat: np.ndarray) -> Tuple[int, int]:
        """2次元ヒートマップから最大値の座標を取得します。"""
        h, w = heat.shape
        idx = heat.argmax()
        y, x = idx // w, idx % w
        return int(x), int(y)

    def _to_original_scale(
        self,
        coord: Tuple[int, int],
        from_size: Tuple[int, int],
        to_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """座標を指定されたサイズにスケール変換します。"""
        x, y = coord
        from_w, from_h = from_size
        to_w, to_h = to_size
        x_new = int(x * (to_w / from_w))
        y_new = int(y * (to_h / from_h))
        return x_new, y_new

    @staticmethod
    def remove_jumps(track: List[dict], max_dist: int = 80) -> List[dict]:
        """トラックから急激な座標の変化（ジャンプ）を除去（None化）します。"""
        track_copy = [p.copy() for p in track]
        last_valid_idx = -1
        
        # 最初の有効な点を見つける
        for i, p in enumerate(track_copy):
            if p.get("x") is not None:
                last_valid_idx = i
                break
        
        if last_valid_idx == -1: return track_copy # 有効な点が一つもない

        for i in range(last_valid_idx + 1, len(track_copy)):
            p = track_copy[i]
            if p.get("x") is None:
                continue
            
            last_p = track_copy[last_valid_idx]
            dist = distance.euclidean((last_p["x"], last_p["y"]), (p["x"], p["y"]))
            
            if dist > max_dist:
                p["x"], p["y"], p["confidence"] = None, None, 0.0
            else:
                last_valid_idx = i
                
        return track_copy

    @staticmethod
    def interpolate_track(track: List[dict]) -> List[dict]:
        """None値で不連続なトラックを線形補間します。"""
        if not track or all(p.get("x") is None for p in track):
            return track

        track_copy = [p.copy() for p in track]

        def interp_nan(arr: np.ndarray) -> np.ndarray:
            nans = np.isnan(arr)
            if not np.any(nans) or np.all(nans):
                return arr
            
            x_known = np.flatnonzero(~nans)
            y_known = arr[~nans]
            x_interp = np.flatnonzero(nans)
            
            arr[nans] = np.interp(x_interp, x_known, y_known)
            return arr

        x_vals = np.array([p.get("x") for p in track_copy], dtype=float)
        y_vals = np.array([p.get("y") for p in track_copy], dtype=float)
        
        x_filled = interp_nan(x_vals)
        y_filled = interp_nan(y_vals)

        for i, p in enumerate(track_copy):
            if p.get("x") is None:
                p["x"] = int(round(x_filled[i]))
                p["y"] = int(round(y_filled[i]))
                # 補間された点には信頼度を低い値で設定するなどの工夫も可能
                if "confidence" not in p or p["confidence"] is None:
                    p["confidence"] = 0.1 

        return track_copy