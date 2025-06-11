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

    def preprocess(self, clips: List[List[np.ndarray]]) -> torch.Tensor:
        """
        StreamingOverlayer からも呼び出されるため、
        引数 clips が *フレームリスト* の場合（= 1 ステップ API 用）と
        *クリップのリスト* の場合の両方を許容する。

        1. clips が ``List[np.ndarray]`` の場合は Streaming 用として扱い、
           そのまま返して後段で処理する。（戻り値は `(data, meta)` のタプル）
        2. clips が ``List[List[np.ndarray]]`` の場合は従来どおりバッチテンソルを返す。
        """
        # StreamingOverlayer ではフレーム単位で渡される
        if clips and isinstance(clips[0], np.ndarray):  # type: ignore[arg-type]
            return clips, None  # type: ignore[return-value]

        # 従来のクリップ単位
        tensors = [self._preprocess_clip(clip) for clip in clips]  # type: ignore[arg-type]
        batch = torch.cat(tensors, dim=0)
        return batch  # type: ignore[return-value]

    def _preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        単一クリップ（複数フレーム）を前処理し、テンソルに変換します。

        Args:
            frames: フレームのリスト

        Returns:
            前処理済みのテンソル
        """
        tens_list = []
        for img_bgr in frames:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=img_rgb)
            tens_list.append(aug["image"])
        clip = torch.cat(tens_list, dim=0)
        return clip.unsqueeze(0).to(self.device)

    def inference(self, tensor_data):  # type: ignore[override]
        """StreamingOverlayer 互換: frames リスト / テンソル双方に対応"""

        # 1) StreamingOverlayer からはフレームリストが渡される
        if isinstance(tensor_data, list):
            frames: List[np.ndarray] = tensor_data  # type: ignore[assignment]
            clips: List[List[np.ndarray]] = []

            # フレームを num_frames ごとに分割
            for i in range(0, len(frames) - self.num_frames + 1, self.num_frames):
                clip = frames[i : i + self.num_frames]
                if len(clip) == self.num_frames:
                    clips.append(clip)

            # フレーム数が足りない場合は最後のフレームでパディングして 1 クリップ生成
            if not clips and frames:
                padded = frames + [frames[-1]] * (self.num_frames - len(frames))
                clips.append(padded)

            if not clips:
                return []  # type: ignore[return-value]

            return self.predict(clips)  # type: ignore[return-value]

        # 2) 従来どおりテンソルが渡された場合は元のロジック
        if isinstance(tensor_data, torch.Tensor):
            with torch.no_grad():
                if self.use_half:
                    with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                        preds = self.model(tensor_data)
                else:
                    preds = self.model(tensor_data)
            return preds  # type: ignore[return-value]

        raise TypeError("Unsupported tensor_data type for inference")

    def postprocess(self, preds: torch.Tensor, clips: List[List[np.ndarray]]) -> List[dict]:
        """
        モデル出力を後処理し、予測結果をフォーマットします。

        Args:
            preds: モデル出力テンソル
            clips: 元の入力クリップ（元の解像度へのマッピングに使用）

        Returns:
            クリップごとの予測結果のリスト。各結果は {"x": int, "y": int, "confidence": float} 形式。
        """
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

    def predict(self, clips: List[List[np.ndarray]]) -> List[dict]:
        """
        複数のクリップ（各クリップは複数フレーム）を処理し、ボールの位置を予測します。
        前処理、モデル推論、後処理を一連の流れで実行します。

        Args:
            clips: 各クリップのリスト。各クリップは複数のフレーム（通常3つ）を含む。

        Returns:
            クリップごとの予測結果のリスト。各結果は {"x": int, "y": int, "confidence": float} 形式。
        """
        # 1. 前処理
        batch = self.preprocess(clips)
        
        # 2. モデル推論
        preds = self.inference(batch)
        
        # 3. 後処理
        results = self.postprocess(preds, clips)
        
        return results

    def extract_features(self, clips: List[List[np.ndarray]]) -> torch.Tensor:
        """
        特徴抽出を行います。可視化などに使用します。

        Args:
            clips: 各クリップのリスト

        Returns:
            抽出された特徴テンソル
        """
        # 前処理
        batch = self.preprocess(clips)
        
        # 特徴抽出
        with torch.no_grad():
            if self.use_half:
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    feats_nhwc = self.model.backbone(batch)
            else:
                feats_nhwc = self.model.backbone(batch)
        
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats_nhwc]
        return feats

    def visualize_features(self, feats: List[torch.Tensor], original_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        特徴マップを可視化します。

        Args:
            feats: 特徴テンソルのリスト
            original_size: 元の画像サイズ

        Returns:
            可視化された特徴マップのリスト
        """
        lowfeat = feats[self.feature_layer]
        avg = lowfeat.mean(dim=1)
        arr = avg.cpu().numpy()

        # 可視化処理
        seq = []
        for m in arr:
            norm = (m - m.min()) / (m.max() - m.min() + 1e-6) * 255
            img = norm.astype(np.uint8)
            img = cv2.resize(img, original_size, interpolation=cv2.INTER_CUBIC)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            seq.append(img)
        return seq

    def visualize_heatmaps(self, preds: torch.Tensor, original_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        ヒートマップを可視化します。

        Args:
            preds: モデル出力テンソル
            original_size: 元の画像サイズ

        Returns:
            可視化されたヒートマップのリスト
        """
        heatmaps = torch.sigmoid(preds)
        arr = heatmaps.cpu().numpy()

        # 可視化処理
        seq = []
        for m in arr:
            norm = (m * 255).astype(np.uint8)
            img = cv2.resize(norm, original_size, interpolation=cv2.INTER_CUBIC)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            seq.append(img)
        return seq

    def overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        予測結果を画像に重ね合わせます。

        Args:
            frame: 元の画像
            result: 予測結果

        Returns:
            重ね合わせ後の画像
        """
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

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
    ) -> None:
        """
        動画ファイルに対して予測を実行し、結果を出力します。

        Args:
            input_path: 入力動画のパス
            output_path: 出力動画のパス
            batch_size: バッチサイズ
        """
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

        try:
            with tqdm(total=total, desc="Ball 推論処理") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    buffer_frames.append(frame)

                    if len(buffer_frames) >= self.num_frames:
                        clip = buffer_frames[-self.num_frames :]
                        clip_buffer.append(clip)
                        last_frames.append(frame)

                        if len(clip_buffer) >= batch_size:
                            self._process_and_write(
                                clip_buffer, last_frames, writer, w_org, h_org, pbar
                            )
                            clip_buffer = []
                            last_frames = []

                # 残りの clip があれば処理
                if clip_buffer:
                    self._process_and_write(
                        clip_buffer, last_frames, writer, w_org, h_org, pbar
                    )
        finally:
            if writer:
                writer.release()
            cap.release()
            
    def _process_and_write(self, clip_buffer, last_frames, writer, w_org, h_org, pbar):
        """
        バッチ処理して結果を書き出します。
        
        Args:
            clip_buffer: クリップのバッファ
            last_frames: 最終フレームのリスト
            writer: 動画書き込みオブジェクト
            w_org: 元の幅
            h_org: 元の高さ
            pbar: 進捗バー
        """
        if self.visualize_mode == "features":
            # 特徴抽出モード
            feats = self.extract_features(clip_buffer)
            feature_frames = self.visualize_features(feats, (w_org, h_org))
            for vf in feature_frames:
                if writer:
                    writer.write(vf)
                pbar.update(1)
        elif self.visualize_mode == "heatmap":
            # ヒートマップモード
            batch = self.preprocess(clip_buffer)
            preds = self.inference(batch)
            heatmap_frames = self.visualize_heatmaps(preds, (w_org, h_org))
            for hm in heatmap_frames:
                if writer:
                    writer.write(hm)
                pbar.update(1)
        else:
            # 通常の予測モード
            results = self.predict(clip_buffer)
            for result, last_frame in zip(results, last_frames, strict=False):
                out_frame = last_frame.copy()
                if result["confidence"] >= self.threshold:
                    out_frame = self.overlay(out_frame, result)
                if writer:
                    writer.write(out_frame)
                pbar.update(1)

    def _argmax_coord(self, heat: np.ndarray) -> Tuple[int, int]:
        """2次元のヒートマップから最も確率が高い座標を取得"""
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
        """座標をヒートマップサイズから元画像サイズにスケール変換"""
        x, y = coord
        from_w, from_h = from_size
        to_w, to_h = to_size
        x_new = int(x * (to_w / from_w))
        y_new = int(y * (to_h / from_h))
        return x_new, y_new

    @staticmethod
    def remove_jumps(track, max_dist=80):
        """トラックから大きなジャンプを検出して除去"""
        track_copy = track.copy()
        if len(track) <= 1:
            return track_copy
        last_valid = None
        for i, p in enumerate(track_copy):
            if p["x"] is None or p["confidence"] is None:
                continue
            if last_valid is not None:
                dist = distance.euclidean((track_copy[last_valid]["x"], track_copy[last_valid]["y"]), (p["x"], p["y"]))
                if dist > max_dist:
                    p["x"], p["y"], p["confidence"] = None, None, 0.0
                else:
                    last_valid = i
            else:
                last_valid = i
        return track_copy

    @staticmethod
    def interpolate_track(track):
        """None 値のある不連続なトラックを線形補間"""
        if all(p["x"] is None for p in track):
            return track

        def interp_nan(arr):
            nans = np.isnan(arr)
            if all(nans):
                return arr
            if any(nans):
                arr[nans] = np.interp(
                    np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans]
                )
            return arr

        x_vals = np.array([p["x"] if p["x"] is not None else np.nan for p in track])
        y_vals = np.array([p["y"] if p["y"] is not None else np.nan for p in track])
        conf = np.array(
            [p["confidence"] if p["confidence"] is not None else np.nan for p in track]
        )

        x_filled = interp_nan(x_vals)
        y_filled = interp_nan(y_vals)
        conf_filled = interp_nan(conf)

        for i in range(len(track)):
            if track[i]["x"] is None:
                track[i]["x"] = int(round(x_filled[i]))
                track[i]["y"] = int(round(y_filled[i]))
                track[i]["confidence"] = float(conf_filled[i])

        return track
