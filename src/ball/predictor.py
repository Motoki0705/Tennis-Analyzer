import logging
import queue
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from src.utils.logging_utils import setup_logger


class PreprocessThread(threading.Thread):
    """
    前処理を並列実行するためのスレッドクラス
    """
    def __init__(self, predictor, input_queue, output_queue, clips_metadata_queue):
        super().__init__(daemon=True)
        self.predictor = predictor
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.clips_metadata_queue = clips_metadata_queue
        self.running = True
        self.logger = setup_logger(self.__class__)

    def run(self):
        while self.running:
            try:
                # タイムアウト付きでキューからデータを取得
                clips, metadata = self.input_queue.get(timeout=0.5)
                
                # 前処理実行
                try:
                    batch = self.predictor.preprocess(clips)
                    # 前処理結果と元データのメタ情報を出力キューに送信
                    self.output_queue.put((batch, metadata))
                    # メタデータを別キューにも送信（後処理用）
                    self.clips_metadata_queue.put((clips, metadata))
                except Exception as e:
                    self.logger.error(f"前処理エラー: {e}")
                
                # キュータスク完了を通知
                self.input_queue.task_done()
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                self.logger.error(f"前処理スレッドエラー: {e}")
    
    def stop(self):
        self.running = False


class PostprocessThread(threading.Thread):
    """
    後処理を並列実行するためのスレッドクラス
    """
    def __init__(self, predictor, input_queue, clips_metadata_queue, output_queue):
        super().__init__(daemon=True)
        self.predictor = predictor
        self.input_queue = input_queue  # モデル出力を受け取る
        self.clips_metadata_queue = clips_metadata_queue  # 元画像情報を受け取る
        self.output_queue = output_queue  # 最終結果を出力する
        self.running = True
        self.logger = setup_logger(self.__class__)

    def run(self):
        while self.running:
            try:
                # タイムアウト付きでキューからモデル出力を取得
                preds, metadata = self.input_queue.get(timeout=0.5)
                
                # 元画像情報を取得（順序が一致している必要がある）
                clips, clips_metadata = self.clips_metadata_queue.get(timeout=0.5)
                
                # メタデータの一致を確認
                if metadata != clips_metadata:
                    self.logger.error(f"メタデータの不一致: {metadata} != {clips_metadata}")
                    self.input_queue.task_done()
                    self.clips_metadata_queue.task_done()
                    continue
                
                # 後処理実行
                try:
                    results = self.predictor.postprocess(preds, clips)
                    # 後処理結果を出力キューに送信
                    self.output_queue.put((results, metadata))
                except Exception as e:
                    self.logger.error(f"後処理エラー: {e}")
                
                # キュータスク完了を通知
                self.input_queue.task_done()
                self.clips_metadata_queue.task_done()
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                self.logger.error(f"後処理スレッドエラー: {e}")
    
    def stop(self):
        self.running = False


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
        use_parallel: bool = False,  # 並列処理を使うかどうか
        queue_size: int = 8,  # キューサイズ
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
        self.use_parallel = use_parallel  # 並列処理を使うかどうか
        self.queue_size = queue_size  # キューサイズ

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
        
        # ────────── 並列処理用のキューとスレッド ──────────
        if self.use_parallel:
            self._setup_parallel_processing()
    
    def _setup_parallel_processing(self):
        """並列処理用のキューとスレッドを初期化"""
        # キュー初期化
        self.preprocess_input_queue = queue.Queue(maxsize=self.queue_size)
        self.preprocess_output_queue = queue.Queue(maxsize=self.queue_size)
        self.inference_output_queue = queue.Queue(maxsize=self.queue_size)
        self.clips_metadata_queue = queue.Queue(maxsize=self.queue_size)
        self.final_output_queue = queue.Queue(maxsize=self.queue_size)
        
        # スレッド初期化
        self.preprocess_thread = PreprocessThread(
            self, 
            self.preprocess_input_queue, 
            self.preprocess_output_queue,
            self.clips_metadata_queue
        )
        self.postprocess_thread = PostprocessThread(
            self, 
            self.inference_output_queue, 
            self.clips_metadata_queue,
            self.final_output_queue
        )
        
        # スレッド開始
        self.preprocess_thread.start()
        self.postprocess_thread.start()
        
        self.logger.info("並列処理スレッドを開始しました")
    
    def shutdown_parallel_processing(self):
        """並列処理用のスレッドを終了"""
        if self.use_parallel:
            self.preprocess_thread.stop()
            self.postprocess_thread.stop()
            # スレッドの終了を待機
            self.preprocess_thread.join(timeout=2.0)
            self.postprocess_thread.join(timeout=2.0)
            self.logger.info("並列処理スレッドを終了しました")

    def preprocess(self, clips: List[List[np.ndarray]]) -> torch.Tensor:
        """
        複数のクリップを前処理し、モデル入力用のテンソルを生成します。

        Args:
            clips: 各クリップのリスト。各クリップは複数のフレーム（通常3つ）を含む。

        Returns:
            前処理済みのバッチテンソル
        """
        tensors = [self._preprocess_clip(clip) for clip in clips]
        batch = torch.cat(tensors, dim=0)
        return batch

    def _preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        tens_list = []
        for img_bgr in frames:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=img_rgb)
            tens_list.append(aug["image"])
        clip = torch.cat(tens_list, dim=0)
        return clip.unsqueeze(0).to(self.device)

    def inference(self, batch: torch.Tensor) -> torch.Tensor:
        """
        モデル推論を実行します。

        Args:
            batch: 前処理済みのバッチテンソル

        Returns:
            モデル出力テンソル
        """
        with torch.no_grad():
            if self.use_half:
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    preds = self.model(batch)
            else:
                preds = self.model(batch)
        return preds

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

    def predict(self, clips: List[List[np.ndarray]], metadata=None) -> List[dict]:
        """
        複数のクリップ（各クリップは複数フレーム）を処理し、ボールの位置を予測します。
        前処理、モデル推論、後処理を一連の流れで実行します。

        Args:
            clips: 各クリップのリスト。各クリップは複数のフレーム（通常3つ）を含む。
            metadata: 処理に関連付けるメタデータ（並列処理時に使用）

        Returns:
            クリップごとの予測結果のリスト。各結果は {"x": int, "y": int, "confidence": float} 形式。
        """
        # 並列処理モードの場合
        if self.use_parallel:
            # メタデータが指定されていない場合は生成
            if metadata is None:
                metadata = {"id": id(clips)}
            
            # 前処理キューに入力を追加
            self.preprocess_input_queue.put((clips, metadata))
            
            # 前処理が完了し、バッチが用意できるまで待機
            batch, batch_metadata = self.preprocess_output_queue.get()
            self.preprocess_output_queue.task_done()
            
            # モデル推論実行（この部分はGPUを占有するため、メインスレッドで実行）
            preds = self.inference(batch)
            
            # 後処理キューに推論結果を追加
            self.inference_output_queue.put((preds, batch_metadata))
            
            # 後処理が完了し、結果が用意できるまで待機
            results, results_metadata = self.final_output_queue.get()
            self.final_output_queue.task_done()
            
            return results
        
        # 通常の同期処理モード
        else:
            # 1. 前処理
            batch = self.preprocess(clips)
            
            # 2. モデル推論
            preds = self.inference(batch)
            
            # 3. 後処理
            results = self.postprocess(preds, clips)
            
            return results

    def _extract_feature_sequence(
        self, clips: List[List[np.ndarray]], original_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        # 前処理
        batch = self.preprocess(clips)
        
        # 特徴抽出
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

        # 後処理（可視化用）
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
        # 前処理
        batch = self.preprocess(clips)
        
        # 推論
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

        # 後処理（可視化用）
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
            # 並列処理を使用している場合は終了処理
            if self.use_parallel:
                self.shutdown_parallel_processing()
                
            if writer:
                writer.release()
            cap.release()

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
