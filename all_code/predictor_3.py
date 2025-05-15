import cv2
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Union
from pathlib import Path
from tqdm import tqdm


class PlayerPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        label_map: Dict[int, str],
        device: Union[str, torch.device] = "cpu",
        threshold: float = 0.6,
        use_half: bool = False
    ):
        # ─── logger 初期化 ───
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

        # ─── モデル・設定 ───
        self.device = device
        self.model = self._prepare_model(model)
        self.processor = processor
        self.label_map = label_map
        self.threshold = threshold
        self.use_half = use_half

    def _prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """モデルをデバイスに送って eval モードに設定"""
        self.logger.info("Preparing model (to device, eval mode)")
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        バッチ推論。フレーム群をまとめてモデルに投げ、
        各フレームごとの検出結果リストを返す。

        Parameters
        ----------
        frames : List[np.ndarray]
            BGR フレームのリスト

        Returns
        -------
        detections : List[List[dict]]
            各フレームに対する検出結果のリスト。
            各 dict は {"bbox": [x0,y0,x1,y1], "score": float, "label": str}。
        """
        if not frames:
            return []

        # RGB 変換
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        # Processor によるエンコード
        inputs = self.processor(images=batch_rgb, return_tensors="pt").to(self.device)

        # 推論
        if self.use_half:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                outputs = self.model(pixel_values=inputs["pixel_values"])
        else:
            with torch.no_grad():
                outputs = self.model(pixel_values=inputs["pixel_values"])
                
        # 後処理
        target_sizes = [f.shape[:2] for f in frames]  # (height, width)
        results_batch = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )

        # フォーマット変換
        all_detections: List[List[dict]] = []
        for results in results_batch:
            frame_dets: List[dict] = []
            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                x0, y0, x1, y1 = box.int().tolist()
                frame_dets.append({
                    "bbox": [x0, y0, x1, y1],
                    "score": float(score),
                    "label": self.label_map.get(int(label_id), str(label_id))
                })
            all_detections.append(frame_dets)

        return all_detections

    def overlay(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        検出結果をフレームにオーバーレイ描画して返す。

        Parameters
        ----------
        frame : np.ndarray
            BGR 画像
        detections : List[dict]
            predict() の出力 (単一フレーム分)

        Returns
        -------
        annotated : np.ndarray
            描画済み BGR 画像
        """
        annotated = frame.copy()
        for det in detections:
            x0, y0, x1, y1 = det["bbox"]
            score = det["score"]
            label = det["label"]
            # バウンディングボックス
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # ラベルとスコア
            text = f"{label}: {score:.2f}"
            cv2.putText(
                annotated, text,
                (x0, max(y0 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, lineType=cv2.LINE_AA
            )
        return annotated

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

        self.logger.info(
            f"読み込み完了 → フレーム数: {total}, FPS: {fps:.2f}, 解像度: {width}×{height}"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        batch: List[np.ndarray] = []
        with tqdm(total=total, desc="Player 推論処理") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                batch.append(frame)
                if len(batch) == batch_size:
                    dets_batch = self.predict(batch)
                    for frm, dets in zip(batch, dets_batch):
                        annotated = self.overlay(frm, dets)
                        writer.write(annotated)
                        pbar.update(1)
                    batch.clear()

            if batch:
                dets_batch = self.predict(batch)
                for frm, dets in zip(batch, dets_batch):
                    annotated = self.overlay(frm, dets)
                    writer.write(annotated)
                    pbar.update(1)

        cap.release()
        writer.release()
        self.logger.info(f"処理完了 → 出力動画: {output_path}")