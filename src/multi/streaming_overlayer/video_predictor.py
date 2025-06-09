# streaming_annotator/video_predictor.py

import queue
import time
from pathlib import Path
from typing import Dict, List, Union, Any

import cv2
import numpy as np
from tqdm import tqdm

from .definitions import PreprocessTask
from .video_utils import FrameLoader
from .workers.base_worker import BaseWorker
from .workers.ball_worker import BallWorker
from .workers.court_worker import CourtWorker
from .workers.pose_worker import PoseWorker

class VideoPredictor:
    """動画に対して複数のモデル推論を並列実行し、結果を描画するクラス。"""

    def __init__(
        self,
        ball_predictor, court_predictor, pose_predictor,
        intervals: Dict[str, int], batch_sizes: Dict[str, int],
        debug: bool = False
    ):
        self.predictors = {
            "ball": ball_predictor,
            "court": court_predictor,
            "pose": pose_predictor,
        }
        self.intervals = intervals
        self.batch_sizes = batch_sizes
        self.debug = debug

        # 結果をフレームインデックス順に集約するための優先度付きキュー
        self.results_queue = queue.PriorityQueue()

        # パイプラインワーカーの初期化
        self.workers = self._initialize_workers()

    def _initialize_workers(self) -> Dict[str, "BaseWorker"]:
        """各モデルに対応するワーカーを初期化します。"""
        workers = {}
        # 各ワーカーに共通の結果キューを渡す
        queues = {
            name: {
                "preprocess": queue.Queue(maxsize=16),
                "inference": queue.Queue(maxsize=16),
                "postprocess": queue.Queue(maxsize=16),
            } for name in self.predictors
        }
        
        # ここで各Workerクラスをインスタンス化
        # 例: workers["court"] = CourtWorker(...)
        # Ball, Court, Poseの各Workerを実装し、ここで初期化する必要があります
        # (上記 `court_worker.py` の例を参照)
        
        # ダミーワーカーで代替
        for name, pred in self.predictors.items():
            # 実際のプロジェクトでは、BallWorker, CourtWorker, PoseWorkerを正しくインポートして使用
            worker_class = CourtWorker # 仮にCourtWorkerで代用
            workers[name] = worker_class(
                 name, pred, queues[name]["preprocess"], queues[name]["inference"],
                 queues[name]["postprocess"], self.results_queue, self.debug
            )
        return workers

    def run(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        """動画処理のメインフローを実行します。"""
        input_path, output_path = Path(input_path), Path(output_path)

        # 1. I/Oとワーカーのセットアップ
        frame_loader = FrameLoader(input_path).start()
        props = frame_loader.get_properties()
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), props["fps"], (props["width"], props["height"]))
        
        for worker in self.workers.values():
            worker.start()

        # 2. フレーム投入 (Dispatcher)
        self._dispatch_frames(frame_loader, props["total_frames"])

        # 3. 結果の集約と描画
        self._aggregate_and_write_results(writer, props["total_frames"])
        
        # 4. クリーンアップ
        for worker in self.workers.values():
            worker.stop()
        frame_loader.release()
        writer.release()
        print(f"✅ 処理完了 → 出力動画: {output_path}")

    def _dispatch_frames(self, frame_loader: FrameLoader, total_frames: int):
        """フレームを読み込み、適切な間隔で各ワーカーにタスクを投入します。"""
        buffers = {name: [] for name in self.predictors}
        meta_buffers = {name: [] for name in self.predictors}
        
        print("🚀 フレームの投入を開始...")
        with tqdm(total=total_frames, desc="フレーム投入中") as pbar:
            while True:
                data = frame_loader.read()
                if data is None: break # 動画の終端
                
                frame_idx, frame = data
                
                for name, interval in self.intervals.items():
                    if frame_idx % interval == 0:
                        buffers[name].append(frame)
                        meta_buffers[name].append((frame_idx, frame.shape[0], frame.shape[1])) # (idx, H, W)
                
                    if len(buffers[name]) >= self.batch_sizes[name]:
                        task = PreprocessTask(f"{name}_{frame_idx}", buffers[name], meta_buffers[name])
                        self.workers[name].preprocess_queue.put(task)
                        buffers[name].clear()
                        meta_buffers[name].clear()
                pbar.update(1)

        # ループ終了後、バッファに残っているフレームを処理
        for name in self.predictors:
            if buffers[name]:
                task = PreprocessTask(f"{name}_final", buffers[name], meta_buffers[name])
                self.workers[name].preprocess_queue.put(task)

    def _aggregate_and_write_results(self, writer: cv2.VideoWriter, total_frames: int):
        """結果キューから推論結果を集約し、描画して動画ファイルに書き込みます。"""
        cached_preds = {name: None for name in self.predictors}
        results_by_frame: Dict[int, Dict[str, Any]] = {}
        
        # VideoCaptureをもう一度開いて描画用のフレームを取得
        cap = cv2.VideoCapture(str(writer.get_filename()))

        print("✍️ 結果の集約と動画書き込みを開始...")
        with tqdm(total=total_frames, desc="動画書き込み中") as pbar:
            for frame_idx in range(total_frames):
                # このフレームインデックスまでの結果をすべてキューから取り出す
                while not self.results_queue.empty() and self.results_queue.queue[0][0] <= frame_idx:
                    res_idx, name, result = self.results_queue.get()
                    if res_idx not in results_by_frame:
                        results_by_frame[res_idx] = {}
                    results_by_frame[res_idx][name] = result

                # 描画用の生フレームを取得
                ret, frame = cap.read()
                if not ret: break

                # キャッシュを更新
                if frame_idx in results_by_frame:
                    for name, result in results_by_frame[frame_idx].items():
                        cached_preds[name] = result
                    del results_by_frame[frame_idx] # メモリ解放

                # オーバーレイ描画
                annotated_frame = frame.copy()
                for name, pred in cached_preds.items():
                    if pred is not None:
                        # 各predictorのoverlayメソッドを呼び出す
                        annotated_frame = self.predictors[name].overlay(annotated_frame, pred)

                writer.write(annotated_frame)
                pbar.update(1)
        
        cap.release()