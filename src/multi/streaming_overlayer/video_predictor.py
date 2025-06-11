# streaming_annotator/video_predictor.py

import queue
import time
from pathlib import Path
from typing import Dict, List, Union, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .definitions import PreprocessTask
from .video_utils import FrameLoader
from .workers.base_worker import BaseWorker
from .workers.ball_worker import BallWorker
from .workers.court_worker import CourtWorker
from .workers.pose_worker import PoseWorker
from .queue_manager import QueueManager, create_queue_manager_for_video_predictor
from .config_utils import (
    create_queue_configs_from_hydra_config,
    get_worker_extended_queue_names,
    apply_performance_settings,
    validate_queue_config,
    log_queue_configuration
)

class VideoPredictor:
    """動画に対して複数のモデル推論を並列実行し、結果を描画するクラス。"""

    def __init__(
        self,
        ball_predictor, court_predictor, pose_predictor,
        intervals: Dict[str, int], batch_sizes: Dict[str, int],
        debug: bool = False,
        custom_queue_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        hydra_queue_config: Optional[Any] = None
    ):
        self.predictors = {
            "ball": ball_predictor,
            "court": court_predictor,
            "pose": pose_predictor,
        }
        self.intervals = intervals
        self.batch_sizes = batch_sizes
        self.debug = debug

        # 拡張可能なキューシステムを初期化
        worker_names = list(self.predictors.keys())
        
        # Hydra設定からキューコンフィグを作成
        if hydra_queue_config is not None:
            try:
                # 設定検証
                if validate_queue_config(hydra_queue_config):
                    log_queue_configuration(hydra_queue_config)
                    
                    # Hydra設定をQueueManager形式に変換
                    queue_configs_from_hydra = create_queue_configs_from_hydra_config(hydra_queue_config)
                    
                    # カスタム設定とマージ
                    final_queue_configs = queue_configs_from_hydra.copy()
                    if custom_queue_configs:
                        final_queue_configs.update(custom_queue_configs)
                    
                    self.queue_manager = create_queue_manager_for_video_predictor(
                        worker_names, 
                        final_queue_configs
                    )
                    
                    # パフォーマンス設定を適用
                    performance_settings = apply_performance_settings(hydra_queue_config)
                    self._apply_performance_settings(performance_settings)
                    
                else:
                    raise ValueError("Hydra キュー設定の検証に失敗しました")
                    
            except Exception as e:
                print(f"⚠️ Hydra設定の読み込みに失敗: {e}")
                print("🔄 デフォルト設定にフォールバック")
                self.queue_manager = create_queue_manager_for_video_predictor(
                    worker_names, 
                    custom_queue_configs
                )
        else:
            self.queue_manager = create_queue_manager_for_video_predictor(
                worker_names, 
                custom_queue_configs
            )

        # パイプラインワーカーの初期化
        self.workers = self._initialize_workers()
        
        # パフォーマンス監視設定
        self.performance_settings = {}
        
        # パフォーマンス設定をここで初期化（デフォルト値）
        self.performance_settings = {"enable_monitoring": True}
    
    def _apply_performance_settings(self, settings: Dict[str, Any]):
        """パフォーマンス設定を適用"""
        self.performance_settings.update(settings)
        
        if settings.get('enable_monitoring', False):
            print("📊 キュー監視機能が有効化されました")
        
        if settings.get('log_queue_status', False):
            print("📝 キュー状態ログ出力が有効化されました")
        
        if settings.get('gpu_optimization', False):
            print("🚀 GPU最適化モードが有効化されました")
    
    def get_queue_status_with_settings(self) -> Dict[str, Any]:
        """設定に基づいてキュー状態を取得"""
        if not self.performance_settings.get('enable_monitoring', True):
            return {"monitoring": "disabled"}
        
        status = self.queue_manager.get_queue_status()
        
        if self.performance_settings.get('log_queue_status', False):
            print("📊 Queue Status:")
            print(f"  Results queue: {status['results_queue_size']} items")
            for worker, info in status['workers'].items():
                print(f"  {worker}: {sum(info['base_queues'].values())} base + {sum(info['extended_queues'].values())} extended")
        
        return status

    def _initialize_workers(self) -> Dict[str, "BaseWorker"]:
        """各モデルに対応するワーカーを初期化します。"""
        workers = {}
        
        # 正しいワーカークラスを割り当て
        worker_classes = {
            "ball": BallWorker,
            "court": CourtWorker,
            "pose": PoseWorker,
        }

        for name, pred in self.predictors.items():
            worker_class = worker_classes.get(name, BaseWorker)
            
            # QueueManagerからキューセットを取得
            queue_set = self.queue_manager.get_worker_queue_set(name)
            if not queue_set:
                raise ValueError(f"Queue set for worker '{name}' not found")
            
            workers[name] = worker_class(
                name,
                pred,
                queue_set,  # キューセット全体を渡す
                self.queue_manager.get_results_queue(),
                self.debug,
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
        self._aggregate_and_write_results(writer, input_path, props["total_frames"])
        
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
                        preprocess_queue = self.queue_manager.get_queue(name, "preprocess")
                        preprocess_queue.put(task)
                        buffers[name].clear()
                        meta_buffers[name].clear()
                pbar.update(1)

        # ループ終了後、バッファに残っているフレームを処理
        for name in self.predictors:
            if buffers[name]:
                task = PreprocessTask(f"{name}_final", buffers[name], meta_buffers[name])
                preprocess_queue = self.queue_manager.get_queue(name, "preprocess")
                preprocess_queue.put(task)

    def _aggregate_and_write_results(self, writer: cv2.VideoWriter, input_path: Path, total_frames: int):
        """結果キューから推論結果を集約し、描画して動画ファイルに書き込みます。"""
        cached_preds = {name: None for name in self.predictors}
        results_by_frame: Dict[int, Dict[str, Any]] = {}
        
        # VideoCaptureをもう一度開いて描画用のフレームを取得
        cap = cv2.VideoCapture(str(input_path))

        print("✍️ 結果の集約と動画書き込みを開始...")
        with tqdm(total=total_frames, desc="動画書き込み中") as pbar:
            for frame_idx in range(total_frames):
                # このフレームインデックスまでの結果をすべてキューから取り出す
                results_queue = self.queue_manager.get_results_queue()
                while not results_queue.empty() and results_queue.queue[0][0] <= frame_idx:
                    res_idx, name, result = results_queue.get()
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
                        # Ball など list が返る場合は 1 フレーム分を想定して 0 番目を使用
                        to_draw = pred[0] if isinstance(pred, list) else pred
                        try:
                            annotated_frame = self.predictors[name].overlay(annotated_frame, to_draw)
                        except Exception:
                            # overlay 失敗時は描画をスキップ
                            pass

                writer.write(annotated_frame)
                pbar.update(1)
        
        cap.release()