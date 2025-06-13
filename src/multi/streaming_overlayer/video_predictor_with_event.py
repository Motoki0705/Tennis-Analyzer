# streaming_overlayer/video_predictor_with_event.py

import queue
import time
import threading
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from .definitions import PreprocessTask
from .video_utils import FrameLoader
from .workers.base_worker import BaseWorker
from .workers.ball_worker import BallWorker
from .workers.court_worker import CourtWorker
from .workers.pose_worker import PoseWorker
from .workers.event_worker import EventWorker
from .queue_manager import QueueManager, create_queue_manager_for_video_predictor
from .config_utils import (
    create_queue_configs_from_hydra_config,
    get_worker_extended_queue_names,
    apply_performance_settings,
    validate_queue_config,
    log_queue_configuration
)


class VideoPredictorWithEvent:
    """
    動画に対して複数のモデル推論を並列実行し、結果を描画するクラス。
    
    ball, court, pose, eventの4つのワーカーを統合し、
    eventワーカーは他の3つのワーカーの結果を利用します。
    """

    def __init__(
        self,
        ball_predictor, court_predictor, pose_predictor, event_predictor=None,
        intervals: Dict[str, int] = None, batch_sizes: Dict[str, int] = None,
        debug: bool = False,
        custom_queue_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        hydra_queue_config: Optional[Any] = None,
        max_preload_frames: int = 64,
        enable_performance_monitoring: bool = True,
        event_sequence_length: int = 16
    ):
        # 基本予測器の設定
        self.predictors = {
            "ball": ball_predictor,
            "court": court_predictor,
            "pose": pose_predictor,
        }
        
        # eventワーカーは後で追加（他のワーカーの結果を使用するため）
        self.event_predictor = event_predictor
        self.event_sequence_length = event_sequence_length
        
        # デフォルト値の設定
        self.intervals = intervals or {"ball": 1, "court": 1, "pose": 1, "event": 1}
        self.batch_sizes = batch_sizes or {"ball": 1, "court": 1, "pose": 1, "event": 1}
        
        self.debug = debug
        self.max_preload_frames = max_preload_frames
        self.enable_performance_monitoring = enable_performance_monitoring

        # 拡張可能なキューシステムを初期化
        worker_names = list(self.predictors.keys())
        if self.event_predictor is not None:
            worker_names.append("event")
        
        # Hydra設定からキューコンフィグを作成
        if hydra_queue_config is not None:
            try:
                if validate_queue_config(hydra_queue_config):
                    log_queue_configuration(hydra_queue_config)
                    
                    queue_configs_from_hydra = create_queue_configs_from_hydra_config(hydra_queue_config)
                    
                    final_queue_configs = queue_configs_from_hydra.copy()
                    if custom_queue_configs:
                        final_queue_configs.update(custom_queue_configs)
                    
                    self.queue_manager = create_queue_manager_for_video_predictor(
                        worker_names, 
                        final_queue_configs
                    )
                    
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
        
        # フレーム処理用スレッドプール
        self.frame_processing_pool = ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="frame_processor"
        )
        
        # パフォーマンス設定（デフォルト値）
        self.performance_settings = {"enable_monitoring": True}
        
        # パフォーマンス監視メトリクス
        self.performance_metrics = {
            "total_frames_processed": 0,
            "total_processing_time": 0.0,
            "frames_per_second": 0.0,
            "queue_throughput": {},
            "worker_performance": {},
            "start_time": None,
            "end_time": None
        }
        
        # スライディングウィンドウ管理
        self.sliding_windows = {}
        self.sliding_window_lock = threading.Lock()
        
        # Eventワーカー統合用の結果バッファ
        self.event_worker = None
        if self.event_predictor is not None:
            self.event_worker = self.workers.get("event")
    
    def _apply_performance_settings(self, settings: Dict[str, Any]):
        """パフォーマンス設定を適用"""
        self.performance_settings.update(settings)
        
        if settings.get('enable_monitoring', False):
            print("📊 キュー監視機能が有効化されました")
        
        if settings.get('log_queue_status', False):
            print("📝 キュー状態ログ出力が有効化されました")
        
        if settings.get('gpu_optimization', False):
            print("🚀 GPU最適化モードが有効化されました")

    def _initialize_workers(self) -> Dict[str, "BaseWorker"]:
        """各モデルに対応するワーカーを初期化します"""
        workers = {}
        
        # 基本ワーカークラスの割り当て
        worker_classes = {
            "ball": BallWorker,
            "court": CourtWorker,
            "pose": PoseWorker,
        }

        # 基本ワーカーを初期化
        for name, pred in self.predictors.items():
            worker_class = worker_classes.get(name, BaseWorker)
            
            queue_set = self.queue_manager.get_worker_queue_set(name)
            if not queue_set:
                raise ValueError(f"Queue set for worker '{name}' not found")
            
            workers[name] = worker_class(
                name,
                pred,  
                queue_set,
                self.queue_manager.get_results_queue(),
                self.debug,
            )
        
        # Eventワーカーを初期化（他のワーカーの結果を使用）
        if self.event_predictor is not None:
            event_queue_set = self.queue_manager.get_worker_queue_set("event")
            if event_queue_set:
                workers["event"] = EventWorker(
                    "event",
                    self.event_predictor,
                    event_queue_set,
                    self.queue_manager.get_results_queue(),
                    self.debug,
                    sequence_length=self.event_sequence_length
                )
            
        return workers

    def run(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        """動画処理のメインフローを実行します"""
        input_path, output_path = Path(input_path), Path(output_path)

        # パフォーマンス監視開始
        self.performance_metrics["start_time"] = time.time()

        try:
            # I/Oとワーカーのセットアップ
            frame_loader = FrameLoader(input_path).start()
            props = frame_loader.get_properties()
            writer = cv2.VideoWriter(
                str(output_path), 
                cv2.VideoWriter_fourcc(*"mp4v"), 
                props["fps"], 
                (props["width"], props["height"])
            )

            # ワーカーを開始
            for name, worker in self.workers.items():
                worker.start()
                if self.debug:
                    print(f"🚀 {name} ワーカーを開始")

            total_frames = props["total_frames"]
            
            # フレーム配信と結果集約を並列実行
            with ThreadPoolExecutor(max_workers=2) as executor:
                # フレーム配信スレッド
                dispatch_future = executor.submit(
                    self._dispatch_frames_parallel, frame_loader, total_frames
                )
                
                # 結果集約・描画スレッド
                aggregation_future = executor.submit(
                    self._aggregate_and_write_results, writer, input_path, total_frames
                )
                
                # 両方の処理を待機
                dispatch_future.result()
                aggregation_future.result()

        finally:
            # リソースクリーンアップ
            if 'frame_loader' in locals():
                frame_loader.stop()
            if 'writer' in locals():
                writer.release()
            
            # ワーカーを停止
            for name, worker in self.workers.items():
                worker.stop()
                if self.debug:
                    print(f"🛑 {name} ワーカーを停止")
            
            # パフォーマンス統計を終了
            self._finalize_performance_metrics()
            if self.enable_performance_monitoring:
                self._print_performance_summary()

    def _dispatch_frames_parallel(self, frame_loader: FrameLoader, total_frames: int):
        """フレームを並列配信します"""
        buffers = {name: [] for name in self.predictors.keys()}
        meta_buffers = {name: [] for name in self.predictors.keys()}
        
        with tqdm(total=total_frames, desc="フレーム処理", unit="frame") as pbar:
            for frame_idx, frame in enumerate(frame_loader):
                # 各ワーカーのバッファにフレームを追加
                self._process_single_frame(frame_idx, frame, buffers, meta_buffers)
                
                # バッチサイズに達したワーカーのタスクを送信
                for name in self.predictors.keys():
                    if len(buffers[name]) >= self.batch_sizes[name]:
                        self._create_and_submit_task(name, frame_idx, buffers[name], meta_buffers[name])
                        buffers[name].clear()
                        meta_buffers[name].clear()
                
                pbar.update(1)
                
                # パフォーマンス監視
                if self.enable_performance_monitoring and frame_idx % 100 == 0:
                    self.get_queue_status_with_settings()
        
        # 残りのバッファを処理
        for name in self.predictors.keys():
            if buffers[name]:
                self._create_and_submit_task(name, frame_idx, buffers[name], meta_buffers[name])

    def _process_single_frame(self, frame_idx: int, frame: np.ndarray, buffers: Dict, meta_buffers: Dict) -> tuple:
        """単一フレームを処理してバッファに追加します"""
        for name in self.predictors.keys():
            # インターバルチェック
            if frame_idx % self.intervals[name] == 0:
                buffers[name].append(frame.copy())
                meta_buffers[name].append((frame_idx, frame.shape))

    def _create_and_submit_task(self, name: str, frame_idx: int, frames: List, meta_data: List):
        """前処理タスクを作成してキューに送信します"""
        task = PreprocessTask(f"{name}_{frame_idx}", frames, meta_data)
        
        try:
            self.workers[name].preprocess_queue.put(task, timeout=1.0)
        except queue.Full:
            if self.debug:
                print(f"⚠️ {name} 前処理キューが満杯です")

    def _aggregate_and_write_results(self, writer: cv2.VideoWriter, input_path: Union[str, Path], total_frames: int):
        """結果を集約してフレームに描画し、動画に書き込みます"""
        processed_frames = 0
        frame_cache = {}  # frame_idx -> cached_predictions
        
        # フレームローダーを再度開いて描画用フレームを取得
        drawing_frame_loader = FrameLoader(input_path).start()
        drawing_frames = {i: frame for i, frame in enumerate(drawing_frame_loader)}
        drawing_frame_loader.stop()
        
        while processed_frames < total_frames:
            try:
                # 結果キューから結果を取得
                item = self.queue_manager.get_results_queue().get(timeout=5.0)
                
                if isinstance(item, dict) and "frame_idx" in item:
                    frame_idx = item["frame_idx"]
                    worker_name = item["worker_name"]
                    prediction = item["prediction"]
                    
                    # フレームキャッシュに結果を保存
                    if frame_idx not in frame_cache:
                        frame_cache[frame_idx] = {}
                    frame_cache[frame_idx][worker_name] = prediction
                    
                    # Eventワーカーが存在する場合、他のワーカーの結果を転送
                    if self.event_worker and worker_name in ["ball", "court", "pose"]:
                        self.event_worker.add_external_result(frame_idx, worker_name, prediction)
                    
                    # このフレームが完成したかチェック
                    expected_workers = set(self.predictors.keys())
                    if self.event_predictor is not None:
                        expected_workers.add("event")
                    
                    available_workers = set(frame_cache[frame_idx].keys())
                    
                    # 基本ワーカー（ball, court, pose）の結果が揃った場合は描画可能
                    basic_workers = {"ball", "court", "pose"}
                    if basic_workers.issubset(available_workers):
                        self._draw_and_write_frame(
                            writer, frame_idx, frame_cache[frame_idx], 
                            drawing_frames.get(frame_idx)
                        )
                        processed_frames += 1
                        
                        # 古いキャッシュをクリーンアップ
                        if frame_idx in frame_cache:
                            del frame_cache[frame_idx]
                
                self.queue_manager.get_results_queue().task_done()
                
            except queue.Empty:
                if self.debug:
                    print(f"⏳ 結果待機中... 処理済み: {processed_frames}/{total_frames}")
                continue
            except Exception as e:
                print(f"⚠️ 結果集約エラー: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue

    def _draw_and_write_frame(self, writer: cv2.VideoWriter, frame_idx: int, 
                            cached_preds: Dict[str, Any], frame: Optional[np.ndarray]):
        """フレームに予測結果を描画して動画に書き込みます"""
        if frame is None:
            # フレームが取得できない場合は黒いフレームを作成
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        try:
            # オーバーレイ描画
            annotated_frame = frame.copy()
            
            # 基本ワーカーの結果を描画
            for name, pred in cached_preds.items():
                if pred is not None and name in self.predictors:
                    try:
                        annotated_frame = self.predictors[name].overlay(annotated_frame, pred)
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️ {name} のオーバーレイ失敗 for frame {frame_idx}: {e}")
                        pass
            
            # Eventワーカーの結果を描画（存在する場合）
            if "event" in cached_preds and self.event_predictor is not None:
                try:
                    annotated_frame = self.event_predictor.overlay(annotated_frame, cached_preds["event"])
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ event のオーバーレイ失敗 for frame {frame_idx}: {e}")
                    pass
            
            # 動画に書き込み
            writer.write(annotated_frame)
            
            if self.debug and frame_idx % 100 == 0:
                print(f"📝 フレーム {frame_idx} 描画完了")
                
        except Exception as e:
            print(f"⚠️ フレーム描画エラー {frame_idx}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー時は元のフレームを書き込み
            writer.write(frame)

    def get_queue_status_with_settings(self) -> Dict[str, Any]:
        """設定に基づいてキュー状態を取得します"""
        if not self.performance_settings.get('enable_monitoring', True):
            return {"monitoring": "disabled"}
        
        status = self.queue_manager.get_queue_status()
        
        if self.performance_settings.get('log_queue_status', False):
            print("📊 Queue Status:")
            print(f"  Results queue: {status['results_queue_size']} items")
            for worker, info in status['workers'].items():
                print(f"  {worker}: {sum(info['base_queues'].values())} base + {sum(info['extended_queues'].values())} extended")
        
        return status

    def _finalize_performance_metrics(self):
        """パフォーマンス統計を終了します"""
        self.performance_metrics["end_time"] = time.time()
        total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
        self.performance_metrics["total_processing_time"] = total_time
        
        if total_time > 0:
            self.performance_metrics["frames_per_second"] = self.performance_metrics["total_frames_processed"] / total_time

    def _print_performance_summary(self):
        """パフォーマンスサマリーを出力します"""
        metrics = self.performance_metrics
        print("\n📊 === パフォーマンスサマリー ===")
        print(f"総処理時間: {metrics['total_processing_time']:.2f}秒")
        print(f"処理済みフレーム数: {metrics['total_frames_processed']}")
        print(f"平均FPS: {metrics['frames_per_second']:.2f}")
        
        # ワーカー別統計
        print("\n🔧 ワーカー別統計:")
        for name, worker in self.workers.items():
            if hasattr(worker, 'get_performance_stats'):
                stats = worker.get_performance_stats()
                print(f"  {name}: {stats}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得します"""
        return self.performance_metrics.copy()


def create_video_predictor_with_event(
    ball_predictor, court_predictor, pose_predictor, event_predictor=None,
    **kwargs
) -> VideoPredictorWithEvent:
    """
    Eventワーカー統合済みのVideoPredictorを作成するファクトリ関数
    
    Args:
        ball_predictor: ボール予測器
        court_predictor: コート予測器  
        pose_predictor: ポーズ予測器
        event_predictor: イベント予測器（オプション）
        **kwargs: その他のVideoPredictor初期化パラメータ
        
    Returns:
        VideoPredictorWithEvent: 設定済みのVideoPredictor インスタンス
    """
    return VideoPredictorWithEvent(
        ball_predictor=ball_predictor,
        court_predictor=court_predictor,
        pose_predictor=pose_predictor,
        event_predictor=event_predictor,
        **kwargs
    ) 