import queue
import time
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import logging

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

logger = logging.getLogger(__name__)


class MultiEventPredictor:
    """
    複数のモデル推論と統合されたイベント検知を並列実行し、結果を動画に描画するクラス。
    
    従来のball、court、poseに加えて、これらの情報を統合したevent検知を行います。
    """

    def __init__(
        self,
        ball_predictor, 
        court_predictor, 
        pose_predictor, 
        event_predictor,
        intervals: Dict[str, int], 
        batch_sizes: Dict[str, int],
        event_sequence_length: int = 16,
        debug: bool = False
    ):
        """
        Args:
            ball_predictor: ボール検知予測器
            court_predictor: コート検知予測器 
            pose_predictor: ポーズ検知予測器
            event_predictor: イベント検知予測器
            intervals: 各タスクの処理間隔
            batch_sizes: 各タスクのバッチサイズ
            event_sequence_length: イベント検知用シーケンス長
            debug: デバッグモード
        """
        self.predictors = {
            "ball": ball_predictor,
            "court": court_predictor,
            "pose": pose_predictor,
            "event": event_predictor,
        }
        self.intervals = intervals
        self.batch_sizes = batch_sizes
        self.event_sequence_length = event_sequence_length
        self.debug = debug

        # 結果をフレームインデックス順に集約するための優先度付きキュー
        self.results_queue = queue.PriorityQueue()
        
        # イベント検知用の特徴量統合キュー
        self.integrated_features_queue = queue.Queue(maxsize=32)

        # パイプラインワーカーの初期化
        self.workers = self._initialize_workers()
        
        # 各タスクの最新結果を保持
        self.latest_results = {
            "ball": None,
            "court": None, 
            "pose": None,
            "event": None
        }

    def _initialize_workers(self) -> Dict[str, "BaseWorker"]:
        """各モデルに対応するワーカーを初期化します。"""
        workers = {}
        
        # 各ワーカー用のキューを作成
        queues = {
            name: {
                "preprocess": queue.Queue(maxsize=16),
                "inference": queue.Queue(maxsize=16),
                "postprocess": queue.Queue(maxsize=16),
            } for name in self.predictors
        }
        
        # 各Workerクラスをインスタンス化
        worker_classes = {
            "ball": BallWorker,
            "court": CourtWorker,
            "pose": PoseWorker,
            "event": EventWorker,
        }
        
        for name, pred in self.predictors.items():
            if name == "event":
                # EventWorkerは特別な処理
                workers[name] = EventWorker(
                    name, pred, queues[name]["preprocess"], queues[name]["inference"],
                    queues[name]["postprocess"], self.results_queue, self.debug,
                    sequence_length=self.event_sequence_length
                )
            else:
                worker_class = worker_classes.get(name)
                if worker_class:
                    workers[name] = worker_class(
                        name, pred, queues[name]["preprocess"], queues[name]["inference"],
                        queues[name]["postprocess"], self.results_queue, self.debug
                    )
                else:
                    logger.warning(f"Worker class not found for {name}, using base worker")
                    # フォールバック
                    workers[name] = BaseWorker(
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
        writer = cv2.VideoWriter(
            str(output_path), 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            props["fps"], 
            (props["width"], props["height"])
        )
        
        # ワーカーを開始
        for worker in self.workers.values():
            worker.start()

        # 2. フレーム投入と特徴量統合のスレッドを開始
        self._start_feature_integration_thread()

        # 3. フレーム投入 (Dispatcher)
        self._dispatch_frames(frame_loader, props["total_frames"])

        # 4. 結果の集約と描画
        self._aggregate_and_write_results(writer, props["total_frames"])
        
        # 5. クリーンアップ
        for worker in self.workers.values():
            worker.stop()
        frame_loader.release()
        writer.release()
        
        logger.info(f"✅ 処理完了 → 出力動画: {output_path}")

    def _start_feature_integration_thread(self):
        """特徴量統合のためのバックグラウンドスレッドを開始します。"""
        import threading
        
        def integration_loop():
            """他のワーカーの結果を統合してEventWorkerに送信するループ"""
            while True:
                try:
                    # 結果キューから新しい結果を取得
                    if not self.results_queue.empty():
                        frame_idx, task_name, result = self.results_queue.get(timeout=0.1)
                        
                        # 基本タスクの結果のみを統合対象とする
                        if task_name in ["ball", "court", "pose"]:
                            self.latest_results[task_name] = result
                            
                            # すべてのタスクの結果が揃ったかチェック
                            if all(self.latest_results[name] is not None 
                                   for name in ["ball", "court", "pose"]):
                                
                                # 統合特徴量を作成してEventWorkerに送信
                                integrated_features = self._create_integrated_features(frame_idx)
                                if integrated_features:
                                    task = PreprocessTask(
                                        f"event_{frame_idx}",
                                        [integrated_features], 
                                        [(frame_idx, 0, 0)]  # frame_idx, dummy height, width
                                    )
                                    
                                    # EventWorkerに送信
                                    try:
                                        self.workers["event"].preprocess_queue.put(task, timeout=0.1)
                                    except queue.Full:
                                        logger.warning("EventWorker preprocess queue is full")
                        
                        # 結果を元のキューに戻す（他の処理用）
                        self.results_queue.put((frame_idx, task_name, result))
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.debug:
                        logger.error(f"特徴量統合エラー: {e}")
                    time.sleep(0.01)
        
        thread = threading.Thread(target=integration_loop, daemon=True)
        thread.start()

    def _create_integrated_features(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        各タスクの結果を統合して特徴量辞書を作成します。
        
        Args:
            frame_idx: フレームインデックス
            
        Returns:
            Dict[str, Any]: 統合された特徴量
        """
        try:
            ball_result = self.latest_results["ball"]
            court_result = self.latest_results["court"] 
            pose_result = self.latest_results["pose"]
            
            # Ball特徴量の抽出
            ball_features = None
            if ball_result and len(ball_result) > 0:
                # ball_resultの形式に応じて適切に処理
                if isinstance(ball_result, (list, tuple)) and len(ball_result) > 0:
                    ball_data = ball_result[0] if isinstance(ball_result[0], dict) else {}
                    # 座標と信頼度を抽出 [x, y, confidence]
                    ball_features = [
                        ball_data.get('x', 0.0), 
                        ball_data.get('y', 0.0), 
                        ball_data.get('confidence', 0.0)
                    ]
            
            # Court特徴量の抽出
            court_features = None
            if court_result:
                # コートキーポイントを平坦化 [x1, y1, v1, x2, y2, v2, ...]
                if isinstance(court_result, (list, np.ndarray)):
                    court_features = list(court_result.flatten())[:45]  # 15 keypoints * 3
                    # 不足分を0で埋める
                    while len(court_features) < 45:
                        court_features.append(0.0)
            
            # Player特徴量の抽出
            player_bbox_features = None
            player_pose_features = None
            if pose_result:
                # プレイヤー検知結果からbboxとposeを抽出
                player_bbox_features = []
                player_pose_features = []
                
                if isinstance(pose_result, list):
                    for player_data in pose_result:
                        if isinstance(player_data, dict):
                            # BBox: [x1, y1, x2, y2, confidence]
                            bbox = player_data.get('bbox', [0, 0, 0, 0])
                            confidence = player_data.get('confidence', 0.0)
                            bbox_feat = list(bbox) + [confidence]
                            player_bbox_features.append(bbox_feat[:5])
                            
                            # Pose: [x1, y1, v1, x2, y2, v2, ...] for 17 keypoints
                            pose = player_data.get('keypoints', [])
                            if len(pose) >= 51:  # 17 * 3
                                player_pose_features.append(pose[:51])
                            else:
                                # 不足分を0で埋める
                                padded_pose = list(pose) + [0.0] * (51 - len(pose))
                                player_pose_features.append(padded_pose)
            
            integrated_features = {
                'ball': ball_features,
                'court': court_features,
                'player_bbox': player_bbox_features,
                'player_pose': player_pose_features,
                'frame_idx': frame_idx
            }
            
            if self.debug:
                logger.debug(f"フレーム{frame_idx}の統合特徴量を作成")
                
            return integrated_features
            
        except Exception as e:
            logger.error(f"統合特徴量作成エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def _dispatch_frames(self, frame_loader: FrameLoader, total_frames: int):
        """フレームを読み込み、適切な間隔で各ワーカーにタスクを投入します。"""
        buffers = {name: [] for name in self.predictors if name != "event"}  # eventは除外
        meta_buffers = {name: [] for name in self.predictors if name != "event"}
        
        logger.info("🚀 フレームの投入を開始...")
        with tqdm(total=total_frames, desc="フレーム投入中") as pbar:
            while True:
                data = frame_loader.read()
                if data is None: 
                    break  # 動画の終端
                
                frame_idx, frame = data
                
                # 基本タスク（ball, court, pose）のみディスパッチ
                for name, interval in self.intervals.items():
                    if name == "event":
                        continue  # eventは統合処理で別途処理
                        
                    if frame_idx % interval == 0:
                        buffers[name].append(frame)
                        meta_buffers[name].append((frame_idx, frame.shape[0], frame.shape[1]))
                    
                    if len(buffers[name]) >= self.batch_sizes[name]:
                        task = PreprocessTask(f"{name}_{frame_idx}", buffers[name], meta_buffers[name])
                        self.workers[name].preprocess_queue.put(task)
                        buffers[name].clear()
                        meta_buffers[name].clear()
                        
                pbar.update(1)

        # ループ終了後、バッファに残っているフレームを処理
        for name in buffers:
            if buffers[name]:
                task = PreprocessTask(f"{name}_final", buffers[name], meta_buffers[name])
                self.workers[name].preprocess_queue.put(task)

    def _aggregate_and_write_results(self, writer: cv2.VideoWriter, total_frames: int):
        """結果キューから推論結果を集約し、描画して動画ファイルに書き込みます。"""
        cached_preds = {name: None for name in self.predictors}
        results_by_frame: Dict[int, Dict[str, Any]] = {}
        
        # VideoCaptureをもう一度開いて描画用のフレームを取得
        input_path = writer.get(cv2.CAP_PROP_FILENAME)
        cap = cv2.VideoCapture(input_path) if input_path else None

        logger.info("✍️ 結果の集約と動画書き込みを開始...")
        with tqdm(total=total_frames, desc="動画書き込み中") as pbar:
            for frame_idx in range(total_frames):
                # このフレームインデックスまでの結果をすべてキューから取り出す
                while not self.results_queue.empty() and self.results_queue.queue[0][0] <= frame_idx:
                    res_idx, name, result = self.results_queue.get()
                    if res_idx not in results_by_frame:
                        results_by_frame[res_idx] = {}
                    results_by_frame[res_idx][name] = result

                # 描画用の生フレームを取得
                if cap:
                    ret, frame = cap.read()
                    if not ret: 
                        break
                else:
                    # フォールバック: 黒いフレームを生成
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # キャッシュを更新
                if frame_idx in results_by_frame:
                    for name, result in results_by_frame[frame_idx].items():
                        cached_preds[name] = result
                    del results_by_frame[frame_idx]  # メモリ解放

                # オーバーレイ描画
                annotated_frame = frame.copy()
                for name, pred in cached_preds.items():
                    if pred is not None and hasattr(self.predictors[name], 'overlay'):
                        # 各predictorのoverlayメソッドを呼び出す
                        annotated_frame = self.predictors[name].overlay(annotated_frame, pred)

                writer.write(annotated_frame)
                pbar.update(1)
        
        if cap:
            cap.release() 