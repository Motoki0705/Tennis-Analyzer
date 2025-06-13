import torch
import logging
import threading
import numpy as np
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Dict, Optional, Tuple
from queue import Queue, Empty

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

logger = logging.getLogger(__name__)


class EventWorker(BaseWorker):
    """
    イベント検知のための統合ワーカー。
    
    他のワーカー（ball, court, pose）からの結果を統合し、
    時系列データを作成してevent推論を実行します。
    """
    
    def __init__(self, name, predictor, queue_set, results_q, debug=False, 
                 sequence_length: int = 16, max_players: int = 4):
        # QueueManagerから基本キューを取得
        preprocess_q = queue_set.get_queue("preprocess")
        inference_q = queue_set.get_queue("inference")
        postprocess_q = queue_set.get_queue("postprocess")
        
        super().__init__(name, predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)
        
        # シーケンス管理
        self.sequence_length = sequence_length
        self.max_players = max_players
        
        # 各ワーカーからの結果を保持するバッファ
        # frame_idx -> {worker_name: result}
        self.result_buffer: Dict[int, Dict[str, Any]] = {}  
        self.buffer_lock = threading.Lock()
        
        # 時系列シーケンス管理
        self.sequence_frames: deque = deque(maxlen=sequence_length)
        self.sequence_lock = threading.Lock()
        
        # スレッドプール
        self.preprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_preprocess")
        self.postprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_postprocess")
        
        # パフォーマンス監視
        self.preprocess_count = 0
        self.inference_count = 0
        self.postprocess_count = 0
        
        # 外部結果統合用キュー（他のワーカーからの結果を受け取る）
        self.external_results_queue = Queue()
        self.integration_thread = None
        self.integration_running = False
        
    def start(self):
        """ワーカーと統合スレッドを開始します"""
        super().start()
        
        # 外部結果統合スレッドを開始
        self.integration_running = True
        self.integration_thread = threading.Thread(
            target=self._external_results_integration_loop, 
            name=f"{self.name}_integration"
        )
        self.integration_thread.daemon = True
        self.integration_thread.start()
        
        logger.info(f"EventWorker統合スレッドを開始: {self.name}")
        
    def stop(self):
        """ワーカーとスレッドプールを停止します"""
        # 統合スレッドを停止
        self.integration_running = False
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=1.0)
        
        # スレッドプールをシャットダウン
        if hasattr(self, 'preprocess_pool'):
            self.preprocess_pool.shutdown(wait=False)
        if hasattr(self, 'postprocess_pool'):
            self.postprocess_pool.shutdown(wait=False)
        
        # ベースクラスの停止処理
        super().stop()

    def add_external_result(self, frame_idx: int, worker_name: str, result: Any):
        """
        外部ワーカーからの結果を追加します
        
        Args:
            frame_idx: フレームインデックス
            worker_name: ワーカー名 (ball, court, pose)
            result: ワーカーの結果
        """
        try:
            self.external_results_queue.put((frame_idx, worker_name, result), timeout=1.0)
        except Exception as e:
            logger.warning(f"外部結果の追加に失敗: {frame_idx}, {worker_name}, {e}")

    def _external_results_integration_loop(self):
        """外部結果統合ループ"""
        while self.integration_running:
            try:
                frame_idx, worker_name, result = self.external_results_queue.get(timeout=0.1)
                self._integrate_external_result(frame_idx, worker_name, result)
                self.external_results_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"外部結果統合エラー: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _integrate_external_result(self, frame_idx: int, worker_name: str, result: Any):
        """外部結果をバッファに統合し、シーケンスを更新"""
        with self.buffer_lock:
            if frame_idx not in self.result_buffer:
                self.result_buffer[frame_idx] = {}
            
            self.result_buffer[frame_idx][worker_name] = result
            
            # 全ワーカーの結果が揃ったかチェック
            required_workers = {"ball", "court", "pose"}
            if set(self.result_buffer[frame_idx].keys()) >= required_workers:
                self._update_sequence(frame_idx, self.result_buffer[frame_idx])
                
                # バッファをクリーンアップ（古いフレームを削除）
                self._cleanup_buffer()

    def _update_sequence(self, frame_idx: int, combined_results: Dict[str, Any]):
        """シーケンスを更新し、必要に応じて推論タスクを生成"""
        with self.sequence_lock:
            # フレームデータを特徴量に変換
            frame_features = self._convert_to_features(frame_idx, combined_results)
            
            # シーケンスに追加
            self.sequence_frames.append((frame_idx, frame_features))
            
            # シーケンスが十分な長さになったら前処理タスクを生成
            if len(self.sequence_frames) >= self.sequence_length:
                sequence_data = list(self.sequence_frames)
                
                # 前処理タスクを作成
                from ..definitions import PreprocessTask
                task = PreprocessTask(
                    task_id=f"event_{frame_idx}",
                    frames=[],  # 既に特徴量化済み
                    meta_data=[(idx, features) for idx, features in sequence_data]
                )
                
                # 前処理キューに追加
                try:
                    self.preprocess_queue.put(task, timeout=1.0)
                    if self.debug:
                        logger.debug(f"EventWorker前処理タスク作成: {frame_idx}")
                except Exception as e:
                    logger.warning(f"前処理タスク追加失敗: {e}")

    def _convert_to_features(self, frame_idx: int, combined_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        各ワーカーの結果を統一された特徴量に変換
        
        Args:
            frame_idx: フレームインデックス
            combined_results: {worker_name: result} の辞書
            
        Returns:
            Dict[str, np.ndarray]: 特徴量辞書
        """
        features = {}
        
        try:
            # Ball特徴量 [3] (x, y, confidence)
            ball_result = combined_results.get("ball", {})
            if ball_result and 'x' in ball_result and 'y' in ball_result:
                # 座標を正規化（ここでは仮の画像サイズを使用）
                # 実際の実装では画像サイズを取得する必要があります
                img_width, img_height = 1920, 1080  # 仮の値
                features["ball"] = np.array([
                    ball_result['x'] / img_width,
                    ball_result['y'] / img_height,
                    ball_result.get('confidence', 0.0)
                ], dtype=np.float32)
            else:
                features["ball"] = np.zeros(3, dtype=np.float32)
            
            # Court特徴量 [45] (15個のキーポイント × 3)
            court_result = combined_results.get("court", [])
            court_features = np.zeros(45, dtype=np.float32)
            if court_result and isinstance(court_result, list):
                # 上位15個のキーポイントを使用
                for i, point in enumerate(court_result[:15]):
                    if isinstance(point, dict) and 'x' in point and 'y' in point:
                        court_features[i*3] = point['x'] / img_width
                        court_features[i*3+1] = point['y'] / img_height
                        court_features[i*3+2] = point.get('confidence', 0.0)
            features["court"] = court_features
            
            # Player特徴量 [max_players, 5+51]
            pose_result = combined_results.get("pose", [])
            player_bbox_features = np.zeros((self.max_players, 5), dtype=np.float32)
            player_pose_features = np.zeros((self.max_players, 51), dtype=np.float32)  # 17個のキーポイント × 3
            
            if pose_result and isinstance(pose_result, list):
                for i, person in enumerate(pose_result[:self.max_players]):
                    if isinstance(person, dict):
                        # BBox特徴量
                        bbox = person.get('bbox', [0, 0, 0, 0])
                        if len(bbox) >= 4:
                            player_bbox_features[i] = np.array([
                                bbox[0] / img_width,      # x1
                                bbox[1] / img_height,     # y1
                                (bbox[0] + bbox[2]) / img_width,   # x2
                                (bbox[1] + bbox[3]) / img_height,  # y2
                                person.get('det_score', 0.0)       # confidence
                            ], dtype=np.float32)
                        
                        # Pose特徴量
                        keypoints = person.get('keypoints', [])
                        scores = person.get('scores', [])
                        if keypoints and len(keypoints) >= 17:
                            for j, (x, y) in enumerate(keypoints[:17]):
                                player_pose_features[i, j*3] = x / img_width
                                player_pose_features[i, j*3+1] = y / img_height
                                player_pose_features[i, j*3+2] = scores[j] if j < len(scores) else 0.0
            
            features["player_bbox"] = player_bbox_features
            features["player_pose"] = player_pose_features
            
            return features
            
        except Exception as e:
            logger.error(f"特徴量変換エラー {frame_idx}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー時はゼロ特徴量を返す
            return {
                "ball": np.zeros(3, dtype=np.float32),
                "court": np.zeros(45, dtype=np.float32),
                "player_bbox": np.zeros((self.max_players, 5), dtype=np.float32),
                "player_pose": np.zeros((self.max_players, 51), dtype=np.float32)
            }

    def _cleanup_buffer(self):
        """古いフレームデータをバッファから削除"""
        if len(self.result_buffer) > self.sequence_length * 2:
            # 古いフレームを削除（最新のsequence_length * 2フレームのみ保持）
            frame_indices = sorted(self.result_buffer.keys())
            for frame_idx in frame_indices[:-self.sequence_length * 2]:
                del self.result_buffer[frame_idx]

    def _process_preprocess_task(self, task):
        """前処理タスクを処理します"""
        try:
            future = self.preprocess_pool.submit(self._execute_preprocess, task)
            
            try:
                processed_data = future.result(timeout=5.0)
                
                if processed_data is not None:
                    # 推論タスクをキューに送信
                    inference_task = InferenceTask(
                        task.task_id, 
                        processed_data, 
                        task.meta_data
                    )
                    self.inference_queue.put(inference_task)
                    
                    self.preprocess_count += 1
                    
                    if self.debug:
                        logger.debug(f"EventWorker前処理完了: {task.task_id}")
                        
            except Exception as e:
                logger.error(f"EventWorker前処理実行エラー: {task.task_id}, {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"EventWorker前処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _execute_preprocess(self, task) -> Optional[Dict[str, torch.Tensor]]:
        """実際の前処理を実行します"""
        try:
            if not task.meta_data or len(task.meta_data) < self.sequence_length:
                return None
            
            # シーケンスデータからテンソルを構築
            batch_size = 1
            ball_sequence = []
            court_sequence = []
            player_bbox_sequence = []
            player_pose_sequence = []
            
            # 最新のsequence_lengthフレームを使用
            recent_frames = task.meta_data[-self.sequence_length:]
            
            for frame_idx, features in recent_frames:
                ball_sequence.append(features["ball"])
                court_sequence.append(features["court"])
                player_bbox_sequence.append(features["player_bbox"])
                player_pose_sequence.append(features["player_pose"])
            
            # テンソルに変換
            processed_data = {
                'ball_features': torch.tensor(np.array(ball_sequence), dtype=torch.float32).unsqueeze(0),  # [1, T, 3]
                'court_features': torch.tensor(np.array(court_sequence), dtype=torch.float32).unsqueeze(0),  # [1, T, 45]
                'player_bbox_features': torch.tensor(np.array(player_bbox_sequence), dtype=torch.float32).unsqueeze(0),  # [1, T, P, 5]
                'player_pose_features': torch.tensor(np.array(player_pose_sequence), dtype=torch.float32).unsqueeze(0)   # [1, T, P, 51]
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"EventWorker前処理実行中にエラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def _process_inference_task(self, task):
        """推論タスクを処理します"""
        try:
            # GPU推論を実行
            with torch.no_grad():
                event_predictions = self.predictor.inference(task.tensor_data)
            
            # 後処理タスクをキューに送信
            postprocess_task = PostprocessTask(
                task.task_id,
                event_predictions,
                task.meta_data
            )
            self.postprocess_queue.put(postprocess_task)
            
            self.inference_count += 1
            
            if self.debug:
                logger.debug(f"EventWorker推論完了: {task.task_id}")
                
        except Exception as e:
            logger.error(f"EventWorker推論エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_postprocess_task(self, task):
        """後処理タスクを処理します"""
        try:
            future = self.postprocess_pool.submit(self._execute_postprocess, task)
            
            try:
                results = future.result(timeout=3.0)
                
                if results:
                    # 最新フレームの結果を結果キューに送信
                    if task.meta_data:
                        latest_frame_idx = task.meta_data[-1][0]  # 最新フレームのインデックス
                        
                        result_item = {
                            "frame_idx": latest_frame_idx,
                            "worker_name": self.name,
                            "prediction": results
                        }
                        
                        self.results_queue.put(result_item)
                        
                        self.postprocess_count += 1
                        
                        if self.debug:
                            logger.debug(f"EventWorker後処理完了: {task.task_id}, frame: {latest_frame_idx}")
                            
            except Exception as e:
                logger.error(f"EventWorker後処理実行エラー: {task.task_id}, {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"EventWorker後処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _execute_postprocess(self, task) -> Optional[Dict[str, Any]]:
        """実際の後処理を実行します"""
        try:
            # EventPredictorの後処理を呼び出し
            results = self.predictor.postprocess(task.tensor_data)
            
            if self.debug:
                logger.debug(f"EventWorker後処理結果: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"EventWorker後処理実行中にエラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def get_performance_stats(self) -> Dict[str, int]:
        """パフォーマンス統計を取得します"""
        return {
            "preprocess_count": self.preprocess_count,
            "inference_count": self.inference_count,
            "postprocess_count": self.postprocess_count,
            "sequence_length": len(self.sequence_frames),
            "buffer_size": len(self.result_buffer)
        } 