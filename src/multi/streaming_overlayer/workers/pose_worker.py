import torch
import logging
import queue
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

from .base_worker import BaseWorker
from ..definitions import InferenceTask, PostprocessTask

logger = logging.getLogger(__name__)


class PoseWorker(BaseWorker):
    """
    プレイヤーポーズ検知のためのパイプラインワーカー。
    
    Detection と Pose の処理を分離し、GPU使用率を最大化する設計：
    1. Detection pipeline: preprocess → inference → postprocess
    2. Pose pipeline: preprocess → inference → postprocess
    3. 各段階は独立したスレッドで並列実行
    4. GPU inference は専用キューで効率的に処理
    """

    def __init__(self, name, predictor, queue_set, results_q, debug=False):
        # QueueManagerから基本キューを取得
        preprocess_q = queue_set.get_queue("preprocess")
        inference_q = queue_set.get_queue("inference")
        postprocess_q = queue_set.get_queue("postprocess")
        
        super().__init__(name, predictor, preprocess_q, inference_q, postprocess_q, results_q, debug)
        
        # 拡張キューをQueueManagerから取得
        self.detection_inference_queue = queue_set.get_queue("detection_inference")
        self.detection_postprocess_queue = queue_set.get_queue("detection_postprocess")
        self.pose_inference_queue = queue_set.get_queue("pose_inference")
        self.pose_postprocess_queue = queue_set.get_queue("pose_postprocess")
        
        # キューの存在確認
        required_queues = [
            ("detection_inference", self.detection_inference_queue),
            ("detection_postprocess", self.detection_postprocess_queue),
            ("pose_inference", self.pose_inference_queue),
            ("pose_postprocess", self.pose_postprocess_queue)
        ]
        
        for queue_name, q in required_queues:
            if q is None:
                raise ValueError(f"Required extended queue '{queue_name}' not found for {name} worker")
        
        # 追加スレッドのリスト
        self.additional_threads = []
        
        # スレッドプール（前処理・後処理用）
        self.detection_preprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_det_preprocess")
        self.detection_postprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_det_postprocess")
        self.pose_preprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_pose_preprocess")
        self.pose_postprocess_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{name}_pose_postprocess")
        
        # パフォーマンス監視メトリクス
        self.metrics = {
            "detection_preprocess_count": 0,
            "detection_inference_count": 0,
            "detection_postprocess_count": 0,
            "pose_inference_count": 0,
            "pose_postprocess_count": 0,
            "total_frames_processed": 0,
            "total_players_detected": 0,
            "detection_inference_time": [],
            "pose_inference_time": [],
            "queue_sizes": {
                "detection_inference": [],
                "detection_postprocess": [],
                "pose_inference": [],
                "pose_postprocess": []
            }
        }

    def start(self):
        """
        ワーカーを開始し、detection と pose の独立したパイプラインを構築します。
        """
        if self.running:
            logger.warning(f"{self.name} worker is already running")
            return
            
        self.running = True
        
        # 基本的な3つのスレッド（継承元のBaseWorkerで管理）
        base_threads = [
            threading.Thread(target=self._preprocess_loop, name=f"{self.name}_preprocess"),
            threading.Thread(target=self._inference_loop, name=f"{self.name}_inference"),
            threading.Thread(target=self._postprocess_loop, name=f"{self.name}_postprocess"),
        ]
        
        # Detection専用パイプライン
        detection_threads = [
            threading.Thread(target=self._detection_inference_loop, name=f"{self.name}_detection_inference"),
            threading.Thread(target=self._detection_postprocess_loop, name=f"{self.name}_detection_postprocess"),
        ]
        
        # Pose専用パイプライン
        pose_threads = [
            threading.Thread(target=self._pose_inference_loop, name=f"{self.name}_pose_inference"),
            threading.Thread(target=self._pose_postprocess_loop, name=f"{self.name}_pose_postprocess"),
        ]
        
        # 全スレッドを開始
        all_threads = base_threads + detection_threads + pose_threads
        for thread in all_threads:
            thread.daemon = True
            thread.start()
            
        self.threads = base_threads
        self.additional_threads = detection_threads + pose_threads
            
        logger.info(f"Started {self.name} worker with optimized pipeline ({len(all_threads)} threads)")

    def stop(self):
        """ワーカーを停止し、すべてのスレッドを終了します。"""
        if not self.running:
            return
            
        self.running = False
        
        # スレッドプールをシャットダウン
        pools = [
            ('detection_preprocess_pool', self.detection_preprocess_pool),
            ('detection_postprocess_pool', self.detection_postprocess_pool),
            ('pose_preprocess_pool', self.pose_preprocess_pool),
            ('pose_postprocess_pool', self.pose_postprocess_pool)
        ]
        for name, pool in pools:
            if hasattr(self, name):
                try:
                    pool.shutdown(wait=False)
                except Exception as e:
                    logger.warning(f"Error shutting down {name}: {e}")
        
        # 基本スレッドの停止
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        # 追加スレッドの停止
        for thread in self.additional_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        self.threads.clear()
        self.additional_threads.clear()
        logger.info(f"Stopped {self.name} worker")

    def _process_preprocess_task(self, task):
        """
        前処理タスクを処理し、Detection の前処理を実行します。
        
        Args:
            task: PreprocessTask - フレームデータとメタデータを含む
        """
        try:
            # Detection の前処理を実行
            det_inputs = self.predictor.preprocess_detection(task.frames)
            
            # Detection 推論キューに送信
            detection_task = InferenceTask(
                task.task_id + "_detection", 
                det_inputs, 
                task.meta_data,
                original_frames=task.frames
            )
            self.detection_inference_queue.put(detection_task)
            
            if self.debug:
                logger.debug(f"PoseWorker Detection前処理完了: {task.task_id}, frames={len(task.frames)}")
                
        except Exception as e:
            logger.error(f"PoseWorker Detection前処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_inference_task(self, task):
        """
        基本の推論タスクは使用しません（Detection/Pose専用キューを使用）。
        """
        pass

    def _process_postprocess_task(self, task):
        """
        基本の後処理タスクは使用しません（Detection/Pose専用キューを使用）。
        """
        pass

    def _detection_inference_loop(self):
        """Detection推論ループ - GPU使用率最大化のため専用スレッド"""
        while self.running:
            try:
                task = self.detection_inference_queue.get(timeout=0.1)
                self._process_detection_inference(task)
                self.detection_inference_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} detection inference error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _detection_postprocess_loop(self):
        """Detection後処理ループ - 独立したスレッドで並列実行"""
        while self.running:
            try:
                task = self.detection_postprocess_queue.get(timeout=0.1)
                self._process_detection_postprocess(task)
                self.detection_postprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} detection postprocess error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _pose_inference_loop(self):
        """Pose推論ループ - GPU使用率最大化のため専用スレッド"""
        while self.running:
            try:
                task = self.pose_inference_queue.get(timeout=0.1)
                self._process_pose_inference(task)
                self.pose_inference_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} pose inference error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _pose_postprocess_loop(self):
        """Pose後処理ループ - 独立したスレッドで並列実行"""
        while self.running:
            try:
                task = self.pose_postprocess_queue.get(timeout=0.1)
                self._process_pose_postprocess(task)
                self.pose_postprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} pose postprocess error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _process_detection_inference(self, task):
        """
        Detection 推論を実行します。
        
        Args:
            task: InferenceTask - Detection用前処理済みデータを含む
        """
        try:
            start_time = time.time()
            
            with torch.no_grad():
                # Detection 推論を実行
                det_outputs = self.predictor.inference_detection(task.tensor_data)
            
            # パフォーマンスメトリクス更新
            inference_time = time.time() - start_time
            self.metrics["detection_inference_count"] += 1
            self.metrics["detection_inference_time"].append(inference_time)
            
            # Detection 後処理キューに送信
            detection_post_task = PostprocessTask(
                task.task_id,
                det_outputs,
                task.meta_data,
                original_frames=task.original_frames
            )
            self.detection_postprocess_queue.put(detection_post_task)
            
            if self.debug:
                logger.debug(f"PoseWorker Detection推論完了: {task.task_id}, time={inference_time:.3f}s")
                
        except Exception as e:
            logger.error(f"PoseWorker Detection推論エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_detection_postprocess(self, task):
        """
        Detection 後処理を実行し、Pose 前処理を開始します。
        
        Args:
            task: PostprocessTask - Detection推論結果を含む
        """
        try:
            # Detection 後処理を実行
            det_outputs = task.inference_output
            batch_boxes, batch_scores, batch_valid, images_for_pose = self.predictor.postprocess_detection(
                det_outputs, task.original_frames
            )
            
            if not images_for_pose:
                # プレーヤー検出なしの場合、空結果を返す
                self._send_empty_results(task.meta_data)
                return
            
            # Pose 前処理を実行
            pose_inputs = self.predictor.preprocess_pose(images_for_pose, batch_boxes)
            
            # Pose 推論キューに送信
            pose_task = InferenceTask(
                task.task_id.replace("_detection", "_pose"),
                pose_inputs,
                task.meta_data,
                batch_boxes=batch_boxes,
                batch_scores=batch_scores,
                batch_valid=batch_valid,
                original_frames=task.original_frames
            )
            self.pose_inference_queue.put(pose_task)
            
            if self.debug:
                logger.debug(f"PoseWorker Detection後処理完了: {task.task_id}, detected_players={len(images_for_pose)}")
                
        except Exception as e:
            logger.error(f"PoseWorker Detection後処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_pose_inference(self, task):
        """
        Pose 推論を実行します。
        
        Args:
            task: InferenceTask - Pose用前処理済みデータを含む
        """
        try:
            start_time = time.time()
            
            with torch.no_grad():
                # Pose 推論を実行
                pose_outputs = self.predictor.inference_pose(task.tensor_data)
            
            # パフォーマンスメトリクス更新
            inference_time = time.time() - start_time
            self.metrics["pose_inference_count"] += 1
            self.metrics["pose_inference_time"].append(inference_time)
            
            # Pose 後処理キューに送信
            pose_post_task = PostprocessTask(
                task.task_id,
                pose_outputs,
                task.meta_data,
                batch_boxes=task.batch_boxes,
                batch_scores=task.batch_scores,
                batch_valid=task.batch_valid,
                original_frames=task.original_frames
            )
            self.pose_postprocess_queue.put(pose_post_task)
            
            if self.debug:
                logger.debug(f"PoseWorker Pose推論完了: {task.task_id}, time={inference_time:.3f}s")
                
        except Exception as e:
            logger.error(f"PoseWorker Pose推論エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _process_pose_postprocess(self, task):
        """
        Pose 後処理を実行し、最終結果を出力します。
        
        Args:
            task: PostprocessTask - Pose推論結果を含む
        """
        try:
            # Pose 後処理を実行
            pose_outputs = task.inference_output
            pose_results = self.predictor.postprocess_pose(
                pose_outputs,
                task.batch_boxes,
                task.batch_scores,
                task.batch_valid,
                len(task.original_frames)
            )
            
            # 結果を検証して結果キューに送信
            self._send_pose_results(pose_results, task.meta_data)
            
            if self.debug:
                total_detections = sum(len(frame_poses) for frame_poses in pose_results)
                logger.debug(f"PoseWorker Pose後処理完了: {task.task_id}, total_detections={total_detections}")
                
        except Exception as e:
            logger.error(f"PoseWorker Pose後処理エラー: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _send_empty_results(self, meta_data):
        """プレーヤー検出なしの場合の空結果を送信"""
        if not meta_data:
            return
            
        for i, meta in enumerate(meta_data):
            if isinstance(meta, (list, tuple)) and len(meta) >= 1:
                frame_idx = meta[0]
            else:
                frame_idx = i
            
            self.results_queue.put((frame_idx, "pose", []))

    def _send_pose_results(self, pose_results, meta_data):
        """検証済みのポーズ結果を送信"""
        total_players_in_batch = 0
        
        for frame_idx, pose_result_per_frame in enumerate(pose_results):
            # メタデータからフレームインデックスを取得
            if meta_data and frame_idx < len(meta_data):
                if isinstance(meta_data[frame_idx], (list, tuple)) and len(meta_data[frame_idx]) >= 1:
                    actual_frame_idx = meta_data[frame_idx][0]
                else:
                    actual_frame_idx = frame_idx
            else:
                actual_frame_idx = frame_idx
            
            # 結果の妥当性チェック
            validated_results = self._validate_pose_results(pose_result_per_frame)
            player_count = len(validated_results) if validated_results else 0
            total_players_in_batch += player_count
            
            # 結果をキューに追加
            self.results_queue.put((actual_frame_idx, "pose", validated_results))
            
            if self.debug:
                logger.debug(f"PoseWorker: フレーム{actual_frame_idx}でプレイヤー{player_count}人検出")
        
        # パフォーマンスメトリクス更新
        self.metrics["total_frames_processed"] += len(pose_results)
        self.metrics["total_players_detected"] += total_players_in_batch

    def _validate_pose_results(self, pose_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ポーズ検出結果の妥当性をチェックし、不正な値を修正します。
        
        Args:
            pose_results: フレーム内のプレイヤー検出結果のリスト
            
        Returns:
            検証済みの検出結果のリスト
        """
        if not pose_results:
            return []
        
        validated_results = []
        
        for player_data in pose_results:
            try:
                if not isinstance(player_data, dict):
                    logger.warning(f"PoseWorker: 不正なプレイヤーデータ形式: {type(player_data)}")
                    continue
                
                # 必須フィールドの存在確認
                required_fields = ["bbox", "keypoints", "scores"]
                if not all(field in player_data for field in required_fields):
                    logger.warning(f"PoseWorker: 必須フィールドが不足: {list(player_data.keys())}")
                    continue
                
                # バウンディングボックスの妥当性チェック
                bbox = player_data.get("bbox", [])
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    logger.warning(f"PoseWorker: 不正なbbox形式: {bbox}")
                    continue
                
                # キーポイントの妥当性チェック
                keypoints = player_data.get("keypoints", [])
                scores = player_data.get("scores", [])
                
                if len(keypoints) != len(scores):
                    logger.warning(f"PoseWorker: キーポイントとスコア数不一致: {len(keypoints)} vs {len(scores)}")
                    # より短い方に合わせる
                    min_len = min(len(keypoints), len(scores))
                    keypoints = keypoints[:min_len]
                    scores = scores[:min_len]
                
                # 検証済みデータを作成
                validated_data = {
                    "bbox": [int(x) for x in bbox],  # 整数に変換
                    "det_score": float(player_data.get("det_score", 0.0)),
                    "keypoints": [(int(x), int(y)) for x, y in keypoints],  # 整数タプルに変換
                    "scores": [float(s) for s in scores],  # float型に変換
                }
                
                validated_results.append(validated_data)
                
            except Exception as e:
                logger.warning(f"PoseWorker: プレイヤーデータ検証エラー: {e}")
                continue
        
        return validated_results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        パフォーマンスメトリクスを取得します。
        
        Returns:
            現在のパフォーマンス統計
        """
        current_queue_sizes = {
            "detection_inference": self.detection_inference_queue.qsize(),
            "detection_postprocess": self.detection_postprocess_queue.qsize(),
            "pose_inference": self.pose_inference_queue.qsize(),
            "pose_postprocess": self.pose_postprocess_queue.qsize(),
            "preprocess": self.preprocess_queue.qsize(),
            "inference": self.inference_queue.qsize(),
            "postprocess": self.postprocess_queue.qsize(),
            "results": self.results_queue.qsize()
        }
        
        metrics_copy = self.metrics.copy()
        metrics_copy["current_queue_sizes"] = current_queue_sizes
        
        # 平均処理時間を計算
        if self.metrics["detection_inference_time"]:
            metrics_copy["avg_detection_inference_time"] = np.mean(self.metrics["detection_inference_time"])
        else:
            metrics_copy["avg_detection_inference_time"] = 0.0
            
        if self.metrics["pose_inference_time"]:
            metrics_copy["avg_pose_inference_time"] = np.mean(self.metrics["pose_inference_time"])
        else:
            metrics_copy["avg_pose_inference_time"] = 0.0
        
        # スループット計算
        total_processed = self.metrics["total_frames_processed"]
        if total_processed > 0:
            metrics_copy["avg_players_per_frame"] = self.metrics["total_players_detected"] / total_processed
        else:
            metrics_copy["avg_players_per_frame"] = 0.0
        
        return metrics_copy

    def reset_metrics(self):
        """パフォーマンスメトリクスをリセットします。"""
        self.metrics = {
            "detection_preprocess_count": 0,
            "detection_inference_count": 0,
            "detection_postprocess_count": 0,
            "pose_inference_count": 0,
            "pose_postprocess_count": 0,
            "total_frames_processed": 0,
            "total_players_detected": 0,
            "detection_inference_time": [],
            "pose_inference_time": [],
            "queue_sizes": {
                "detection_inference": [],
                "detection_postprocess": [],
                "pose_inference": [],
                "pose_postprocess": []
            }
        }
