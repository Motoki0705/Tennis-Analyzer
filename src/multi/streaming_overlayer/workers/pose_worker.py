# streaming_annotator/workers/pose_worker.py

import logging
import queue
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base_worker import BaseWorker, BatchTask
from ..core.interfaces import ItemId, TopicName, TaskData, ResultData

logger = logging.getLogger(__name__)


class PoseData:
    """単一の人物のポーズデータを格納するクラス"""
    def __init__(self, bbox: List[float], det_score: float, 
                 keypoints: List[tuple], scores: List[float]):
        self.bbox = bbox
        self.det_score = det_score
        self.keypoints = keypoints
        self.scores = scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "det_score": self.det_score,
            "keypoints": self.keypoints,
            "scores": self.scores
        }


class PoseDetectionResult:
    """ポーズ検出結果を格納するクラス"""
    
    def __init__(self, poses: Optional[List[PoseData]] = None, 
                 player_count: int = 0,
                 confidence: float = 0.0):
        """
        Args:
            poses: 検出されたポーズのリスト
            player_count: 検出されたプレイヤー数
            confidence: 検出の全体的な信頼度（例: 平均スコア）
        """
        self.poses = poses or []
        self.player_count = player_count
        self.confidence = confidence
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "poses": [p.to_dict() for p in self.poses],
            "player_count": self.player_count,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class PoseDetectionWorker(BaseWorker):
    """
    ポーズ検出処理を行うワーカー。
    
    新しいBaseWorkerアーキテクチャに準拠した実装。
    内部で「Detection -> Pose」の2段階処理を実行します。
    """
    
    def __init__(self, name: str, predictor: Any, results_queue: queue.Queue,
                 max_concurrent_tasks: int = 2, debug: bool = False,
                 batch_size: int = 4, batch_timeout: float = 0.1,
                 task_queue_maxsize: int = 1000):
        """
        Args:
            name: ワーカー名
            predictor: ポーズ検出予測器 (detectionとposeのメソッドを持つ)
            results_queue: 結果出力キュー
            max_concurrent_tasks: 最大同時実行タスク数
            debug: デバッグモード
            batch_size: バッチサイズ
            batch_timeout: バッチタイムアウト
            task_queue_maxsize: タスクキューの最大サイズ
        """
        super().__init__(name, results_queue, max_concurrent_tasks, debug,
                        batch_size, batch_timeout, task_queue_maxsize)
        
        self.predictor = predictor
        
        # 処理統計の拡張
        self.pose_stats = {
            "detections_made": 0,
            "total_players_detected": 0,
            "average_players_per_frame": 0.0,
        }
    
    def get_published_topic(self) -> TopicName:
        """このワーカーが出版するトピック名を返します"""
        return "pose_detection"
    
    def get_dependencies(self) -> List[TopicName]:
        """このワーカーの依存関係を返します（上流ワーカーなので空リスト）"""
        return []

    def process_batch(self, batch_tasks: List[BatchTask]) -> List[ResultData]:
        """
        バッチタスクの処理を実行します。
        
        1. バッチ内の全フレームに対してDetection推論を実行。
        2. 検出された人物領域を収集。
        3. 収集した人物領域に対してPose推論をバッチで実行。
        4. 結果を元のフレームにマッピングして返す。
        
        Args:
            batch_tasks: バッチタスクのリスト
        
        Returns:
            List[ResultData]: 処理結果のリスト（batch_tasksと同じ順序）
        """
        if not batch_tasks:
            return []

        try:
            frames_batch = [task.task_data for task in batch_tasks]
            
            # --- 1. Detection Stage ---
            all_boxes, all_scores, all_valid, images_for_pose, frame_indices_for_pose = self._run_detection_stage(frames_batch)

            # --- 2. Pose Stage ---
            if images_for_pose:
                pose_results_flat = self._run_pose_stage(images_for_pose, all_boxes, all_scores, all_valid)
            else:
                pose_results_flat = []

            # --- 3. Map results back to original frames ---
            final_results = self._map_poses_to_frames(pose_results_flat, frame_indices_for_pose, len(frames_batch))

            # 統計更新
            self._update_pose_stats(final_results)

            return final_results

        except Exception as e:
            logger.error(f"{self.name} batch processing error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            # エラー時はダミー結果を返す
            return [PoseDetectionResult() for _ in batch_tasks]

    def _run_detection_stage(self, frames_batch: List[np.ndarray]) -> tuple:
        """Detection推論のステージを実行"""
        with torch.no_grad():
            # Preprocess
            det_inputs = self.predictor.preprocess_detection(frames_batch)
            # Inference
            det_outputs = self.predictor.inference_detection(det_inputs)
            # Postprocess
            # この後処理は、Pose推論に必要な情報を抽出する
            batch_boxes, batch_scores, batch_valid, images_for_pose, frame_indices_for_pose = \
                self.predictor.postprocess_detection(det_outputs, frames_batch)
        
        return batch_boxes, batch_scores, batch_valid, images_for_pose, frame_indices_for_pose

    def _run_pose_stage(self, images_for_pose: List, all_boxes: List, all_scores: List, all_valid: List) -> List:
        """Pose推論のステージを実行"""
        with torch.no_grad():
            # Preprocess
            pose_inputs = self.predictor.preprocess_pose(images_for_pose, all_boxes)
            # Inference
            pose_outputs = self.predictor.inference_pose(pose_inputs)
            # Postprocess
            # この後処理は、最終的なポーズデータを生成する
            pose_results_flat = self.predictor.postprocess_pose(
                pose_outputs, all_boxes, all_scores, all_valid
            )
        
        return pose_results_flat

    def _map_poses_to_frames(self, pose_results_flat: List, frame_indices_for_pose: List, batch_size: int) -> List[PoseDetectionResult]:
        """検出されたポーズを元のフレームにマッピングし、PoseDetectionResultを作成する"""
        
        # 結果を格納するリストを初期化
        final_results_by_frame = [[] for _ in range(batch_size)]
        
        # 検出された各ポーズを対応するフレームに割り当てる
        for pose_data, original_frame_idx in zip(pose_results_flat, frame_indices_for_pose):
            if original_frame_idx < batch_size:
                final_results_by_frame[original_frame_idx].append(pose_data)
        
        # PoseDetectionResultオブジェクトのリストを作成
        output_results = []
        for frame_poses in final_results_by_frame:
            validated_poses = self._validate_and_convert_poses(frame_poses)
            player_count = len(validated_poses)
            
            # 平均信頼度を計算 (簡易版)
            avg_confidence = 0.0
            if player_count > 0:
                total_score = sum(p.det_score for p in validated_poses)
                avg_confidence = total_score / player_count

            output_results.append(
                PoseDetectionResult(
                    poses=validated_poses,
                    player_count=player_count,
                    confidence=avg_confidence
                )
            )
            
        return output_results

    def _validate_and_convert_poses(self, poses: List[Dict]) -> List[PoseData]:
        """Predictorからの辞書リストをPoseDataオブジェクトのリストに変換・検証する"""
        validated_list = []
        for pose_dict in poses:
            try:
                # 辞書の形式を検証
                if not all(k in pose_dict for k in ["bbox", "keypoints", "scores", "det_score"]):
                    logger.warning(f"Skipping invalid pose dict: {pose_dict}")
                    continue
                
                # PoseDataオブジェクトを作成
                pose_data_obj = PoseData(
                    bbox=pose_dict["bbox"],
                    det_score=float(pose_dict["det_score"]),
                    keypoints=pose_dict["keypoints"],
                    scores=pose_dict["scores"]
                )
                validated_list.append(pose_data_obj)

            except (TypeError, KeyError) as e:
                logger.warning(f"Error converting pose dict to object: {e}")
                continue
        return validated_list

    def _update_pose_stats(self, results: List[PoseDetectionResult]) -> None:
        """ポーズ検出統計を更新します"""
        with self.lock:
            num_frames = len(results)
            players_in_batch = sum(res.player_count for res in results)
            
            total_processed = self.pose_stats["detections_made"]
            total_players = self.pose_stats["total_players_detected"]
            
            self.pose_stats["detections_made"] += num_frames
            self.pose_stats["total_players_detected"] += players_in_batch
            
            # 平均プレイヤー数を更新
            new_total_processed = total_processed + num_frames
            if new_total_processed > 0:
                self.pose_stats["average_players_per_frame"] = \
                    (total_players + players_in_batch) / new_total_processed

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を返します（基底クラスの統計に追加）"""
        base_stats = super().get_stats()
        
        with self.lock:
            base_stats.update(self.pose_stats)
        
        return base_stats