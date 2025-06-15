# streaming_annotator/workers/court_worker.py

import logging
import queue
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base_worker import BaseWorker, BatchTask
from ..core.interfaces import ItemId, TopicName, TaskData, ResultData

logger = logging.getLogger(__name__)


class CourtDetectionResult:
    """コート検出結果を格納するクラス"""
    
    def __init__(self, keypoints: Optional[List[tuple]] = None,
                 lines: Optional[List[tuple]] = None,
                 confidence: float = 0.0,
                 court_mask: Optional[np.ndarray] = None):
        """
        Args:
            keypoints: コートのキーポイント座標のリスト
            lines: コートのライン座標のリスト
            confidence: 検出信頼度
            court_mask: コート領域のマスク
        """
        self.keypoints = keypoints or []
        self.lines = lines or []
        self.confidence = confidence
        self.court_mask = court_mask
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "keypoints": self.keypoints,
            "lines": self.lines,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "has_mask": self.court_mask is not None,
            "keypoints_count": len(self.keypoints),
            "lines_count": len(self.lines)
        }


class CourtDetectionWorker(BaseWorker):
    """
    コート検出処理を行うワーカー。
    
    上流ワーカー（依存関係なし）として動作し、
    フレーム画像からテニスコートの構造を検出します。
    標準的なバッチ処理を実装。
    """
    
    def __init__(self, name: str, predictor: Any, results_queue: queue.Queue,
                 max_concurrent_tasks: int = 3, debug: bool = False,
                 batch_size: int = 4, batch_timeout: float = 0.1,
                 task_queue_maxsize: int = 1000):
        """
        Args:
            name: ワーカー名
            predictor: コート検出予測器
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
        self.court_stats = {
            "detections_made": 0,
            "courts_detected": 0,
            "average_confidence": 0.0,
            "average_keypoints": 0.0,
            "average_lines": 0.0
        }
    
    def get_published_topic(self) -> TopicName:
        """このワーカーが出版するトピック名を返します"""
        return "court_detection"
    
    def get_dependencies(self) -> List[TopicName]:
        """このワーカーの依存関係を返します（上流ワーカーなので空リスト）"""
        return []
    
    def process_batch(self, batch_tasks: List[BatchTask]) -> List[ResultData]:
        """
        バッチタスクの処理を実行します。
        
        標準的なバッチ処理を実装。
        
        Args:
            batch_tasks: バッチタスクのリスト
        
        Returns:
            List[ResultData]: 処理結果のリスト（batch_tasksと同じ順序）
        """
        try:
            # フレーム画像の検証とバッチ準備
            frames_batch = []
            valid_tasks = []
            
            for batch_task in batch_tasks:
                if batch_task.task_data is None:
                    logger.warning(f"Task data is None for item {batch_task.item_id}")
                    continue
                
                if not isinstance(batch_task.task_data, np.ndarray):
                    logger.warning(f"Expected numpy array, got {type(batch_task.task_data)} for item {batch_task.item_id}")
                    continue
                
                frames_batch.append(batch_task.task_data)
                valid_tasks.append(batch_task)
            
            if not frames_batch:
                # 有効なタスクがない場合はダミー結果を返す
                return [CourtDetectionResult(confidence=0.0) for _ in batch_tasks]
            
            # バッチ推論の実行
            batch_predictions = self._run_batch_inference(frames_batch)
            
            # 結果の処理
            results = []
            valid_idx = 0
            
            for batch_task in batch_tasks:
                if valid_idx < len(valid_tasks) and batch_task == valid_tasks[valid_idx]:
                    # 有効なタスクの場合
                    prediction = batch_predictions[valid_idx] if valid_idx < len(batch_predictions) else None
                    result = self._postprocess_prediction(prediction, batch_task.task_data.shape[:2])
                    valid_idx += 1
                else:
                    # 無効なタスクの場合はダミー結果
                    result = CourtDetectionResult(confidence=0.0)
                
                # 統計更新
                self._update_court_stats(result)
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Court detection batch processing error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー時はダミー結果を返す
            return [CourtDetectionResult(confidence=0.0) for _ in batch_tasks]
    
    def _run_batch_inference(self, frames_batch: List[np.ndarray]) -> List[Any]:
        """
        バッチ推論を実行します。
        
        Args:
            frames_batch: フレーム画像のバッチ
        
        Returns:
            推論結果のリスト
        """
        try:
            # 前処理
            processed_batch = self._preprocess_frames_batch(frames_batch)
            
            # バッチ推論
            with torch.no_grad():
                if hasattr(self.predictor, 'inference_batch'):
                    # バッチ推論専用メソッドがある場合
                    predictions = self.predictor.inference_batch(processed_batch)
                elif hasattr(self.predictor, 'inference'):
                    # 通常の推論メソッドを使用
                    predictions = self.predictor.inference(processed_batch)
                else:
                    raise AttributeError("Predictor has no inference method")
            
            # 予測結果がリストでない場合はリストに変換
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            return predictions
        
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            raise
    
    def _preprocess_frames_batch(self, frames_batch: List[np.ndarray]) -> Any:
        """
        フレームバッチの前処理を行います。
        
        Args:
            frames_batch: フレーム画像のバッチ
        
        Returns:
            前処理済みデータ
        """
        try:
            # predictorの前処理メソッドを呼び出し
            if hasattr(self.predictor, 'preprocess_batch'):
                return self.predictor.preprocess_batch(frames_batch)
            elif hasattr(self.predictor, 'preprocess'):
                return self.predictor.preprocess(frames_batch)
            else:
                # フォールバック: 基本的な前処理
                return self._basic_preprocess_batch(frames_batch)
        
        except Exception as e:
            logger.error(f"Batch preprocessing error: {e}")
            raise
    
    def _basic_preprocess_batch(self, frames_batch: List[np.ndarray]) -> List[np.ndarray]:
        """基本的なバッチ前処理を行います"""
        # フォールバック実装: 基本的な前処理を各フレームに適用
        processed_frames = []
        for frame in frames_batch:
            # 例: リサイズ、正規化など
            processed_frame = self._basic_preprocess_frame(frame)
            processed_frames.append(processed_frame)
        return processed_frames
    
    def _basic_preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """基本的なフレーム前処理を行います"""
        # 実際の実装は予測器の要件に応じて調整
        # 例: リサイズ、正規化など
        return frame
    
    def _postprocess_prediction(self, prediction: Any, frame_shape: tuple) -> CourtDetectionResult:
        """
        推論結果の後処理を行います。
        
        Args:
            prediction: 推論結果
            frame_shape: フレームの形状 (height, width)
        
        Returns:
            CourtDetectionResult: 処理済みの検出結果
        """
        try:
            if prediction is None:
                return CourtDetectionResult(confidence=0.0)
            
            # predictorの後処理メソッドを呼び出し
            if hasattr(self.predictor, 'postprocess'):
                processed_result = self.predictor.postprocess(prediction)
                
                # 結果をCourtDetectionResultに変換
                return self._convert_to_detection_result(processed_result, frame_shape)
            else:
                # フォールバック: 基本的な後処理
                return self._basic_postprocess(prediction, frame_shape)
        
        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            return CourtDetectionResult(confidence=0.0)
    
    def _convert_to_detection_result(self, processed_result: Any, frame_shape: tuple) -> CourtDetectionResult:
        """予測器の結果をCourtDetectionResultに変換します"""
        try:
            # 結果の形式に応じて変換
            if isinstance(processed_result, dict):
                return CourtDetectionResult(
                    keypoints=processed_result.get('keypoints', []),
                    lines=processed_result.get('lines', []),
                    confidence=processed_result.get('confidence', 0.0),
                    court_mask=processed_result.get('court_mask')
                )
            elif hasattr(processed_result, 'keypoints'):
                # オブジェクト形式の場合
                return CourtDetectionResult(
                    keypoints=getattr(processed_result, 'keypoints', []),
                    lines=getattr(processed_result, 'lines', []),
                    confidence=getattr(processed_result, 'confidence', 0.0),
                    court_mask=getattr(processed_result, 'court_mask', None)
                )
            else:
                # その他の形式の場合はダミー結果
                return CourtDetectionResult(confidence=0.0)
        
        except Exception as e:
            logger.warning(f"Error converting prediction result: {e}")
            return CourtDetectionResult(confidence=0.0)
    
    def _basic_postprocess(self, prediction: Any, frame_shape: tuple) -> CourtDetectionResult:
        """基本的な後処理を行います"""
        # フォールバック実装
        return CourtDetectionResult(confidence=0.0)
    
    def _update_court_stats(self, result: CourtDetectionResult) -> None:
        """コート検出統計を更新します"""
        with self.lock:
            self.court_stats["detections_made"] += 1
            
            if result.confidence > 0.5:  # 閾値は調整可能
                self.court_stats["courts_detected"] += 1
            
            # 平均信頼度の更新
            total = self.court_stats["detections_made"]
            current_avg = self.court_stats["average_confidence"]
            self.court_stats["average_confidence"] = \
                (current_avg * (total - 1) + result.confidence) / total
            
            # 平均キーポイント数の更新
            current_keypoints_avg = self.court_stats["average_keypoints"]
            self.court_stats["average_keypoints"] = \
                (current_keypoints_avg * (total - 1) + len(result.keypoints)) / total
            
            # 平均ライン数の更新
            current_lines_avg = self.court_stats["average_lines"]
            self.court_stats["average_lines"] = \
                (current_lines_avg * (total - 1) + len(result.lines)) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を返します（基底クラスの統計に追加）"""
        base_stats = super().get_stats()
        
        with self.lock:
            base_stats.update({
                "court_detections_made": self.court_stats["detections_made"],
                "courts_detected": self.court_stats["courts_detected"],
                "average_confidence": self.court_stats["average_confidence"],
                "average_keypoints": self.court_stats["average_keypoints"],
                "average_lines": self.court_stats["average_lines"],
                "detection_rate": (
                    self.court_stats["courts_detected"] / max(1, self.court_stats["detections_made"])
                )
            })
        
        return base_stats