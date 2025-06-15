"""
高度拡張性を持つストリーミング処理パイプライン - EventWorker実装

イベント検出処理を行うワーカー。
新しいアーキテクチャに対応した実装。
依存関係を持つ下流ワーカーとしてのバッチ処理を実装。
"""

import logging
import queue
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base_worker import BaseWorker, BatchTask
from ..core.interfaces import ItemId, TopicName, TaskData, ResultData

logger = logging.getLogger(__name__)


class EventDetectionResult:
    """イベント検出結果を格納するクラス"""
    
    def __init__(self, event_type: Optional[str] = None,
                 confidence: float = 0.0,
                 timestamp: Optional[float] = None,
                 ball_position: Optional[tuple] = None,
                 court_region: Optional[str] = None,
                 player_positions: Optional[List[tuple]] = None):
        """
        Args:
            event_type: 検出されたイベントタイプ（例: "serve", "hit", "bounce"）
            confidence: 検出信頼度
            timestamp: イベント発生時刻
            ball_position: イベント時のボール位置
            court_region: イベント発生コート領域
            player_positions: イベント時のプレイヤー位置
        """
        self.event_type = event_type
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        self.ball_position = ball_position
        self.court_region = court_region
        self.player_positions = player_positions or []
        self.detection_timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "event_type": self.event_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "ball_position": self.ball_position,
            "court_region": self.court_region,
            "player_positions": self.player_positions,
            "detection_timestamp": self.detection_timestamp,
            "has_event": self.event_type is not None
        }


class EventDetectionWorker(BaseWorker):
    """
    イベント検出処理を行うワーカー。
    
    下流ワーカーとして動作し、ボール検出、コート検出、
    ポーズ検出の結果を統合してテニスイベントを検出します。
    依存関係を持つ下流ワーカーとしてのバッチ処理を実装。
    """
    
    def __init__(self, name: str, predictor: Any, results_queue: queue.Queue,
                 max_concurrent_tasks: int = 3, debug: bool = False,
                 batch_size: int = 2, batch_timeout: float = 0.2,
                 task_queue_maxsize: int = 1000):
        """
        Args:
            name: ワーカー名
            predictor: イベント検出予測器
            results_queue: 結果出力キュー
            max_concurrent_tasks: 最大同時実行タスク数
            debug: デバッグモード
            batch_size: バッチサイズ（複雑な処理のため小さめ）
            batch_timeout: バッチタイムアウト（長めに設定）
            task_queue_maxsize: タスクキューの最大サイズ
        """
        super().__init__(name, results_queue, max_concurrent_tasks, debug,
                        batch_size, batch_timeout, task_queue_maxsize)
        
        self.predictor = predictor
        
        # 処理統計の拡張
        self.event_stats = {
            "detections_made": 0,
            "events_detected": 0,
            "event_types": {},  # イベントタイプ別カウント
            "average_confidence": 0.0,
            "missing_dependencies": 0
        }
    
    def get_published_topic(self) -> TopicName:
        """このワーカーが出版するトピック名を返します"""
        return "event_detection"
    
    def get_dependencies(self) -> List[TopicName]:
        """このワーカーの依存関係を返します"""
        return ["ball_detection", "court_detection", "pose_detection"]
    
    def process_batch(self, batch_tasks: List[BatchTask]) -> List[ResultData]:
        """
        バッチタスクの処理を実行します。
        
        依存関係を持つ下流ワーカーとしてのバッチ処理を実装。
        
        Args:
            batch_tasks: バッチタスクのリスト
        
        Returns:
            List[ResultData]: 処理結果のリスト（batch_tasksと同じ順序）
        """
        try:
            # 依存関係の検証とバッチ準備
            valid_batch_data = []
            valid_tasks = []
            
            for batch_task in batch_tasks:
                # 依存関係の検証
                if not self._validate_dependencies(batch_task.dependencies):
                    logger.warning(f"Missing dependencies for item {batch_task.item_id}")
                    with self.lock:
                        self.event_stats["missing_dependencies"] += 1
                    continue
                
                # フレーム画像の検証
                if batch_task.task_data is None or not isinstance(batch_task.task_data, np.ndarray):
                    logger.warning(f"Invalid task data for item {batch_task.item_id}")
                    continue
                
                # バッチデータの準備
                batch_data = self._prepare_batch_data(batch_task)
                if batch_data is not None:
                    valid_batch_data.append(batch_data)
                    valid_tasks.append(batch_task)
            
            if not valid_batch_data:
                # 有効なタスクがない場合はダミー結果を返す
                return [EventDetectionResult(confidence=0.0) for _ in batch_tasks]
            
            # バッチ推論の実行
            batch_predictions = self._run_batch_inference(valid_batch_data)
            
            # 結果の処理
            results = []
            valid_idx = 0
            
            for batch_task in batch_tasks:
                if valid_idx < len(valid_tasks) and batch_task == valid_tasks[valid_idx]:
                    # 有効なタスクの場合
                    prediction = batch_predictions[valid_idx] if valid_idx < len(batch_predictions) else None
                    result = self._postprocess_prediction(prediction, batch_task)
                    valid_idx += 1
                else:
                    # 無効なタスクの場合はダミー結果
                    result = EventDetectionResult(confidence=0.0)
                
                # 統計更新
                self._update_event_stats(result)
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Event detection batch processing error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー時はダミー結果を返す
            return [EventDetectionResult(confidence=0.0) for _ in batch_tasks]
    
    def _validate_dependencies(self, dependencies: Dict[TopicName, ResultData]) -> bool:
        """
        依存関係の検証を行います。
        
        Args:
            dependencies: 依存するトピックの結果辞書
        
        Returns:
            bool: 依存関係が満たされているかどうか
        """
        required_topics = self.get_dependencies()
        
        for topic in required_topics:
            if topic not in dependencies:
                logger.debug(f"Missing dependency: {topic}")
                return False
            
            # 結果の有効性をチェック
            result = dependencies[topic]
            if result is None:
                logger.debug(f"Null result for dependency: {topic}")
                return False
            
            # エラー結果のチェック
            if isinstance(result, dict) and result.get("error", False):
                logger.debug(f"Error result for dependency: {topic}")
                return False
        
        return True
    
    def _prepare_batch_data(self, batch_task: BatchTask) -> Optional[Dict[str, Any]]:
        """
        バッチ処理用のデータを準備します。
        
        Args:
            batch_task: バッチタスク
        
        Returns:
            準備されたバッチデータ（無効な場合はNone）
        """
        try:
            # フレーム画像
            frame = batch_task.task_data
            
            # 依存関係の結果を取得
            ball_result = batch_task.dependencies.get("ball_detection")
            court_result = batch_task.dependencies.get("court_detection")
            pose_result = batch_task.dependencies.get("pose_detection")
            
            # バッチデータの構築
            batch_data = {
                "item_id": batch_task.item_id,
                "frame": frame,
                "ball_detection": self._extract_ball_features(ball_result),
                "court_detection": self._extract_court_features(court_result),
                "pose_detection": self._extract_pose_features(pose_result),
                "timestamp": batch_task.submit_time
            }
            
            return batch_data
        
        except Exception as e:
            logger.error(f"Error preparing batch data for item {batch_task.item_id}: {e}")
            return None
    
    def _extract_ball_features(self, ball_result: Any) -> Dict[str, Any]:
        """ボール検出結果から特徴を抽出します"""
        try:
            if hasattr(ball_result, 'position'):
                return {
                    "position": ball_result.position,
                    "confidence": getattr(ball_result, 'confidence', 0.0),
                    "visibility": getattr(ball_result, 'visibility', False)
                }
            elif isinstance(ball_result, dict):
                return {
                    "position": ball_result.get('position'),
                    "confidence": ball_result.get('confidence', 0.0),
                    "visibility": ball_result.get('visibility', False)
                }
            else:
                return {"position": None, "confidence": 0.0, "visibility": False}
        except Exception as e:
            logger.warning(f"Error extracting ball features: {e}")
            return {"position": None, "confidence": 0.0, "visibility": False}
    
    def _extract_court_features(self, court_result: Any) -> Dict[str, Any]:
        """コート検出結果から特徴を抽出します"""
        try:
            if hasattr(court_result, 'keypoints'):
                return {
                    "keypoints": court_result.keypoints,
                    "lines": getattr(court_result, 'lines', []),
                    "confidence": getattr(court_result, 'confidence', 0.0)
                }
            elif isinstance(court_result, dict):
                return {
                    "keypoints": court_result.get('keypoints', []),
                    "lines": court_result.get('lines', []),
                    "confidence": court_result.get('confidence', 0.0)
                }
            else:
                return {"keypoints": [], "lines": [], "confidence": 0.0}
        except Exception as e:
            logger.warning(f"Error extracting court features: {e}")
            return {"keypoints": [], "lines": [], "confidence": 0.0}
    
    def _extract_pose_features(self, pose_result: Any) -> Dict[str, Any]:
        """ポーズ検出結果から特徴を抽出します"""
        try:
            # ポーズ検出結果の形式に応じて調整
            if hasattr(pose_result, 'keypoints'):
                return {
                    "keypoints": pose_result.keypoints,
                    "confidence": getattr(pose_result, 'confidence', 0.0),
                    "player_count": getattr(pose_result, 'player_count', 0)
                }
            elif isinstance(pose_result, dict):
                return {
                    "keypoints": pose_result.get('keypoints', []),
                    "confidence": pose_result.get('confidence', 0.0),
                    "player_count": pose_result.get('player_count', 0)
                }
            else:
                return {"keypoints": [], "confidence": 0.0, "player_count": 0}
        except Exception as e:
            logger.warning(f"Error extracting pose features: {e}")
            return {"keypoints": [], "confidence": 0.0, "player_count": 0}
    
    def _run_batch_inference(self, batch_data: List[Dict[str, Any]]) -> List[Any]:
        """
        バッチ推論を実行します。
        
        Args:
            batch_data: バッチデータのリスト
        
        Returns:
            推論結果のリスト
        """
        try:
            # 前処理
            processed_batch = self._preprocess_batch_data(batch_data)
            
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
    
    def _preprocess_batch_data(self, batch_data: List[Dict[str, Any]]) -> Any:
        """
        バッチデータの前処理を行います。
        
        Args:
            batch_data: バッチデータのリスト
        
        Returns:
            前処理済みデータ
        """
        try:
            # predictorの前処理メソッドを呼び出し
            if hasattr(self.predictor, 'preprocess_batch'):
                return self.predictor.preprocess_batch(batch_data)
            elif hasattr(self.predictor, 'preprocess'):
                return self.predictor.preprocess(batch_data)
            else:
                # フォールバック: 基本的な前処理
                return self._basic_preprocess_batch(batch_data)
        
        except Exception as e:
            logger.error(f"Batch preprocessing error: {e}")
            raise
    
    def _basic_preprocess_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基本的なバッチ前処理を行います"""
        # フォールバック実装: そのまま返す
        return batch_data
    
    def _postprocess_prediction(self, prediction: Any, batch_task: BatchTask) -> EventDetectionResult:
        """
        推論結果の後処理を行います。
        
        Args:
            prediction: 推論結果
            batch_task: 元のバッチタスク
        
        Returns:
            EventDetectionResult: 処理済みの検出結果
        """
        try:
            if prediction is None:
                return EventDetectionResult(confidence=0.0)
            
            # predictorの後処理メソッドを呼び出し
            if hasattr(self.predictor, 'postprocess'):
                processed_result = self.predictor.postprocess(prediction)
                
                # 結果をEventDetectionResultに変換
                return self._convert_to_detection_result(processed_result, batch_task)
            else:
                # フォールバック: 基本的な後処理
                return self._basic_postprocess(prediction, batch_task)
        
        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            return EventDetectionResult(confidence=0.0)
    
    def _convert_to_detection_result(self, processed_result: Any, batch_task: BatchTask) -> EventDetectionResult:
        """予測器の結果をEventDetectionResultに変換します"""
        try:
            # 依存関係から追加情報を取得
            ball_result = batch_task.dependencies.get("ball_detection")
            court_result = batch_task.dependencies.get("court_detection")
            pose_result = batch_task.dependencies.get("pose_detection")
            
            # ボール位置の取得
            ball_position = None
            if hasattr(ball_result, 'position'):
                ball_position = ball_result.position
            elif isinstance(ball_result, dict):
                ball_position = ball_result.get('position')
            
            # プレイヤー位置の取得
            player_positions = []
            if hasattr(pose_result, 'keypoints'):
                # ポーズキーポイントからプレイヤー位置を推定
                player_positions = self._extract_player_positions(pose_result.keypoints)
            elif isinstance(pose_result, dict):
                keypoints = pose_result.get('keypoints', [])
                player_positions = self._extract_player_positions(keypoints)
            
            # 結果の形式に応じて変換
            if isinstance(processed_result, dict):
                return EventDetectionResult(
                    event_type=processed_result.get('event_type'),
                    confidence=processed_result.get('confidence', 0.0),
                    timestamp=processed_result.get('timestamp', batch_task.submit_time),
                    ball_position=ball_position,
                    court_region=processed_result.get('court_region'),
                    player_positions=player_positions
                )
            elif hasattr(processed_result, 'event_type'):
                # オブジェクト形式の場合
                return EventDetectionResult(
                    event_type=getattr(processed_result, 'event_type', None),
                    confidence=getattr(processed_result, 'confidence', 0.0),
                    timestamp=getattr(processed_result, 'timestamp', batch_task.submit_time),
                    ball_position=ball_position,
                    court_region=getattr(processed_result, 'court_region', None),
                    player_positions=player_positions
                )
            else:
                # その他の形式の場合はダミー結果
                return EventDetectionResult(confidence=0.0)
        
        except Exception as e:
            logger.warning(f"Error converting prediction result: {e}")
            return EventDetectionResult(confidence=0.0)
    
    def _extract_player_positions(self, keypoints: List[Any]) -> List[tuple]:
        """キーポイントからプレイヤー位置を抽出します"""
        try:
            positions = []
            # キーポイントの形式に応じて調整
            # 例: 各プレイヤーの重心位置を計算
            if keypoints:
                # 簡単な実装例
                for kp in keypoints:
                    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                        positions.append((kp[0], kp[1]))
            return positions
        except Exception as e:
            logger.warning(f"Error extracting player positions: {e}")
            return []
    
    def _basic_postprocess(self, prediction: Any, batch_task: BatchTask) -> EventDetectionResult:
        """基本的な後処理を行います"""
        # フォールバック実装
        return EventDetectionResult(confidence=0.0)
    
    def _update_event_stats(self, result: EventDetectionResult) -> None:
        """イベント検出統計を更新します"""
        with self.lock:
            self.event_stats["detections_made"] += 1
            
            if result.event_type and result.confidence > 0.5:  # 閾値は調整可能
                self.event_stats["events_detected"] += 1
                
                # イベントタイプ別カウント
                event_type = result.event_type
                if event_type not in self.event_stats["event_types"]:
                    self.event_stats["event_types"][event_type] = 0
                self.event_stats["event_types"][event_type] += 1
            
            # 平均信頼度の更新
            total = self.event_stats["detections_made"]
            current_avg = self.event_stats["average_confidence"]
            self.event_stats["average_confidence"] = \
                (current_avg * (total - 1) + result.confidence) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を返します（基底クラスの統計に追加）"""
        base_stats = super().get_stats()
        
        with self.lock:
            base_stats.update({
                "event_detections_made": self.event_stats["detections_made"],
                "events_detected": self.event_stats["events_detected"],
                "event_types": self.event_stats["event_types"].copy(),
                "average_confidence": self.event_stats["average_confidence"],
                "missing_dependencies": self.event_stats["missing_dependencies"],
                "detection_rate": (
                    self.event_stats["events_detected"] / max(1, self.event_stats["detections_made"])
                )
            })
        
        return base_stats 