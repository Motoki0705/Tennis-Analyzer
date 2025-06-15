# streaming_annotator/workers/ball_worker.py

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base_worker import BaseWorker, BatchTask
from ..core.interfaces import ItemId, TopicName, TaskData, ResultData

logger = logging.getLogger(__name__)


class BallDetectionResult:
    """ボール検出結果を格納するクラス"""
    
    def __init__(self, position: Optional[tuple] = None, 
                 confidence: float = 0.0, 
                 visibility: bool = False,
                 heatmap: Optional[np.ndarray] = None):
        """
        Args:
            position: ボールの位置 (x, y)
            confidence: 検出信頼度
            visibility: ボールが見えているかどうか
            heatmap: 検出ヒートマップ
        """
        self.position = position
        self.confidence = confidence
        self.visibility = visibility
        self.heatmap = heatmap
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "position": self.position,
            "confidence": self.confidence,
            "visibility": self.visibility,
            "timestamp": self.timestamp,
            "has_heatmap": self.heatmap is not None
        }


class BallDetectionWorker(BaseWorker):
    """
    ボール検出処理を行うワーカー。
    
    上流ワーカー（依存関係なし）として動作し、
    フレーム画像からボールの位置を検出します。
    スライディングウィンドウを使った独自のバッチ処理を実装。
    """
    
    def __init__(self, name: str, predictor: Any, results_queue: queue.Queue,
                 max_concurrent_tasks: int = 3, debug: bool = False,
                 sliding_window_size: int = 3, batch_size: int = 8,
                 batch_timeout: float = 0.05, task_queue_maxsize: int = 1000):
        """
        Args:
            name: ワーカー名
            predictor: ボール検出予測器
            results_queue: 結果出力キュー
            max_concurrent_tasks: 最大同時実行タスク数
            debug: デバッグモード
            sliding_window_size: スライディングウィンドウサイズ
            batch_size: バッチサイズ
            batch_timeout: バッチタイムアウト
            task_queue_maxsize: タスクキューの最大サイズ
        """
        super().__init__(name, results_queue, max_concurrent_tasks, debug,
                        batch_size, batch_timeout, task_queue_maxsize)
        
        self.predictor = predictor
        self.sliding_window_size = sliding_window_size
        
        # スライディングウィンドウ管理（アイテムIDごと）
        self.sliding_windows: Dict[ItemId, List[np.ndarray]] = {}
        self.window_lock = threading.RLock()
        
        # 処理統計の拡張
        self.ball_stats = {
            "detections_made": 0,
            "balls_detected": 0,
            "average_confidence": 0.0
        }
    
    def get_published_topic(self) -> TopicName:
        """このワーカーが出版するトピック名を返します"""
        return "ball_detection"
    
    def get_dependencies(self) -> List[TopicName]:
        """このワーカーの依存関係を返します（上流ワーカーなので空リスト）"""
        return []
    
    def process_batch(self, batch_tasks: List[BatchTask]) -> List[ResultData]:
        """
        バッチタスクの処理を実行します。
        
        スライディングウィンドウを考慮した独自のバッチ処理を実装。
        
        Args:
            batch_tasks: バッチタスクのリスト
        
        Returns:
            List[ResultData]: 処理結果のリスト（batch_tasksと同じ順序）
        """
        try:
            results = []
            
            # 各タスクを処理してクリップを準備
            clips_batch = []
            task_clip_mapping = []  # (task_index, clip_indices)
            
            for task_idx, batch_task in enumerate(batch_tasks):
                # フレーム画像の検証
                if batch_task.task_data is None:
                    raise ValueError(f"Task data is None for item {batch_task.item_id}")
                
                if not isinstance(batch_task.task_data, np.ndarray):
                    raise ValueError(f"Expected numpy array, got {type(batch_task.task_data)} for item {batch_task.item_id}")
                
                # スライディングウィンドウの更新
                clips = self._update_sliding_window(batch_task.item_id, batch_task.task_data)
                
                if clips:
                    # クリップをバッチに追加
                    clip_start_idx = len(clips_batch)
                    clips_batch.extend(clips)
                    clip_end_idx = len(clips_batch)
                    task_clip_mapping.append((task_idx, list(range(clip_start_idx, clip_end_idx))))
                else:
                    # ウィンドウが十分でない場合
                    task_clip_mapping.append((task_idx, []))
            
            # バッチ推論の実行
            if clips_batch:
                batch_predictions = self._run_batch_inference(clips_batch)
            else:
                batch_predictions = []
            
            # 結果を各タスクに分配
            for task_idx, batch_task in enumerate(batch_tasks):
                # このタスクに対応するクリップのインデックスを取得
                clip_indices = None
                for mapping_task_idx, mapping_clip_indices in task_clip_mapping:
                    if mapping_task_idx == task_idx:
                        clip_indices = mapping_clip_indices
                        break
                
                if clip_indices and batch_predictions:
                    # 対応する予測結果を取得
                    task_predictions = [batch_predictions[i] for i in clip_indices]
                    
                    # 後処理
                    result = self._postprocess_predictions(
                        task_predictions, 
                        batch_task.task_data.shape[:2]
                    )
                else:
                    # ウィンドウが十分でない場合はダミー結果
                    result = BallDetectionResult(
                        position=None,
                        confidence=0.0,
                        visibility=False
                    )
                
                # 統計更新
                self._update_ball_stats(result)
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Ball detection batch processing error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # エラー時はダミー結果を返す
            return [BallDetectionResult(position=None, confidence=0.0, visibility=False) 
                   for _ in batch_tasks]
    
    def _update_sliding_window(self, item_id: ItemId, frame: np.ndarray) -> Optional[List[List[np.ndarray]]]:
        """
        スライディングウィンドウを更新し、処理用のクリップを返します。
        
        Args:
            item_id: フレームID
            frame: フレーム画像
        
        Returns:
            処理用のフレームクリップのリスト（十分なフレームが蓄積された場合）
        """
        with self.window_lock:
            # アイテムIDごとのウィンドウを管理
            if item_id not in self.sliding_windows:
                self.sliding_windows[item_id] = []
            
            # フレームを追加
            self.sliding_windows[item_id].append(frame.copy())
            
            # ウィンドウサイズを制限
            if len(self.sliding_windows[item_id]) > self.sliding_window_size:
                self.sliding_windows[item_id].pop(0)
            
            # 十分なフレームが蓄積されたかチェック
            if len(self.sliding_windows[item_id]) >= self.sliding_window_size:
                return [list(self.sliding_windows[item_id])]
            
            return None
    
    def _run_batch_inference(self, clips_batch: List[List[np.ndarray]]) -> List[Any]:
        """
        バッチ推論を実行します。
        
        Args:
            clips_batch: フレームクリップのバッチ
        
        Returns:
            推論結果のリスト
        """
        try:
            # 前処理
            processed_batch = self._preprocess_clips_batch(clips_batch)
            
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
            
            return predictions
        
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            raise
    
    def _preprocess_clips_batch(self, clips_batch: List[List[np.ndarray]]) -> Any:
        """
        クリップバッチの前処理を行います。
        
        Args:
            clips_batch: フレームクリップのバッチ
        
        Returns:
            前処理済みデータ
        """
        try:
            # predictorの前処理メソッドを呼び出し
            if hasattr(self.predictor, 'preprocess_batch'):
                return self.predictor.preprocess_batch(clips_batch)
            elif hasattr(self.predictor, 'preprocess'):
                return self.predictor.preprocess(clips_batch)
            else:
                # フォールバック: 基本的な前処理
                return self._basic_preprocess_batch(clips_batch)
        
        except Exception as e:
            logger.error(f"Batch preprocessing error: {e}")
            raise
    
    def _basic_preprocess_batch(self, clips_batch: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """基本的なバッチ前処理を行います"""
        # フォールバック実装: そのまま返す
        return clips_batch
    
    def _postprocess_predictions(self, predictions: List[Any], frame_shape: tuple) -> BallDetectionResult:
        """
        推論結果の後処理を行います。
        
        Args:
            predictions: 推論結果のリスト
            frame_shape: フレームの形状 (height, width)
        
        Returns:
            BallDetectionResult: 処理済みの検出結果
        """
        try:
            # 複数の予測結果から最適なものを選択（例：最新のフレーム）
            if predictions:
                latest_prediction = predictions[-1]  # 最新の予測を使用
            else:
                latest_prediction = None
            
            # predictorの後処理メソッドを呼び出し
            if hasattr(self.predictor, 'postprocess'):
                processed_result = self.predictor.postprocess(latest_prediction)
                
                # 結果をBallDetectionResultに変換
                return self._convert_to_detection_result(processed_result, frame_shape)
            else:
                # フォールバック: 基本的な後処理
                return self._basic_postprocess(latest_prediction, frame_shape)
        
        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            return BallDetectionResult(position=None, confidence=0.0, visibility=False)
    
    def _convert_to_detection_result(self, processed_result: Any, frame_shape: tuple) -> BallDetectionResult:
        """予測器の結果をBallDetectionResultに変換します"""
        try:
            # 結果の形式に応じて変換
            if isinstance(processed_result, dict):
                return BallDetectionResult(
                    position=processed_result.get('position'),
                    confidence=processed_result.get('confidence', 0.0),
                    visibility=processed_result.get('visibility', False),
                    heatmap=processed_result.get('heatmap')
                )
            elif hasattr(processed_result, 'position'):
                # オブジェクト形式の場合
                return BallDetectionResult(
                    position=getattr(processed_result, 'position', None),
                    confidence=getattr(processed_result, 'confidence', 0.0),
                    visibility=getattr(processed_result, 'visibility', False),
                    heatmap=getattr(processed_result, 'heatmap', None)
                )
            else:
                # その他の形式の場合はダミー結果
                return BallDetectionResult()
        
        except Exception as e:
            logger.warning(f"Error converting prediction result: {e}")
            return BallDetectionResult()
    
    def _basic_postprocess(self, predictions: Any, frame_shape: tuple) -> BallDetectionResult:
        """基本的な後処理を行います"""
        # フォールバック実装
        return BallDetectionResult(
            position=None,
            confidence=0.0,
            visibility=False
        )
    
    def _update_ball_stats(self, result: BallDetectionResult) -> None:
        """ボール検出統計を更新します"""
        with self.lock:
            self.ball_stats["detections_made"] += 1
            
            if result.visibility and result.position:
                self.ball_stats["balls_detected"] += 1
            
            # 平均信頼度の更新
            total = self.ball_stats["detections_made"]
            current_avg = self.ball_stats["average_confidence"]
            self.ball_stats["average_confidence"] = \
                (current_avg * (total - 1) + result.confidence) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を返します（基底クラスの統計に追加）"""
        base_stats = super().get_stats()
        
        with self.lock:
            base_stats.update({
                "ball_detections_made": self.ball_stats["detections_made"],
                "balls_detected": self.ball_stats["balls_detected"],
                "average_confidence": self.ball_stats["average_confidence"],
                "detection_rate": (
                    self.ball_stats["balls_detected"] / max(1, self.ball_stats["detections_made"])
                ),
                "sliding_windows_count": len(self.sliding_windows),
                "sliding_window_size": self.sliding_window_size
            })
        
        return base_stats
    
    def stop(self) -> None:
        """ワーカーを停止し、リソースをクリーンアップします"""
        super().stop()
        
        # スライディングウィンドウをクリア
        with self.window_lock:
            self.sliding_windows.clear()
        
        logger.info(f"BallDetectionWorker {self.name} stopped and cleaned up")
