"""
QueueManagerとVideoPredictor、PoseWorkerの統合テスト
"""
import pytest
import queue
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
from PIL import Image

from src.multi.streaming_overlayer.queue_manager import (
    QueueManager, 
    QueueConfig, 
    WorkerQueueSet,
    create_queue_manager_for_video_predictor,
    WORKER_QUEUE_CONFIGS
)
from src.multi.streaming_overlayer.workers.pose_worker import PoseWorker
from src.multi.streaming_overlayer.video_predictor import VideoPredictor


class TestQueueManagerIntegration:
    """QueueManagerとワーカー統合のテストクラス"""

    @pytest.fixture
    def mock_predictors(self):
        """モック予測器の作成"""
        mock_ball = Mock()
        mock_court = Mock()
        mock_pose = Mock()
        
        # Pose predictor の詳細なモック
        mock_pose.preprocess_detection.return_value = {"pixel_values": Mock()}
        mock_pose.inference_detection.return_value = {"pred_boxes": Mock(), "pred_logits": Mock()}
        mock_pose.postprocess_detection.return_value = (
            [[100, 50, 80, 120]],  # batch_boxes
            [[0.95]],  # batch_scores
            [0],  # batch_valid
            [Image.new('RGB', (640, 480))]  # images_for_pose
        )
        mock_pose.preprocess_pose.return_value = {"inputs": Mock()}
        mock_pose.inference_pose.return_value = {"keypoints": Mock(), "scores": Mock()}
        mock_pose.postprocess_pose.return_value = [
            [  # フレーム0
                {
                    "bbox": [100, 50, 80, 120],
                    "det_score": 0.95,
                    "keypoints": [(110, 60), (115, 65), (120, 70)],
                    "scores": [0.9, 0.8, 0.7]
                }
            ]
        ]
        
        return mock_ball, mock_court, mock_pose

    def test_queue_manager_initialization(self):
        """QueueManagerの初期化テスト"""
        worker_names = ["ball", "court", "pose"]
        queue_manager = create_queue_manager_for_video_predictor(worker_names)
        
        # 基本的な初期化確認
        assert queue_manager.results_queue is not None
        assert len(queue_manager.worker_queue_sets) == 3
        
        # 各ワーカーのキューセット確認
        for worker_name in worker_names:
            queue_set = queue_manager.get_worker_queue_set(worker_name)
            assert queue_set is not None
            assert queue_set.worker_name == worker_name
            
            # 基本キューの存在確認
            assert queue_set.get_queue("preprocess") is not None
            assert queue_set.get_queue("inference") is not None
            assert queue_set.get_queue("postprocess") is not None

    def test_pose_worker_extended_queues(self):
        """PoseWorkerの拡張キューテスト"""
        worker_names = ["pose"]
        queue_manager = create_queue_manager_for_video_predictor(worker_names)
        
        pose_queue_set = queue_manager.get_worker_queue_set("pose")
        assert pose_queue_set is not None
        
        # 拡張キューの存在確認
        extended_queues = ["detection_inference", "detection_postprocess", "pose_inference", "pose_postprocess"]
        for queue_name in extended_queues:
            q = pose_queue_set.get_queue(queue_name)
            assert q is not None
            assert isinstance(q, queue.Queue)

    def test_pose_worker_with_queue_manager(self, mock_predictors):
        """QueueManagerを使用したPoseWorkerの初期化テスト"""
        _, _, mock_pose = mock_predictors
        
        # QueueManagerの設定
        worker_names = ["pose"]
        queue_manager = create_queue_manager_for_video_predictor(worker_names)
        
        # PoseWorkerの初期化
        pose_queue_set = queue_manager.get_worker_queue_set("pose")
        results_queue = queue_manager.get_results_queue()
        
        pose_worker = PoseWorker(
            name="test_pose",
            predictor=mock_pose,
            queue_set=pose_queue_set,
            results_q=results_queue,
            debug=True
        )
        
        # 必要なキューが正しく設定されていることを確認
        assert pose_worker.detection_inference_queue is not None
        assert pose_worker.detection_postprocess_queue is not None
        assert pose_worker.pose_inference_queue is not None
        assert pose_worker.pose_postprocess_queue is not None

    def test_queue_manager_custom_configs(self):
        """カスタムキュー設定のテスト"""
        custom_configs = {
            "custom_queue": {
                "maxsize": 64,
                "queue_type": "Queue",
                "description": "カスタムテスト用キュー"
            }
        }
        
        worker_names = ["ball"]
        queue_manager = create_queue_manager_for_video_predictor(worker_names, custom_configs)
        
        # カスタム設定が追加されていることを確認
        assert "custom_queue" in queue_manager.queue_configs
        custom_config = queue_manager.queue_configs["custom_queue"]
        assert custom_config.maxsize == 64
        assert custom_config.description == "カスタムテスト用キュー"

    def test_queue_status_monitoring(self):
        """キュー状態監視のテスト"""
        worker_names = ["ball", "court", "pose"]
        queue_manager = create_queue_manager_for_video_predictor(worker_names)
        
        # 初期状態の確認
        status = queue_manager.get_queue_status()
        assert "workers" in status
        assert "results_queue_size" in status
        assert status["results_queue_size"] == 0
        
        # 各ワーカーの状態確認
        for worker_name in worker_names:
            assert worker_name in status["workers"]
            worker_status = status["workers"][worker_name]
            assert "base_queues" in worker_status
            assert "extended_queues" in worker_status

    def test_queue_manager_worker_configs(self):
        """ワーカー別キュー設定のテスト"""
        # 設定の確認
        assert "ball" in WORKER_QUEUE_CONFIGS
        assert "court" in WORKER_QUEUE_CONFIGS
        assert "pose" in WORKER_QUEUE_CONFIGS
        
        # Poseワーカーの拡張キュー設定確認
        pose_config = WORKER_QUEUE_CONFIGS["pose"]
        expected_queues = ["detection_inference", "detection_postprocess", "pose_inference", "pose_postprocess"]
        assert pose_config["extended_queues"] == expected_queues

    def test_video_predictor_integration(self, mock_predictors):
        """VideoPredictorとQueueManagerの統合テスト"""
        ball_pred, court_pred, pose_pred = mock_predictors
        
        # VideoPredictor の初期化
        intervals = {"ball": 1, "court": 30, "pose": 5}
        batch_sizes = {"ball": 16, "court": 16, "pose": 16}
        
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals=intervals,
            batch_sizes=batch_sizes,
            debug=True
        )
        
        # QueueManagerが正しく初期化されていることを確認
        assert video_predictor.queue_manager is not None
        assert len(video_predictor.queue_manager.worker_queue_sets) == 3
        
        # ワーカーが正しく初期化されていることを確認
        assert len(video_predictor.workers) == 3
        assert "ball" in video_predictor.workers
        assert "court" in video_predictor.workers
        assert "pose" in video_predictor.workers

    def test_queue_clear_functionality(self):
        """キュークリア機能のテスト"""
        worker_names = ["ball", "pose"]
        queue_manager = create_queue_manager_for_video_predictor(worker_names)
        
        # キューにアイテムを追加
        ball_preprocess_q = queue_manager.get_queue("ball", "preprocess")
        pose_detection_q = queue_manager.get_queue("pose", "detection_inference")
        results_q = queue_manager.get_results_queue()
        
        ball_preprocess_q.put("test_item")
        pose_detection_q.put("test_item")
        results_q.put("test_item")
        
        # キューにアイテムが追加されたことを確認
        assert not ball_preprocess_q.empty()
        assert not pose_detection_q.empty()
        assert not results_q.empty()
        
        # キューをクリア
        queue_manager.clear_all_queues()
        
        # キューが空になったことを確認
        assert ball_preprocess_q.empty()
        assert pose_detection_q.empty()
        assert results_q.empty()

    def test_queue_error_handling(self):
        """キューエラーハンドリングのテスト"""
        queue_manager = QueueManager()
        
        # 存在しない設定からキューを作成しようとする
        with pytest.raises(ValueError, match="Queue configuration 'nonexistent' not found"):
            queue_manager.create_queue_from_config("nonexistent")
        
        # 重複したワーカーを初期化しようとする
        queue_manager.initialize_worker_queues("test_worker")
        with pytest.raises(ValueError, match="Worker 'test_worker' already has queue set initialized"):
            queue_manager.initialize_worker_queues("test_worker")

    def test_different_queue_types(self):
        """異なるキュータイプのテスト"""
        queue_manager = QueueManager()
        
        # PriorityQueueの作成テスト
        priority_q = queue_manager.create_queue_from_config("results")
        assert isinstance(priority_q, queue.PriorityQueue)
        
        # 通常のQueueの作成テスト
        normal_q = queue_manager.create_queue_from_config("preprocess")
        assert isinstance(normal_q, queue.Queue)
        assert not isinstance(normal_q, queue.PriorityQueue)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 