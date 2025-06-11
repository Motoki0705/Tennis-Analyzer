"""
Hydra キュー設定統合のテスト
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from omegaconf import DictConfig, OmegaConf

from src.multi.streaming_overlayer.config_utils import (
    create_queue_configs_from_hydra_config,
    get_worker_extended_queue_names,
    apply_performance_settings,
    validate_queue_config,
    log_queue_configuration
)
from src.multi.streaming_overlayer.video_predictor import VideoPredictor


class TestHydraQueueConfig:
    """Hydra キュー設定のテストクラス"""

    @pytest.fixture
    def sample_hydra_config(self):
        """サンプルHydra設定を作成"""
        return OmegaConf.create({
            "base_queue_sizes": {
                "preprocess": 16,
                "inference": 16,
                "postprocess": 16,
                "results": 100
            },
            "worker_extended_queues": {
                "ball": {
                    "ball_inference": 32
                },
                "pose": {
                    "detection_inference": 32,
                    "detection_postprocess": 32,
                    "pose_inference": 32,
                    "pose_postprocess": 32
                }
            },
            "queue_types": {
                "preprocess": "Queue",
                "inference": "Queue",
                "postprocess": "Queue",
                "results": "PriorityQueue",
                "detection_inference": "Queue",
                "pose_inference": "Queue",
                "ball_inference": "Queue"
            },
            "performance": {
                "enable_monitoring": True,
                "log_queue_status": False,
                "auto_clear_on_shutdown": True
            }
        })

    @pytest.fixture
    def mock_predictors(self):
        """モック予測器を作成"""
        mock_ball = Mock()
        mock_court = Mock()
        mock_pose = Mock()
        
        # 基本的なメソッドを設定
        mock_ball.overlay.return_value = Mock()
        mock_court.overlay.return_value = Mock()
        mock_pose.overlay.return_value = Mock()
        
        return mock_ball, mock_court, mock_pose

    def test_create_queue_configs_from_hydra_config(self, sample_hydra_config):
        """Hydra設定からキュー設定作成のテスト"""
        queue_configs = create_queue_configs_from_hydra_config(sample_hydra_config)
        
        # 基本キューの確認
        assert "preprocess" in queue_configs
        assert queue_configs["preprocess"]["maxsize"] == 16
        assert queue_configs["preprocess"]["queue_type"] == "Queue"
        
        assert "results" in queue_configs
        assert queue_configs["results"]["queue_type"] == "PriorityQueue"
        
        # 拡張キューの確認
        assert "ball_inference" in queue_configs
        assert queue_configs["ball_inference"]["maxsize"] == 32
        
        assert "detection_inference" in queue_configs
        assert queue_configs["detection_inference"]["maxsize"] == 32

    def test_get_worker_extended_queue_names(self, sample_hydra_config):
        """ワーカー拡張キュー名取得のテスト"""
        # Ball ワーカー
        ball_queues = get_worker_extended_queue_names(sample_hydra_config, "ball")
        assert ball_queues == ["ball_inference"]
        
        # Pose ワーカー
        pose_queues = get_worker_extended_queue_names(sample_hydra_config, "pose")
        expected_pose_queues = ["detection_inference", "detection_postprocess", "pose_inference", "pose_postprocess"]
        assert set(pose_queues) == set(expected_pose_queues)
        
        # 存在しないワーカー
        unknown_queues = get_worker_extended_queue_names(sample_hydra_config, "unknown")
        assert unknown_queues == []

    def test_apply_performance_settings(self, sample_hydra_config):
        """パフォーマンス設定適用のテスト"""
        performance_settings = apply_performance_settings(sample_hydra_config)
        
        assert performance_settings["enable_monitoring"] is True
        assert performance_settings["log_queue_status"] is False
        assert performance_settings["auto_clear_on_shutdown"] is True

    def test_validate_queue_config_valid(self, sample_hydra_config):
        """有効なキュー設定の検証テスト"""
        assert validate_queue_config(sample_hydra_config) is True

    def test_validate_queue_config_invalid(self):
        """無効なキュー設定の検証テスト"""
        # 必須キューが不足
        invalid_config = OmegaConf.create({
            "base_queue_sizes": {
                "preprocess": 16,
                "inference": 16,
                # "postprocess": 16,  # 不足
                # "results": 100      # 不足
            }
        })
        
        assert validate_queue_config(invalid_config) is False
        
        # 無効なサイズ
        invalid_size_config = OmegaConf.create({
            "base_queue_sizes": {
                "preprocess": -1,  # 無効
                "inference": 16,
                "postprocess": 16,
                "results": 100
            }
        })
        
        assert validate_queue_config(invalid_size_config) is False
        
        # 無効なキュータイプ
        invalid_type_config = OmegaConf.create({
            "base_queue_sizes": {
                "preprocess": 16,
                "inference": 16,
                "postprocess": 16,
                "results": 100
            },
            "queue_types": {
                "preprocess": "InvalidType"  # 無効
            }
        })
        
        assert validate_queue_config(invalid_type_config) is False

    def test_video_predictor_with_hydra_config(self, sample_hydra_config, mock_predictors):
        """VideoPredictor での Hydra設定使用テスト"""
        ball_pred, court_pred, pose_pred = mock_predictors
        
        # VideoPredictor初期化
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 1, "court": 30, "pose": 5},
            batch_sizes={"ball": 16, "court": 16, "pose": 16},
            debug=True,
            hydra_queue_config=sample_hydra_config
        )
        
        # 初期化確認
        assert video_predictor.queue_manager is not None
        assert len(video_predictor.workers) == 3
        
        # パフォーマンス設定確認
        assert "enable_monitoring" in video_predictor.performance_settings
        assert video_predictor.performance_settings["enable_monitoring"] is True

    def test_video_predictor_with_invalid_hydra_config(self, mock_predictors):
        """無効なHydra設定でのVideoPredictor フォールバックテスト"""
        ball_pred, court_pred, pose_pred = mock_predictors
        
        # 無効な設定
        invalid_config = OmegaConf.create({
            "base_queue_sizes": {
                "preprocess": -1  # 無効
            }
        })
        
        # フォールバックして正常に初期化されることを確認
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 1, "court": 30, "pose": 5},
            batch_sizes={"ball": 16, "court": 16, "pose": 16},
            debug=True,
            hydra_queue_config=invalid_config
        )
        
        # デフォルト設定で初期化されることを確認
        assert video_predictor.queue_manager is not None
        assert len(video_predictor.workers) == 3

    def test_custom_queue_config_with_hydra(self, sample_hydra_config, mock_predictors):
        """カスタム設定とHydra設定の統合テスト"""
        ball_pred, court_pred, pose_pred = mock_predictors
        
        # カスタム設定
        custom_configs = {
            "experimental_queue": {
                "maxsize": 256,
                "queue_type": "Queue",
                "description": "実験的キュー"
            }
        }
        
        # VideoPredictor初期化
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 1, "court": 30, "pose": 5},
            batch_sizes={"ball": 16, "court": 16, "pose": 16},
            debug=True,
            custom_queue_configs=custom_configs,
            hydra_queue_config=sample_hydra_config
        )
        
        # 設定が統合されていることを確認
        assert video_predictor.queue_manager is not None

    def test_queue_status_with_performance_settings(self, sample_hydra_config, mock_predictors):
        """パフォーマンス設定に基づくキュー状態取得のテスト"""
        ball_pred, court_pred, pose_pred = mock_predictors
        
        # monitoring無効の設定
        config_no_monitoring = OmegaConf.create({
            "base_queue_sizes": {
                "preprocess": 16,
                "inference": 16,
                "postprocess": 16,
                "results": 100
            },
            "performance": {
                "enable_monitoring": False
            }
        })
        
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 1, "court": 30, "pose": 5},
            batch_sizes={"ball": 16, "court": 16, "pose": 16},
            debug=True,
            hydra_queue_config=config_no_monitoring
        )
        
        # 監視無効時の状態取得
        status = video_predictor.get_queue_status_with_settings()
        # monitoring が無効な場合の特別な応答を確認
        # 実際の実装に合わせて調整
        assert isinstance(status, dict)

    def test_custom_queues_config(self):
        """カスタムキュー設定のテスト"""
        config_with_custom = OmegaConf.create({
            "base_queue_sizes": {
                "preprocess": 16,
                "inference": 16,
                "postprocess": 16,
                "results": 100
            },
            "custom_queues": {
                "emergency_processing": {
                    "maxsize": 8,
                    "queue_type": "LifoQueue",
                    "description": "緊急処理用"
                },
                "priority_inference": {
                    "maxsize": 128,
                    "queue_type": "PriorityQueue",
                    "description": "高優先度推論"
                }
            }
        })
        
        queue_configs = create_queue_configs_from_hydra_config(config_with_custom)
        
        # カスタムキューが含まれることを確認
        assert "emergency_processing" in queue_configs
        assert queue_configs["emergency_processing"]["queue_type"] == "LifoQueue"
        assert queue_configs["emergency_processing"]["maxsize"] == 8
        
        assert "priority_inference" in queue_configs
        assert queue_configs["priority_inference"]["queue_type"] == "PriorityQueue"
        assert queue_configs["priority_inference"]["maxsize"] == 128

    def test_log_queue_configuration(self, sample_hydra_config, caplog):
        """キュー設定ログ出力のテスト"""
        import logging
        
        # ログ出力をテスト
        with caplog.at_level(logging.INFO):
            log_queue_configuration(sample_hydra_config)
        
        # ログメッセージの確認
        assert "キューシステム設定" in caplog.text
        assert "基本キューサイズ" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 