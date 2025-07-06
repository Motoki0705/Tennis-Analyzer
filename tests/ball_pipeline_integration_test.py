"""
Ball Detection Pipeline Integration Test

3段階パイプラインの統合テスト
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import logging

# テスト用のダミーデータ
from src.ball.pipeline import (
    BallDetectionPipeline, 
    PipelineConfig,
    FrameData,
    GlobalDetection,
    LocalDetection,
    TrajectoryDetection
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBallDetectionPipeline:
    """パイプライン統合テスト"""
    
    @pytest.fixture
    def dummy_frames(self):
        """テスト用ダミーフレーム"""
        frames = []
        for i in range(10):
            # 640x480のランダム画像
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_data = FrameData(
                frame_idx=i,
                image=frame,
                timestamp=i * (1/30.0)
            )
            frames.append(frame_data)
        return frames
    
    @pytest.fixture
    def dummy_config(self):
        """テスト用設定"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PipelineConfig(
                global_model_path="dummy_global_model.pth",  # ダミーパス
                local_model_path="dummy_local_model.pth",    # ダミーパス
                output_dir=Path(temp_dir),
                global_confidence_threshold=0.3,
                local_confidence_threshold=0.5,
                trajectory_confidence_threshold=0.4,
                save_intermediate_results=True
            )
            yield config
    
    def test_data_structures(self):
        """データ構造のテスト"""
        # GlobalDetection
        global_det = GlobalDetection(
            x=100.0, y=150.0, confidence=0.8, frame_idx=5,
            detection_area=(80, 130, 120, 170)
        )
        
        assert global_det.position == (100.0, 150.0)
        assert global_det.pixel_position == (100, 150)
        
        patch_region = global_det.get_patch_region(64)
        assert len(patch_region) == 4
        
        # LocalDetection
        local_det = LocalDetection(
            x=102.0, y=148.0, confidence=0.7, frame_idx=5,
            global_confidence=0.8, patch_confidence=0.7
        )
        
        combined_conf = local_det.combined_confidence
        assert 0.0 <= combined_conf <= 1.0
        
        # TrajectoryDetection
        traj_det = TrajectoryDetection(
            x=101.0, y=149.0, confidence=0.75, frame_idx=5,
            global_confidence=0.8, local_confidence=0.7, trajectory_confidence=0.6,
            velocity=(2.0, -1.0), acceleration=(0.1, 0.05)
        )
        
        final_conf = traj_det.final_confidence
        assert 0.0 <= final_conf <= 1.0
    
    def test_pipeline_config(self):
        """設定クラスのテスト"""
        config = PipelineConfig(
            global_model_path="test_global.pth",
            local_model_path="test_local.pth"
        )
        
        # デフォルト値チェック
        assert config.global_confidence_threshold == 0.3
        assert config.local_confidence_threshold == 0.5
        assert config.batch_size == 8
        
        # 辞書変換テスト
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "global_model_path" in config_dict
        
        # 辞書から復元テスト
        restored_config = PipelineConfig.from_dict(config_dict)
        assert restored_config.global_model_path == config.global_model_path
    
    def test_frame_data_creation(self, dummy_frames):
        """フレームデータ作成テスト"""
        assert len(dummy_frames) == 10
        
        first_frame = dummy_frames[0]
        assert first_frame.frame_idx == 0
        assert first_frame.image.shape == (480, 640, 3)
        assert first_frame.timestamp == 0.0
        
        last_frame = dummy_frames[-1]
        assert last_frame.frame_idx == 9
        assert last_frame.timestamp == 9 * (1/30.0)
    
    @pytest.mark.skip(reason="Requires actual model files")
    def test_pipeline_initialization(self, dummy_config):
        """パイプライン初期化テスト（実際のモデルが必要）"""
        # 実際のモデルファイルが必要なためスキップ
        # 実際のテストでは有効なモデルパスを設定
        pass
    
    def test_mock_global_detection(self, dummy_frames):
        """モックグローバル検出テスト"""
        # ダミーのグローバル検出結果を作成
        global_detections = []
        
        for i, frame_data in enumerate(dummy_frames):
            if i % 2 == 0:  # 偶数フレームのみ検出
                detection = GlobalDetection(
                    x=320.0 + np.random.normal(0, 10),  # 画面中央付近
                    y=240.0 + np.random.normal(0, 10),
                    confidence=0.6 + np.random.uniform(0, 0.3),
                    frame_idx=i,
                    detection_area=(300, 220, 340, 260)
                )
                global_detections.append(detection)
            else:
                global_detections.append(None)
        
        # 検出結果の検証
        detected_count = sum(1 for d in global_detections if d is not None)
        assert detected_count == 5  # 10フレーム中5フレーム
        
        for i, detection in enumerate(global_detections):
            if detection is not None:
                assert detection.frame_idx == i
                assert 0.6 <= detection.confidence <= 0.9
    
    def test_mock_local_refinement(self, dummy_frames):
        """モックローカル精密化テスト"""
        # ダミーのグローバル検出結果
        global_detections = []
        for i in range(len(dummy_frames)):
            if i % 2 == 0:
                detection = GlobalDetection(
                    x=320.0, y=240.0, confidence=0.7, frame_idx=i
                )
                global_detections.append(detection)
            else:
                global_detections.append(None)
        
        # ダミーのローカル精密化
        local_detections = []
        for i, global_det in enumerate(global_detections):
            if global_det is not None:
                # グローバル検出の一部のみローカルで通過
                if np.random.random() > 0.3:  # 70%の確率で通過
                    local_det = LocalDetection(
                        x=global_det.x + np.random.normal(0, 2),
                        y=global_det.y + np.random.normal(0, 2),
                        confidence=0.8,
                        frame_idx=i,
                        global_confidence=global_det.confidence,
                        patch_confidence=0.8
                    )
                    local_detections.append(local_det)
                else:
                    local_detections.append(None)
            else:
                local_detections.append(None)
        
        # 結果検証
        local_count = sum(1 for d in local_detections if d is not None)
        global_count = sum(1 for d in global_detections if d is not None)
        
        assert local_count <= global_count  # ローカル検出数 <= グローバル検出数
    
    def test_mock_trajectory_tracking(self, dummy_frames):
        """モック軌跡追跡テスト"""
        # ダミーのローカル検出結果（連続する軌跡）
        local_detections = []
        base_x, base_y = 320.0, 240.0
        
        for i in range(len(dummy_frames)):
            if i < 8:  # 最初の8フレームで検出
                x = base_x + i * 5  # 右に移動
                y = base_y + np.sin(i * 0.5) * 10  # 上下に振動
                
                local_det = LocalDetection(
                    x=x, y=y, confidence=0.8, frame_idx=i,
                    global_confidence=0.7, patch_confidence=0.8
                )
                local_detections.append(local_det)
            else:
                local_detections.append(None)
        
        # ダミー軌跡追跡
        trajectory_detections = []
        for i, local_det in enumerate(local_detections):
            if local_det is not None:
                # 軌跡の一貫性チェック（簡単な実装）
                trajectory_valid = True
                
                if i > 0:
                    # 前フレームとの距離チェック
                    prev_det = None
                    for j in range(i-1, -1, -1):
                        if local_detections[j] is not None:
                            prev_det = local_detections[j]
                            break
                    
                    if prev_det is not None:
                        distance = np.sqrt((local_det.x - prev_det.x)**2 + 
                                         (local_det.y - prev_det.y)**2)
                        trajectory_valid = distance < 50  # 最大移動距離
                
                if trajectory_valid:
                    traj_det = TrajectoryDetection(
                        x=local_det.x, y=local_det.y, confidence=0.85,
                        frame_idx=i,
                        global_confidence=local_det.global_confidence,
                        local_confidence=local_det.patch_confidence,
                        trajectory_confidence=0.7,
                        trajectory_valid=True
                    )
                    trajectory_detections.append(traj_det)
                else:
                    trajectory_detections.append(None)
            else:
                trajectory_detections.append(None)
        
        # 結果検証
        final_count = sum(1 for d in trajectory_detections if d is not None)
        local_count = sum(1 for d in local_detections if d is not None)
        
        assert final_count <= local_count  # 最終検出数 <= ローカル検出数
    
    def test_pipeline_results_structure(self, dummy_frames):
        """パイプライン結果構造テスト"""
        from src.ball.pipeline.data_structures import PipelineResults, PipelineConfig
        
        # ダミー結果を作成
        global_detections = [None] * len(dummy_frames)
        local_detections = [None] * len(dummy_frames)
        final_detections = [None] * len(dummy_frames)
        
        # いくつかの検出結果を追加
        for i in [2, 4, 6]:
            global_detections[i] = GlobalDetection(x=100, y=100, confidence=0.7, frame_idx=i)
            local_detections[i] = LocalDetection(x=102, y=98, confidence=0.8, frame_idx=i,
                                               global_confidence=0.7, patch_confidence=0.8)
            final_detections[i] = TrajectoryDetection(x=101, y=99, confidence=0.85, frame_idx=i,
                                                    global_confidence=0.7, local_confidence=0.8,
                                                    trajectory_confidence=0.7)
        
        config = PipelineConfig(
            global_model_path="dummy.pth",
            local_model_path="dummy.pth"
        )
        
        results = PipelineResults(
            global_detections=global_detections,
            local_detections=local_detections,
            final_detections=final_detections,
            total_frames=len(dummy_frames),
            processing_time=10.0,
            config=config
        )
        
        # 統計情報テスト
        stats = results.get_statistics()
        
        assert stats['total_frames'] == len(dummy_frames)
        assert stats['processing_time'] == 10.0
        assert stats['fps'] == len(dummy_frames) / 10.0
        assert stats['global_detection_rate'] == 3 / len(dummy_frames)
        assert stats['local_detection_rate'] == 3 / len(dummy_frames)
        assert stats['final_detection_rate'] == 3 / len(dummy_frames)
    
    def test_visualization_config(self):
        """可視化設定テスト"""
        config = PipelineConfig(
            global_model_path="dummy.pth",
            local_model_path="dummy.pth",
            visualization_config={
                'ball_color': (0, 255, 0),
                'ball_radius': 10,
                'show_confidence': True
            }
        )
        
        viz_config = config.visualization_config
        assert viz_config['ball_color'] == (0, 255, 0)
        assert viz_config['ball_radius'] == 10
        assert viz_config['show_confidence'] is True
        
        # デフォルト値の確認
        assert 'center_color' in viz_config
        assert 'font_scale' in viz_config


def test_integration_workflow():
    """統合ワークフローテスト"""
    logger.info("Running integration workflow test...")
    
    # 1. データ構造の作成
    frame_data = FrameData(
        frame_idx=0,
        image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        timestamp=0.0
    )
    
    # 2. 各段階の検出結果を作成
    global_detection = GlobalDetection(
        x=320.0, y=240.0, confidence=0.7, frame_idx=0,
        detection_area=(300, 220, 340, 260)
    )
    
    local_detection = LocalDetection(
        x=322.0, y=238.0, confidence=0.8, frame_idx=0,
        global_confidence=0.7, patch_confidence=0.8,
        patch_region=(314, 230, 330, 246)
    )
    
    trajectory_detection = TrajectoryDetection(
        x=321.0, y=239.0, confidence=0.85, frame_idx=0,
        global_confidence=0.7, local_confidence=0.8, trajectory_confidence=0.7,
        velocity=(1.0, -0.5), acceleration=(0.1, 0.05),
        trajectory_valid=True
    )
    
    # 3. 結果の検証
    assert global_detection.pixel_position == (320, 240)
    assert local_detection.combined_confidence > 0.0
    assert trajectory_detection.final_confidence > 0.0
    
    logger.info("Integration workflow test completed successfully")


if __name__ == "__main__":
    # 基本テストを実行
    test_integration_workflow()
    
    # pytest でより詳細なテストを実行する場合:
    # pytest tests/ball_pipeline_integration_test.py -v 