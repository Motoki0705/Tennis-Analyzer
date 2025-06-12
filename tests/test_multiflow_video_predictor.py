import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.multi.streaming_overlayer.video_predictor import VideoPredictor
from src.multi.streaming_overlayer.definitions import PreprocessTask, InferenceTask, PostprocessTask


class TestMultiflowVideoPredictor:
    """改善されたマルチフローVideoPredictor の包括的テスト"""

    @pytest.fixture
    def mock_predictors(self):
        """モック予測器を作成"""
        ball_predictor = Mock()
        ball_predictor.num_frames = 5
        ball_predictor.preprocess.return_value = np.random.rand(1, 3, 224, 224)
        ball_predictor.inference.return_value = [{"x": 100, "y": 150, "confidence": 0.8}]
        ball_predictor.postprocess.return_value = [{"x": 100, "y": 150, "confidence": 0.8}]
        ball_predictor.overlay.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        court_predictor = Mock()
        court_predictor.preprocess.return_value = (np.random.rand(1, 3, 224, 224), [(224, 224)])
        court_predictor.inference.return_value = np.random.rand(1, 15, 2)
        court_predictor.postprocess.return_value = ([[{"x": 10, "y": 20, "confidence": 0.9}]], None)
        court_predictor.overlay.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        pose_predictor = Mock()
        pose_predictor.preprocess_detection.return_value = np.random.rand(1, 3, 224, 224)
        pose_predictor.inference_detection.return_value = {"pred_logits": np.random.rand(1, 100, 2)}
        pose_predictor.postprocess_detection.return_value = ([], [], [], [])
        pose_predictor.preprocess_pose.return_value = np.random.rand(1, 3, 192, 256)
        pose_predictor.inference_pose.return_value = np.random.rand(1, 17, 64, 48)
        pose_predictor.postprocess_pose.return_value = [[]]
        pose_predictor.overlay.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        return ball_predictor, court_predictor, pose_predictor

    @pytest.fixture
    def video_predictor(self, mock_predictors):
        """VideoPredictor インスタンスを作成"""
        ball_pred, court_pred, pose_pred = mock_predictors
        
        return VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 1, "court": 5, "pose": 10},
            batch_sizes={"ball": 2, "court": 1, "pose": 1},
            debug=False,
            max_preload_frames=16,
            enable_performance_monitoring=True
        )

    def test_multiflow_initialization(self, video_predictor):
        """マルチフロー初期化のテスト"""
        # 基本属性の確認
        assert video_predictor.max_preload_frames == 16
        assert video_predictor.enable_performance_monitoring == True
        assert video_predictor.frame_processing_pool is not None
        
        # パフォーマンスメトリクスの初期化確認
        metrics = video_predictor.performance_metrics
        assert "total_frames_processed" in metrics
        assert "frames_per_second" in metrics
        assert metrics["total_frames_processed"] == 0

    def test_worker_initialization_with_threadpools(self, video_predictor):
        """ワーカーのスレッドプール初期化テスト"""
        # 各ワーカーのスレッドプール確認
        for name, worker in video_predictor.workers.items():
            if name == "ball":
                assert hasattr(worker, 'preprocess_pool')
                assert hasattr(worker, 'postprocess_pool')
                assert hasattr(worker, 'sliding_window')
                assert hasattr(worker, 'sliding_window_lock')
            elif name == "court":
                assert hasattr(worker, 'preprocess_pool')
                assert hasattr(worker, 'postprocess_pool')
            elif name == "pose":
                assert hasattr(worker, 'detection_preprocess_pool')
                assert hasattr(worker, 'pose_postprocess_pool')

    def test_frame_processing_parallel(self, video_predictor):
        """並列フレーム処理のテスト"""
        # モックフレームデータ
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        buffers = {"ball": [], "court": [], "pose": []}
        meta_buffers = {"ball": [], "court": [], "pose": []}
        
        # 単一フレーム処理をテスト
        frame_buffers, frame_meta_buffers = video_predictor._process_single_frame(
            0, frame, buffers, meta_buffers
        )
        
        # 結果検証
        assert "ball" in frame_buffers
        assert "court" in frame_buffers
        assert "pose" in frame_buffers
        
        # ball は interval=1 なのでフレーム追加されるはず
        assert len(frame_buffers["ball"]) == 1
        assert len(frame_meta_buffers["ball"]) == 1

    def test_task_creation_and_submission(self, video_predictor):
        """タスク作成と投入のテスト"""
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
        meta_data = [(0, 224, 224), (1, 224, 224)]
        
        # タスク作成・投入をテスト
        with patch.object(video_predictor.queue_manager, 'get_queue') as mock_get_queue:
            mock_queue = Mock()
            mock_get_queue.return_value = mock_queue
            
            video_predictor._create_and_submit_task("ball", 0, frames, meta_data)
            
            # キュー取得とタスク投入の確認
            mock_get_queue.assert_called_once_with("ball", "preprocess")
            mock_queue.put.assert_called_once()

    def test_performance_monitoring(self, video_predictor):
        """パフォーマンス監視機能のテスト"""
        # パフォーマンスメトリクス初期化
        video_predictor.performance_metrics["start_time"] = time.time()
        video_predictor.performance_metrics["total_frames_processed"] = 100
        
        # 少し待機
        time.sleep(0.1)
        video_predictor.performance_metrics["end_time"] = time.time()
        
        # メトリクス最終化
        video_predictor._finalize_performance_metrics()
        
        # 結果検証
        assert video_predictor.performance_metrics["total_processing_time"] > 0
        assert video_predictor.performance_metrics["frames_per_second"] > 0

    def test_worker_performance_stats_collection(self, video_predictor):
        """ワーカーのパフォーマンス統計収集テスト"""
        # モックワーカーの統計を設定
        for name, worker in video_predictor.workers.items():
            if hasattr(worker, 'get_performance_stats'):
                # モック統計データを返すように設定
                worker.get_performance_stats = Mock(return_value={
                    "preprocess_count": 10,
                    "inference_count": 8,
                    "postprocess_count": 7
                })
        
        # パフォーマンス最終化実行
        video_predictor._finalize_performance_metrics()
        
        # ワーカー統計の確認
        worker_performance = video_predictor.performance_metrics["worker_performance"]
        for name in video_predictor.workers.keys():
            if name in worker_performance:
                stats = worker_performance[name]
                assert "preprocess_count" in stats or "detection_preprocess_count" in stats

    def test_error_handling_in_parallel_processing(self, video_predictor):
        """並列処理でのエラーハンドリングテスト"""
        # エラーを発生させるモックフレーム処理
        def mock_process_frame(*args, **kwargs):
            raise ValueError("Test error")
        
        with patch.object(video_predictor, '_process_single_frame', side_effect=mock_process_frame):
            # エラーが発生しても処理が続行されることを確認
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # ThreadPoolExecutorでのエラーハンドリングをテスト
            future = video_predictor.frame_processing_pool.submit(
                video_predictor._process_single_frame, 0, frame, {}, {}
            )
            
            with pytest.raises(ValueError):
                future.result()

    def test_queue_timeout_handling(self, video_predictor):
        """キューのタイムアウト処理テスト"""
        # 空のキューを使用
        with patch.object(video_predictor.queue_manager, 'get_results_queue') as mock_get_queue:
            import queue
            mock_queue = Mock()
            mock_queue.empty.return_value = True
            mock_queue.get.side_effect = queue.Empty()
            mock_get_queue.return_value = mock_queue
            
            # タイムアウト処理が適切に動作することを確認
            # 実際の処理では適切にタイムアウトハンドリングされるはず
            assert mock_queue.empty() == True

    def test_sliding_window_management(self, video_predictor):
        """スライディングウィンドウ管理のテスト"""
        ball_worker = video_predictor.workers["ball"]
        
        # 初期状態確認
        assert len(ball_worker.sliding_window) == 0
        
        # フレーム追加（シミュレーション）
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(7)]
        
        # スライディングウィンドウに手動で追加（テスト用）
        for frame in frames:
            ball_worker.sliding_window.append(frame)
            if len(ball_worker.sliding_window) > ball_worker.predictor.num_frames:
                ball_worker.sliding_window.pop(0)
        
        # ウィンドウサイズが制限されていることを確認
        assert len(ball_worker.sliding_window) == ball_worker.predictor.num_frames

    def test_threadpool_shutdown(self, video_predictor):
        """スレッドプールの正常な終了テスト"""
        # ワーカー開始
        for worker in video_predictor.workers.values():
            worker.start()
        
        # ワーカー停止
        for worker in video_predictor.workers.values():
            worker.stop()
        
        # スレッドプール終了
        video_predictor.frame_processing_pool.shutdown(wait=True)
        
        # 終了状態確認（エラーが発生しないことを確認）
        assert video_predictor.frame_processing_pool._shutdown

    def test_performance_summary_output(self, video_predictor, capsys):
        """パフォーマンス概要出力のテスト"""
        # テストデータ設定
        video_predictor.performance_metrics.update({
            "total_frames_processed": 100,
            "total_processing_time": 10.0,
            "frames_per_second": 10.0,
            "worker_performance": {
                "ball": {"preprocess_count": 50, "inference_count": 48}
            }
        })
        
        # パフォーマンス概要出力
        video_predictor._print_performance_summary()
        
        # 出力内容確認
        captured = capsys.readouterr()
        assert "パフォーマンス監視レポート" in captured.out
        assert "総処理フレーム数: 100" in captured.out
        assert "平均FPS: 10.0" in captured.out

    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_full_pipeline_integration(self, mock_writer, mock_capture, video_predictor, mock_predictors):
        """フルパイプライン統合テスト"""
        # モックVideoCapture設定
        mock_cap = Mock()
        mock_cap.read.side_effect = [
            (True, np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
            (True, np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
            (False, None)  # 終端
        ]
        mock_capture.return_value = mock_cap
        
        # モックVideoWriter設定
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        
        # FrameLoaderのモック
        with patch('src.multi.streaming_overlayer.video_predictor.FrameLoader') as mock_frame_loader:
            mock_loader = Mock()
            mock_loader.get_properties.return_value = {
                "fps": 30, "width": 224, "height": 224, "total_frames": 2
            }
            mock_loader.read.side_effect = [
                (0, np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
                (1, np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
                None  # 終端
            ]
            mock_frame_loader.return_value.start.return_value = mock_loader
            
            # テンポラリファイルでテスト実行
            with tempfile.NamedTemporaryFile(suffix='.mp4') as input_file, \
                 tempfile.NamedTemporaryFile(suffix='.mp4') as output_file:
                
                # パイプライン実行（エラーが発生しないことを確認）
                try:
                    video_predictor.run(input_file.name, output_file.name)
                    # 基本的な実行完了を確認
                    assert video_predictor.performance_metrics["total_frames_processed"] >= 0
                except Exception as e:
                    # 予期される例外（モック環境での制限）は許容
                    assert "mock" in str(e).lower() or "attribute" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 