#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BallWorkerとPoseWorkerのユニットテスト
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import queue
from unittest.mock import Mock, MagicMock

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multi.streaming_overlayer.workers.ball_worker import BallWorker
from src.multi.streaming_overlayer.workers.pose_worker import PoseWorker
from src.multi.streaming_overlayer.definitions import PreprocessTask, InferenceTask, PostprocessTask


class TestBallWorker:
    """BallWorkerクラスのテスト"""

    @pytest.fixture
    def mock_predictor(self):
        """モックの予測器を作成"""
        mock_pred = Mock()
        mock_pred.preprocess.return_value = (torch.randn(4, 3, 360, 640), None)
        mock_pred.inference.return_value = torch.randn(4, 1, 360, 640)  # ヒートマップ想定
        mock_pred.postprocess.return_value = [{"x": 320, "y": 240, "confidence": 0.8}]
        return mock_pred

    @pytest.fixture
    def ball_worker(self, mock_predictor):
        """BallWorkerのインスタンスを作成"""
        preprocess_q = queue.Queue()
        inference_q = queue.Queue()
        postprocess_q = queue.Queue()
        results_q = queue.Queue()
        
        worker = BallWorker(
            name="ball",
            predictor=mock_predictor,
            preprocess_q=preprocess_q,
            inference_q=inference_q,
            postprocess_q=postprocess_q,
            results_q=results_q,
            debug=True
        )
        return worker

    def test_ball_worker_preprocess(self, ball_worker, mock_predictor):
        """前処理のテスト"""
        frames = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(4)]
        meta_data = [(i, 360, 640) for i in range(4)]
        
        task = PreprocessTask("test_task", frames, meta_data)
        
        ball_worker._process_preprocess_task(task)
        
        # 予測器の前処理が呼ばれたかチェック
        mock_predictor.preprocess.assert_called_once_with(frames)
        
        # 推論キューにタスクが追加されたかチェック
        assert not ball_worker.inference_queue.empty()
        inference_task = ball_worker.inference_queue.get()
        assert isinstance(inference_task, InferenceTask)
        assert inference_task.task_id == "test_task"

    def test_ball_worker_inference(self, ball_worker, mock_predictor):
        """推論のテスト"""
        tensor_data = torch.randn(4, 3, 360, 640)
        meta_data = [(i, 360, 640) for i in range(4)]
        
        task = InferenceTask("test_task", tensor_data, meta_data)
        
        ball_worker._process_inference_task(task)
        
        # 予測器の推論が呼ばれたかチェック
        mock_predictor.inference.assert_called_once_with(tensor_data)
        
        # 後処理キューにタスクが追加されたかチェック
        assert not ball_worker.postprocess_queue.empty()
        postprocess_task = ball_worker.postprocess_queue.get()
        assert isinstance(postprocess_task, PostprocessTask)

    def test_ball_worker_postprocess(self, ball_worker, mock_predictor):
        """後処理のテスト"""
        inference_output = torch.randn(4, 1, 360, 640)
        meta_data = [(i, 360, 640) for i in range(4)]
        
        task = PostprocessTask("test_task", inference_output, meta_data)
        
        ball_worker._process_postprocess_task(task)
        
        # 予測器の後処理が呼ばれたかチェック
        mock_predictor.postprocess.assert_called_once()
        
        # 結果キューに結果が追加されたかチェック
        assert not ball_worker.results_queue.empty()
        result = ball_worker.results_queue.get()
        assert len(result) == 3  # (frame_idx, task_name, result)
        assert result[1] == "ball"


class TestPoseWorker:
    """PoseWorkerクラスのテスト"""

    @pytest.fixture
    def mock_predictor(self):
        """モックの予測器を作成"""
        mock_pred = Mock()
        mock_pred.preprocess.return_value = (torch.randn(2, 3, 384, 384), None)
        mock_pred.inference.return_value = torch.randn(2, 2, 17, 3)  # 2人、17キーポイント、xyz
        mock_pred.postprocess.return_value = [
            [
                {
                    "bbox": [100, 100, 200, 300],
                    "keypoints": [0.5, 0.6, 1.0] * 17,  # 17キーポイント
                    "confidence": 0.9
                }
            ]
        ]
        return mock_pred

    @pytest.fixture
    def pose_worker(self, mock_predictor):
        """PoseWorkerのインスタンスを作成"""
        preprocess_q = queue.Queue()
        inference_q = queue.Queue()
        postprocess_q = queue.Queue()
        results_q = queue.Queue()
        
        worker = PoseWorker(
            name="pose",
            predictor=mock_predictor,
            preprocess_q=preprocess_q,
            inference_q=inference_q,
            postprocess_q=postprocess_q,
            results_q=results_q,
            debug=True
        )
        return worker

    def test_pose_worker_preprocess(self, pose_worker, mock_predictor):
        """前処理のテスト"""
        frames = [np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8) for _ in range(2)]
        meta_data = [(i, 384, 384) for i in range(2)]
        
        task = PreprocessTask("test_task", frames, meta_data)
        
        pose_worker._process_preprocess_task(task)
        
        # 予測器の前処理が呼ばれたかチェック
        mock_predictor.preprocess.assert_called_once_with(frames)
        
        # 推論キューにタスクが追加されたかチェック
        assert not pose_worker.inference_queue.empty()
        inference_task = pose_worker.inference_queue.get()
        assert isinstance(inference_task, InferenceTask)

    def test_pose_worker_inference(self, pose_worker, mock_predictor):
        """推論のテスト"""
        tensor_data = torch.randn(2, 3, 384, 384)
        meta_data = [(i, 384, 384) for i in range(2)]
        
        task = InferenceTask("test_task", tensor_data, meta_data)
        
        pose_worker._process_inference_task(task)
        
        # 予測器の推論が呼ばれたかチェック
        mock_predictor.inference.assert_called_once_with(tensor_data)
        
        # 後処理キューにタスクが追加されたかチェック
        assert not pose_worker.postprocess_queue.empty()

    def test_pose_worker_postprocess(self, pose_worker, mock_predictor):
        """後処理のテスト"""
        inference_output = torch.randn(2, 2, 17, 3)
        meta_data = [(i, 384, 384) for i in range(2)]
        
        task = PostprocessTask("test_task", inference_output, meta_data)
        
        pose_worker._process_postprocess_task(task)
        
        # 予測器の後処理が呼ばれたかチェック
        mock_predictor.postprocess.assert_called_once()
        
        # 結果キューに結果が追加されたかチェック
        assert not pose_worker.results_queue.empty()
        result = pose_worker.results_queue.get()
        assert len(result) == 3  # (frame_idx, task_name, result)
        assert result[1] == "pose"

    def test_pose_worker_error_handling(self, pose_worker):
        """エラーハンドリングのテスト"""
        # 予測器でエラーが発生する場合をテスト
        pose_worker.predictor.preprocess.side_effect = Exception("Test error")
        
        frames = [np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)]
        meta_data = [(0, 384, 384)]
        task = PreprocessTask("test_task", frames, meta_data)
        
        # エラーが発生してもクラッシュしないことを確認
        pose_worker._process_preprocess_task(task)
        
        # 推論キューは空のままであることを確認
        assert pose_worker.inference_queue.empty()


if __name__ == "__main__":
    pytest.main([__file__]) 