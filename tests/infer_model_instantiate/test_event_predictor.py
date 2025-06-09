#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EventPredictorのユニットテスト
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.event.api.event_predictor import EventPredictor


class TestEventPredictor:
    """EventPredictorクラスのテスト"""

    @pytest.fixture
    def mock_checkpoint_path(self, tmp_path):
        """モックのチェックポイントファイルを作成"""
        checkpoint_path = tmp_path / "mock_model.ckpt"
        # ダミーファイルを作成
        checkpoint_path.write_text("dummy checkpoint")
        return str(checkpoint_path)

    @pytest.fixture
    def mock_lit_transformer(self):
        """モックのLitTransformerV2を作成"""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        
        # forwardメソッドのモック
        def mock_forward(**kwargs):
            ball_features = kwargs.get('ball_features')
            if ball_features is not None:
                batch_size, seq_len = ball_features.shape[:2]
                # ダミーのlogitsを返す [B, T, 2]
                return torch.randn(batch_size, seq_len, 2)
            return torch.randn(1, 16, 2)
        
        mock_model.forward = mock_forward
        mock_model.__call__ = mock_forward
        mock_model.side_effect = mock_forward
        return mock_model

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_init(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """初期化のテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(
            checkpoint_path=mock_checkpoint_path,
            device="cpu",
            confidence_threshold=0.7,
            smoothing_window=3
        )
        
        assert predictor.device == torch.device("cpu")
        assert predictor.confidence_threshold == 0.7
        assert predictor.smoothing_window == 3
        assert len(predictor.signal_history) == 0
        mock_lit_class.load_from_checkpoint.assert_called_once()

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_preprocess(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """前処理のテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(mock_checkpoint_path)
        
        # テストデータ
        combined_features = {
            'ball_features': torch.randn(1, 16, 3),
            'player_bbox_features': torch.randn(1, 16, 2, 5),
            'player_pose_features': torch.randn(1, 16, 2, 51),
            'court_features': torch.randn(1, 16, 45),
        }
        
        processed_data, meta_data = predictor.preprocess(combined_features)
        
        assert isinstance(processed_data, dict)
        assert 'ball_features' in processed_data
        assert isinstance(processed_data['ball_features'], torch.Tensor)
        assert meta_data is None

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_inference(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """推論のテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(mock_checkpoint_path)
        
        # テストデータ
        tensor_data = {
            'ball_features': torch.randn(1, 16, 3),
            'player_bbox_features': torch.randn(1, 16, 2, 5),
            'player_pose_features': torch.randn(1, 16, 2, 51),
            'court_features': torch.randn(1, 16, 45),
        }
        
        logits = predictor.inference(tensor_data)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (1, 16, 2)  # [B, T, 2]

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_postprocess(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """後処理のテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(mock_checkpoint_path, confidence_threshold=0.5)
        
        # テストデータ（logits）
        logits = torch.tensor([[[1.0, -1.0], [0.0, 2.0], [-0.5, 0.5]]])  # [1, 3, 2]
        
        result = predictor.postprocess(logits)
        
        assert isinstance(result, dict)
        assert 'hit_probability' in result
        assert 'bounce_probability' in result
        assert 'hit_detected' in result
        assert 'bounce_detected' in result
        assert 'signal_history' in result
        
        # 確率値の範囲チェック
        assert 0.0 <= result['hit_probability'] <= 1.0
        assert 0.0 <= result['bounce_probability'] <= 1.0
        
        # 信号履歴が更新されているかチェック
        assert len(predictor.signal_history) == 1

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_smooth_signals(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """信号平滑化のテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(mock_checkpoint_path, smoothing_window=3)
        
        # 履歴にデータを追加
        predictor.signal_history = [
            (0.8, 0.2),
            (0.7, 0.3),
            (0.9, 0.1),
            (0.6, 0.4),
        ]
        
        smoothed_hit, smoothed_bounce = predictor._smooth_signals()
        
        # 直近3つの平均: hit=(0.7+0.9+0.6)/3=0.733, bounce=(0.3+0.1+0.4)/3=0.267
        assert abs(smoothed_hit - 0.733) < 0.01
        assert abs(smoothed_bounce - 0.267) < 0.01

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_overlay(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """オーバーレイ描画のテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(mock_checkpoint_path)
        
        # テストフレーム
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # テスト予測結果
        prediction = {
            'hit_probability': 0.8,
            'bounce_probability': 0.3,
            'hit_detected': True,
            'bounce_detected': False,
            'signal_history': [(0.7, 0.2), (0.8, 0.3), (0.6, 0.4)]
        }
        
        overlay_frame = predictor.overlay(frame, prediction)
        
        assert isinstance(overlay_frame, np.ndarray)
        assert overlay_frame.shape == frame.shape
        # オーバーレイによって何かしら変更されているはず
        assert not np.array_equal(frame, overlay_frame)

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_reset_history(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """履歴リセットのテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(mock_checkpoint_path)
        
        # 履歴にデータを追加
        predictor.signal_history = [(0.5, 0.3), (0.7, 0.2)]
        assert len(predictor.signal_history) == 2
        
        # リセット
        predictor.reset_history()
        assert len(predictor.signal_history) == 0

    def test_create_event_predictor_factory(self):
        """ファクトリ関数のテスト"""
        with patch('src.event.api.event_predictor.EventPredictor') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            from src.event.api.event_predictor import create_event_predictor
            
            result = create_event_predictor("dummy_path", device="cuda")
            
            mock_class.assert_called_once_with(checkpoint_path="dummy_path", device="cuda")
            assert result == mock_instance

    @patch('src.event.api.event_predictor.LitTransformerV2')
    def test_max_history_length(self, mock_lit_class, mock_checkpoint_path, mock_lit_transformer):
        """履歴の最大長制限のテスト"""
        mock_lit_class.load_from_checkpoint.return_value = mock_lit_transformer
        
        predictor = EventPredictor(mock_checkpoint_path)
        predictor.max_history_length = 3  # テスト用に短く設定
        
        # 最大長を超えるデータを追加
        for i in range(5):
            logits = torch.tensor([[[float(i), float(i+1)]]])  # [1, 1, 2]
            predictor.postprocess(logits)
        
        # 最大長に制限されているかチェック
        assert len(predictor.signal_history) == 3
        
        # 最新の3つだけが残っているかチェック（Sigmoidを考慮した期待値）
        # sigmoid(2) ≈ 0.88, sigmoid(3) ≈ 0.95, sigmoid(4) ≈ 0.98, sigmoid(5) ≈ 0.99
        expected_history = [
            (torch.sigmoid(torch.tensor(2.0)).item(), torch.sigmoid(torch.tensor(3.0)).item()),
            (torch.sigmoid(torch.tensor(3.0)).item(), torch.sigmoid(torch.tensor(4.0)).item()),
            (torch.sigmoid(torch.tensor(4.0)).item(), torch.sigmoid(torch.tensor(5.0)).item())
        ]
        for actual, expected in zip(predictor.signal_history, expected_history):
            assert abs(actual[0] - expected[0]) < 0.01  # Sigmoidの近似値
            assert abs(actual[1] - expected[1]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__]) 