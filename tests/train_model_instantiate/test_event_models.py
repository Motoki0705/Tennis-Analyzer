"""
イベント検出モデルのインスタンス化テスト
"""
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# プロジェクトルートをimportパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.event.models import create_model, EventDetectionModel
from src.event.lit_module import EventDetectionLitModule

# テストするモデルタイプ
MODEL_TYPES = ["lstm", "gru", "bilstm", "bigru"]

# テスト用パラメータ
PARAMS = {
    "ball_dim": 3,
    "player_bbox_dim": 5,
    "player_pose_dim": 51,  # 17キーポイント × 3
    "court_dim": 31,
    "max_players": 2,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "num_classes": 4,
}

@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_event_model_instantiate(model_type):
    """
    イベント検出モデルが正しくインスタンス化できることをテストします
    """
    try:
        # モデルのインスタンス化
        model = create_model(model_type=model_type, **PARAMS)
        
        # モデルが正しいタイプであることを確認
        assert isinstance(model, EventDetectionModel), f"モデルのタイプが期待と異なります: {type(model)}"
        
        # モデルの基本的な構造を確認
        assert hasattr(model, "ball_encoder"), "モデルにball_encoderが含まれていません"
        assert hasattr(model, "player_bbox_encoder"), "モデルにplayer_bbox_encoderが含まれていません"
        assert hasattr(model, "player_pose_encoder"), "モデルにplayer_pose_encoderが含まれていません"
        assert hasattr(model, "court_encoder"), "モデルにcourt_encoderが含まれていません"
        assert hasattr(model, "rnn"), "モデルにrnnが含まれていません"
        
        # RNNタイプの確認
        if model_type.startswith("lstm"):
            assert isinstance(model.rnn, nn.LSTM), f"RNNのタイプが期待と異なります: {type(model.rnn)}"
        elif model_type.startswith("gru"):
            assert isinstance(model.rnn, nn.GRU), f"RNNのタイプが期待と異なります: {type(model.rnn)}"
        
        # 双方向性の確認
        if model_type.startswith("bi"):
            assert model.rnn.bidirectional, "RNNが双方向ではありません"
        else:
            assert not model.rnn.bidirectional, "RNNが双方向になっています"
        
        print(f"✅ モデルタイプ {model_type} のインスタンス化テスト成功")
    except Exception as e:
        pytest.fail(f"モデルタイプ {model_type} のインスタンス化に失敗しました: {e}")


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_event_model_forward(model_type):
    """
    イベント検出モデルのforward処理が正常に動作することをテストします
    """
    # モデルのインスタンス化
    model = create_model(model_type=model_type, **PARAMS)
    
    # 入力テンソルの作成
    batch_size = 2
    seq_len = 8
    max_players = PARAMS["max_players"]
    
    ball_features = torch.randn(batch_size, seq_len, PARAMS["ball_dim"])
    player_bbox_features = torch.randn(batch_size, seq_len, max_players, PARAMS["player_bbox_dim"])
    player_pose_features = torch.randn(batch_size, seq_len, max_players, PARAMS["player_pose_dim"])
    court_features = torch.randn(batch_size, seq_len, PARAMS["court_dim"])
    
    try:
        # forward処理の実行
        outputs = model(ball_features, player_bbox_features, player_pose_features, court_features)
        
        # 出力の形状を確認
        assert outputs.shape == (batch_size, seq_len, PARAMS["num_classes"]), \
            f"出力の形状が期待と異なります: {outputs.shape} != {(batch_size, seq_len, PARAMS['num_classes'])}"
        
        # predict メソッドのテスト
        probs, preds = model.predict(ball_features, player_bbox_features, player_pose_features, court_features)
        
        # 出力の形状を確認
        assert probs.shape == (batch_size, seq_len, PARAMS["num_classes"]), \
            f"確率の形状が期待と異なります: {probs.shape} != {(batch_size, seq_len, PARAMS['num_classes'])}"
        assert preds.shape == (batch_size, seq_len), \
            f"予測クラスの形状が期待と異なります: {preds.shape} != {(batch_size, seq_len)}"
        
        print(f"✅ モデルタイプ {model_type} のforward処理テスト成功")
    except Exception as e:
        pytest.fail(f"モデルタイプ {model_type} のforward処理に失敗しました: {e}")


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_event_lit_module_instantiate(model_type):
    """
    EventDetectionLitModuleが正しくインスタンス化できることをテストします
    """
    try:
        # モデルのインスタンス化
        model = create_model(model_type=model_type, **PARAMS)
        
        # LitModuleのインスタンス化
        lit_module = EventDetectionLitModule(
            model=model,
            num_classes=PARAMS["num_classes"],
            lr=1e-3,
            weight_decay=1e-4,
            warmup_epochs=5,
            max_epochs=100,
        )
        
        # LitModuleの基本的な構造を確認
        assert hasattr(lit_module, "model"), "LitModuleにmodelが含まれていません"
        assert hasattr(lit_module, "train_acc"), "LitModuleにtrain_accが含まれていません"
        assert hasattr(lit_module, "val_f1"), "LitModuleにval_f1が含まれていません"
        
        # forward処理のテスト
        batch_size = 2
        seq_len = 8
        max_players = PARAMS["max_players"]
        
        ball_features = torch.randn(batch_size, seq_len, PARAMS["ball_dim"])
        player_bbox_features = torch.randn(batch_size, seq_len, max_players, PARAMS["player_bbox_dim"])
        player_pose_features = torch.randn(batch_size, seq_len, max_players, PARAMS["player_pose_dim"])
        court_features = torch.randn(batch_size, seq_len, PARAMS["court_dim"])
        
        outputs = lit_module(ball_features, player_bbox_features, player_pose_features, court_features)
        
        # 出力の形状を確認
        assert outputs.shape == (batch_size, seq_len, PARAMS["num_classes"]), \
            f"LitModuleの出力形状が期待と異なります: {outputs.shape} != {(batch_size, seq_len, PARAMS['num_classes'])}"
        
        print(f"✅ モデルタイプ {model_type} のLitModuleインスタンス化テスト成功")
    except Exception as e:
        pytest.fail(f"モデルタイプ {model_type} のLitModuleインスタンス化に失敗しました: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 