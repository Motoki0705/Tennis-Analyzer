"""
イベント検出用Transformerモデルのインスタンス化テスト
"""
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# プロジェクトルートをimportパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.event.models.transformer import EventTransformer, LearnablePositionalEncoding

# テスト用パラメータ
PARAMS = {
    "pose_dim": 51,  # 17キーポイント × 3
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "dropout": 0.1,
    "max_seq_len": 512,
}

def test_learnable_positional_encoding():
    """
    学習可能な位置エンコーディングが正しくインスタンス化できることをテストします
    """
    try:
        # インスタンス化
        max_len = 100
        d_model = 64
        pos_enc = LearnablePositionalEncoding(max_len, d_model)
        
        # 基本的な構造を確認
        assert hasattr(pos_enc, "pe"), "位置エンコーディングにpeが含まれていません"
        assert isinstance(pos_enc.pe, nn.Embedding), "peがEmbeddingでありません"
        
        # forward処理のテスト
        batch_size = 2
        seq_len = 50
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_enc(x)
        
        # 出力の形状を確認
        assert output.shape == (batch_size, seq_len, d_model), \
            f"出力の形状が期待と異なります: {output.shape} != {(batch_size, seq_len, d_model)}"
        
        # 位置エンコーディングが追加されているかを確認
        assert not torch.allclose(x, output), "位置エンコーディングが追加されていません"
        
        print("✅ 位置エンコーディングのインスタンス化テスト成功")
    except Exception as e:
        pytest.fail(f"位置エンコーディングのインスタンス化に失敗しました: {e}")


def test_event_transformer_instantiate():
    """
    EventTransformerが正しくインスタンス化できることをテストします
    """
    try:
        # モデルのインスタンス化
        model = EventTransformer(**PARAMS)
        
        # モデルの基本的な構造を確認
        assert hasattr(model, "ball_embed"), "モデルにball_embedが含まれていません"
        assert hasattr(model, "court_embed"), "モデルにcourt_embedが含まれていません"
        assert hasattr(model, "player_embed"), "モデルにplayer_embedが含まれていません"
        assert hasattr(model, "frame_proj"), "モデルにframe_projが含まれていません"
        assert hasattr(model, "pos_enc"), "モデルにpos_encが含まれていません"
        assert hasattr(model, "encoder"), "モデルにencoderが含まれていません"
        assert hasattr(model, "classifier"), "モデルにclassifierが含まれていません"
        
        # 各コンポーネントのタイプ確認
        assert isinstance(model.pos_enc, LearnablePositionalEncoding), \
            f"位置エンコーディングのタイプが期待と異なります: {type(model.pos_enc)}"
        assert isinstance(model.encoder, nn.TransformerEncoder), \
            f"エンコーダのタイプが期待と異なります: {type(model.encoder)}"
        
        print("✅ EventTransformerのインスタンス化テスト成功")
    except Exception as e:
        pytest.fail(f"EventTransformerのインスタンス化に失敗しました: {e}")


def test_event_transformer_forward():
    """
    EventTransformerのforward処理が正常に動作することをテストします
    """
    # モデルのインスタンス化
    model = EventTransformer(**PARAMS)
    
    # 入力テンソルの作成
    batch_size = 2
    seq_len = 8
    num_players = 2  # プレイヤー数
    
    ball = torch.randn(batch_size, seq_len, 3)
    player_bbox = torch.randn(batch_size, seq_len, num_players, 5)
    player_pose = torch.randn(batch_size, seq_len, num_players, PARAMS["pose_dim"])
    court = torch.randn(batch_size, seq_len, 31)
    
    try:
        # forward処理の実行
        outputs = model(ball, player_bbox, player_pose, court)
        
        # 出力の形状を確認
        assert outputs.shape == (batch_size, seq_len), \
            f"出力の形状が期待と異なります: {outputs.shape} != {(batch_size, seq_len)}"
        
        print("✅ EventTransformerのforward処理テスト成功")
    except Exception as e:
        pytest.fail(f"EventTransformerのforward処理に失敗しました: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 