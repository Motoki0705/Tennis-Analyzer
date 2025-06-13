#!/usr/bin/env python3
"""
VideoSwinTransformer用LightningModuleのテストスクリプト
"""
import torch
import torch.nn as nn
from src.ball.models.seq_lite_transformer import VideoSwinTransformer
from src.ball.lit_module.lit_seq_lite_transformer import LitSeqLiteTransformer


def test_model_only():
    """VideoSwinTransformerモデル単体のテスト"""
    print("=== VideoSwinTransformerモデル単体テスト ===")
    
    # ハイパーパラメータ
    img_size = (64, 128)  # さらに小さいサイズ
    batch_size = 1
    n_frames = 2
    in_channels = 3
    out_channels = 1
    
    # モデルの初期化
    model = VideoSwinTransformer(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        n_frames=n_frames,
        window_size=4,  # 小さいサイズ
        feature_dim=32,  # 軽量化
        transformer_blocks=1,  # 1ブロックのみ
        transformer_heads=2,  # ヘッド数を削減
    )
    
    print(f"モデル初期化完了: {type(model).__name__}")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # ダミーデータの作成
    frames = torch.randn(batch_size, n_frames, in_channels, img_size[0], img_size[1])
    
    print(f"入力データ形状: {frames.shape}")
    
    # モデルの順伝播テスト
    model.eval()
    try:
        with torch.no_grad():
            output = model(frames, debug=True)
            print(f"出力形状: {output.shape}")
            
            # 期待される形状かチェック
            expected_shape = (batch_size, n_frames, out_channels, img_size[0], img_size[1])
            assert output.shape == expected_shape, f"期待される形状 {expected_shape}, 実際の形状 {output.shape}"
            
            print("✓ モデル単体テスト成功")
            return True
    except Exception as e:
        print(f"✗ モデル単体テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lit_seq_lite_transformer():
    """LitSeqLiteTransformerの基本的な動作テスト"""
    print("\n=== LitSeqLiteTransformerテスト ===")
    
    # まずモデル単体テストが成功した場合のみ実行
    if not test_model_only():
        return
    
    # ハイパーパラメータ
    img_size = (64, 128)
    batch_size = 1
    n_frames = 2
    in_channels = 3
    out_channels = 1
    
    # モデルの初期化
    model = LitSeqLiteTransformer(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        n_frames=n_frames,
        window_size=4,
        feature_dim=32,
        transformer_blocks=1,
        transformer_heads=2,
        num_log_images=1
    )
    
    print(f"LitModule初期化完了: {type(model).__name__}")
    
    # ダミーデータの作成
    frames = torch.randn(batch_size, n_frames, in_channels, img_size[0], img_size[1])
    heatmaps = torch.randn(batch_size, n_frames, img_size[0], img_size[1])
    dummy_data = torch.zeros(batch_size, 2)
    
    batch = (frames, heatmaps, dummy_data, dummy_data)
    
    print(f"入力データ形状:")
    print(f"  frames: {frames.shape}")
    print(f"  heatmaps: {heatmaps.shape}")
    
    # モデルの順伝播テスト
    model.eval()
    with torch.no_grad():
        output = model(frames)
        print(f"出力形状: {output.shape}")
        
        # 期待される形状かチェック
        expected_shape = (batch_size, n_frames, out_channels, img_size[0], img_size[1])
        assert output.shape == expected_shape, f"期待される形状 {expected_shape}, 実際の形状 {output.shape}"
        
        print("✓ 順伝播テスト成功")
    
    # 学習ステップのテスト（勾配計算なし）
    model.train()
    with torch.no_grad():
        loss = model._common_step(batch, "train")
        print(f"損失値: {loss.item():.4f}")
        print("✓ 学習ステップテスト成功")
    
    # 検証ステップのテスト
    model.eval()
    with torch.no_grad():
        val_loss = model._common_step(batch, "val")
        print(f"検証損失値: {val_loss.item():.4f}")
        print("✓ 検証ステップテスト成功")
    
    print("✓ 全てのテストが成功しました！")


if __name__ == "__main__":
    test_lit_seq_lite_transformer() 