#!/usr/bin/env python3
"""
Hydra設定システムを使用したキューシステムのデモ

このデモはHydra設定ファイルからキューシステムを構成する方法を示します。
"""
import sys
import os
from pathlib import Path
from unittest.mock import Mock
from omegaconf import DictConfig, OmegaConf

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.multi.streaming_overlayer.video_predictor import VideoPredictor
from src.multi.streaming_overlayer.config_utils import (
    create_queue_configs_from_hydra_config,
    validate_queue_config,
    log_queue_configuration
)


def create_mock_predictors():
    """モック予測器を作成"""
    mock_ball = Mock()
    mock_court = Mock()
    mock_pose = Mock()
    
    # Ball predictor
    mock_ball.preprocess.return_value = Mock()
    mock_ball.inference.return_value = [Mock()]
    mock_ball.overlay.return_value = Mock()
    
    # Court predictor
    mock_court.preprocess.return_value = (Mock(), Mock())
    mock_court.inference.return_value = Mock()
    mock_court.postprocess.return_value = ([Mock()], Mock())
    mock_court.overlay.return_value = Mock()
    
    # Pose predictor
    mock_pose.preprocess_detection.return_value = {"pixel_values": Mock()}
    mock_pose.inference_detection.return_value = {"pred_boxes": Mock()}
    mock_pose.postprocess_detection.return_value = ([Mock()], [Mock()], [Mock()], [Mock()])
    mock_pose.preprocess_pose.return_value = {"inputs": Mock()}
    mock_pose.inference_pose.return_value = {"keypoints": Mock()}
    mock_pose.postprocess_pose.return_value = [[Mock()]]
    mock_pose.overlay.return_value = Mock()
    
    return mock_ball, mock_court, mock_pose


def demo_default_config():
    """デフォルト設定のデモ"""
    print("=" * 60)
    print("1. デフォルト設定でのVideoPredictor初期化")
    print("=" * 60)
    
    # デフォルト設定を読み込み
    config_path = project_root / "configs" / "infer" / "queue" / "default.yaml"
    queue_config = OmegaConf.load(config_path)
    
    # 設定表示
    print("📋 読み込み設定:")
    log_queue_configuration(queue_config)
    
    # モック予測器作成
    ball_pred, court_pred, pose_pred = create_mock_predictors()
    
    # VideoPredictor初期化
    try:
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 1, "court": 30, "pose": 5},
            batch_sizes={"ball": 16, "court": 16, "pose": 16},
            debug=True,
            hydra_queue_config=queue_config
        )
        
        print("✅ VideoPredictor初期化成功")
        
        # キュー状態確認
        status = video_predictor.get_queue_status_with_settings()
        print(f"📊 初期化後のキュー状態: {len(status['workers'])}個のワーカー")
        
    except Exception as e:
        print(f"❌ 初期化失敗: {e}")
    
    print()


def demo_high_performance_config():
    """高性能設定のデモ"""
    print("=" * 60)
    print("2. 高性能設定でのVideoPredictor初期化")
    print("=" * 60)
    
    # 高性能設定を読み込み
    config_path = project_root / "configs" / "infer" / "queue" / "high_performance.yaml"
    queue_config = OmegaConf.load(config_path)
    
    # 設定表示
    print("📋 読み込み設定:")
    log_queue_configuration(queue_config)
    
    # モック予測器作成
    ball_pred, court_pred, pose_pred = create_mock_predictors()
    
    # VideoPredictor初期化
    try:
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 1, "court": 10, "pose": 3},  # 高頻度処理
            batch_sizes={"ball": 32, "court": 32, "pose": 32},  # 大バッチ
            debug=True,
            hydra_queue_config=queue_config
        )
        
        print("✅ VideoPredictor初期化成功（高性能モード）")
        
        # キュー状態確認
        status = video_predictor.get_queue_status_with_settings()
        
        # パフォーマンス設定確認
        perf_settings = video_predictor.performance_settings
        print(f"🚀 パフォーマンス設定: {list(perf_settings.keys())}")
        
    except Exception as e:
        print(f"❌ 初期化失敗: {e}")
    
    print()


def demo_custom_config():
    """カスタム設定のデモ"""
    print("=" * 60)
    print("3. カスタム設定でのVideoPredictor初期化")
    print("=" * 60)
    
    # カスタム設定を読み込み
    config_path = project_root / "configs" / "infer" / "queue" / "custom_example.yaml"
    queue_config = OmegaConf.load(config_path)
    
    # 設定表示
    print("📋 読み込み設定:")
    log_queue_configuration(queue_config)
    
    # モック予測器作成
    ball_pred, court_pred, pose_pred = create_mock_predictors()
    
    # 追加のカスタム設定
    additional_custom_configs = {
        "experimental_queue": {
            "maxsize": 256,
            "queue_type": "Queue",
            "description": "実験的大容量キュー"
        }
    }
    
    # VideoPredictor初期化
    try:
        video_predictor = VideoPredictor(
            ball_predictor=ball_pred,
            court_predictor=court_pred,
            pose_predictor=pose_pred,
            intervals={"ball": 2, "court": 20, "pose": 4},
            batch_sizes={"ball": 24, "court": 24, "pose": 24},
            debug=True,
            custom_queue_configs=additional_custom_configs,
            hydra_queue_config=queue_config
        )
        
        print("✅ VideoPredictor初期化成功（カスタムモード）")
        
        # 特殊処理設定確認
        if hasattr(queue_config, 'special_processing'):
            special = OmegaConf.to_container(queue_config.special_processing)
            print(f"🎯 特殊処理設定: {special}")
        
    except Exception as e:
        print(f"❌ 初期化失敗: {e}")
    
    print()


def demo_config_validation():
    """設定検証のデモ"""
    print("=" * 60)
    print("4. 設定検証デモ")
    print("=" * 60)
    
    # 正常な設定
    print("📋 正常な設定の検証:")
    config_path = project_root / "configs" / "infer" / "queue" / "default.yaml"
    valid_config = OmegaConf.load(config_path)
    
    if validate_queue_config(valid_config):
        print("✅ デフォルト設定は有効です")
    else:
        print("❌ デフォルト設定が無効です")
    
    # 不正な設定（意図的にエラーを作成）
    print("\n📋 不正な設定の検証:")
    invalid_config = OmegaConf.create({
        "base_queue_sizes": {
            "preprocess": -1,  # 無効なサイズ
            "inference": 16,
            "postprocess": 16,
            # "results": 100  # 必須キューが不足
        },
        "queue_types": {
            "preprocess": "InvalidType"  # 無効なタイプ
        }
    })
    
    if validate_queue_config(invalid_config):
        print("✅ 設定は有効です")
    else:
        print("❌ 設定が無効です（期待通り）")
    
    print()


def demo_config_conversion():
    """設定変換のデモ"""
    print("=" * 60)
    print("5. 設定変換デモ")
    print("=" * 60)
    
    # Hydra設定を読み込み
    config_path = project_root / "configs" / "infer" / "queue" / "default.yaml"
    hydra_config = OmegaConf.load(config_path)
    
    # QueueManager形式に変換
    queue_configs = create_queue_configs_from_hydra_config(hydra_config)
    
    print("📋 変換されたキュー設定:")
    for queue_name, config in queue_configs.items():
        print(f"  {queue_name}:")
        print(f"    サイズ: {config['maxsize']}")
        print(f"    タイプ: {config['queue_type']}")
        print(f"    説明: {config['description']}")
    
    print(f"\n✅ {len(queue_configs)}個のキュー設定を変換しました")
    print()


def main():
    """メインデモ実行"""
    print("🚀 Hydra設定システム キューデモ")
    print("Tennis AnalyzerのキューシステムをHydra設定で管理")
    print()
    
    try:
        demo_default_config()
        demo_high_performance_config()
        demo_custom_config()
        demo_config_validation()
        demo_config_conversion()
        
        print("=" * 60)
        print("🎉 Hydra設定デモ完了")
        print("=" * 60)
        
        print("\n📝 利用可能な設定ファイル:")
        queue_config_dir = project_root / "configs" / "infer" / "queue"
        for config_file in queue_config_dir.glob("*.yaml"):
            print(f"  - {config_file.stem}: {config_file}")
        
        print("\n🔧 使用方法:")
        print("  configs/infer/infer.yaml で queue: [設定名] を指定")
        print("  例: queue: high_performance")
        
    except Exception as e:
        print(f"❌ デモ実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 