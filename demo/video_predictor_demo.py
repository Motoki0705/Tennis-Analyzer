#!/usr/bin/env python
"""
VideoPredictor Demo - 動画予測器デモ実行スクリプト
==================================================

このスクリプトは、src/multi/streaming_overlayer/video_predictor.py を使用して
テニス動画に対してボール・コート・ポーズの並列推論を実行するデモです。

使用方法:
    python demo/video_predictor_demo.py --input_path datasets/test/input.mp4 --output_path outputs/demo_output.mp4

または、設定ファイルを指定して実行:
    python demo/video_predictor_demo.py --config_path configs/infer/infer.yaml --input_path datasets/test/input.mp4
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

# カレントディレクトリをPythonパスに追加
sys.path.append('.')

from src.multi.streaming_overlayer.video_predictor import VideoPredictor

# ロガー設定
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def instantiate_model(cfg: DictConfig, task: str) -> torch.nn.Module:
    """
    タスクに応じてモデルをインスタンス化します。
    
    Args:
        cfg (DictConfig): 全体設定
        task (str): タスク名 ('ball', 'court', 'player', 'pose', 'event')
        
    Returns:
        torch.nn.Module: インスタンス化されたモデル
    """
    task_cfg = cfg[task]
    target_class = task_cfg.get("_target_", "")
    
    if not target_class:
        raise ValueError(f"_target_ is required for task '{task}'")
    
    logger.info(f"📥 {task} モデルをロード中: {target_class}")
    
    # transformersモデルの場合（poseタスクなど）
    if target_class.startswith("transformers."):
        logger.info(f"🤗 transformers.from_pretrained を使用: {task}")
        model = instantiate(task_cfg)
        return model
    
    # LightningModuleの場合（ball, court, player, eventタスクなど）
    ckpt_path = task_cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(f"ckpt_path is required for LightningModule task '{task}'")
    
    # モジュールパスとクラス名を分離
    try:
        module_path, class_name = target_class.rsplit('.', 1)
    except ValueError:
        raise ValueError(f"Invalid _target_ format for task '{task}': {target_class}")
    
    try:
        # 動的インポート
        import importlib
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)
        
        # チェックポイントからロード
        ckpt_abs = to_absolute_path(ckpt_path)
        logger.info(f"💾 チェックポイントからロード: {ckpt_abs}")
        
        if not Path(ckpt_abs).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs}")
            
        model = model_cls.load_from_checkpoint(ckpt_abs)
        return model
        
    except Exception as e:
        logger.error(f"❌ {task} モデルの読み込みに失敗: {e}")
        raise


def create_ball_predictor(cfg: DictConfig, device: str, use_half: bool):
    """ボール予測器を作成"""
    logger.info("🎾 ボール予測器を初期化中...")
    ball_model = instantiate_model(cfg, "ball").to(device)
    
    return instantiate(
        cfg.predictors.ball,
        litmodule=ball_model,
        device=device,
        use_half=use_half
    )


def create_court_predictor(cfg: DictConfig, device: str, use_half: bool):
    """コート予測器を作成"""
    logger.info("🏟️ コート予測器を初期化中...")
    court_model = instantiate_model(cfg, "court").to(device)
    
    return instantiate(
        cfg.predictors.court,
        litmodule=court_model,
        device=device,
        use_half=use_half
    )


def create_pose_predictor(cfg: DictConfig, device: str, use_half: bool):
    """ポーズ予測器を作成"""
    logger.info("🤸 ポーズ予測器を初期化中...")
    
    # 検出器（player）はLightningModuleを使用
    player_model = instantiate_model(cfg, "player").to(device)
    det_processor = instantiate(cfg.processors.player)
    
    # ポーズ推定（pose）はTransformersを使用
    pose_model = instantiate_model(cfg, "pose").to(device)
    pose_processor = instantiate(cfg.processors.pose)
    
    return instantiate(
        cfg.predictors.pose,
        det_litmodule=player_model,
        det_processor=det_processor,
        pose_litmodule=pose_model,
        pose_processor=pose_processor,
        device=device,
        use_half=use_half
    )


def load_config(config_path: str = None) -> DictConfig:
    """設定ファイルを読み込む"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "infer" / "infer.yaml"
    
    logger.info(f"📋 設定ファイルを読み込み中: {config_path}")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    # 必要な設定をデフォルト値で補完
    if "common" not in cfg:
        cfg.common = {}
    cfg.common.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.common.setdefault("use_half", True)
    cfg.common.setdefault("batch_size", 16)
    
    return cfg


def setup_video_predictor(cfg: DictConfig, input_path: str, output_path: str) -> VideoPredictor:
    """VideoPredictor を設定・初期化"""
    device = cfg.common.device
    use_half = cfg.common.use_half
    
    logger.info(f"🖥️ デバイス: {device}")
    logger.info(f"📊 Half precision: {use_half}")
    
    # 各予測器の初期化
    ball_predictor = create_ball_predictor(cfg, device, use_half)
    court_predictor = create_court_predictor(cfg, device, use_half)
    pose_predictor = create_pose_predictor(cfg, device, use_half)
    
    # 処理間隔とバッチサイズの設定
    streaming_overlayer_cfg = cfg.predictors.streaming_overlayer
    intervals = streaming_overlayer_cfg.get("intervals", {"ball": 1, "court": 30, "pose": 5})
    batch_sizes = streaming_overlayer_cfg.get("batch_sizes", {"ball": 16, "court": 16, "pose": 16})
    debug = streaming_overlayer_cfg.get("debug", False)
    
    logger.info(f"⏱️ 処理間隔: {intervals}")
    logger.info(f"📦 バッチサイズ: {batch_sizes}")
    
    # VideoPredictor の初期化
    video_predictor = VideoPredictor(
        ball_predictor=ball_predictor,
        court_predictor=court_predictor,
        pose_predictor=pose_predictor,
        intervals=intervals,
        batch_sizes=batch_sizes,
        debug=debug,
        custom_queue_configs=streaming_overlayer_cfg.get("custom_queue_configs"),
        hydra_queue_config=cfg.get("queue") if "queue" in cfg else None,
        max_preload_frames=64,
        enable_performance_monitoring=True
    )
    
    return video_predictor


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="VideoPredictor Demo - テニス動画の並列推論デモ",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="入力動画ファイルのパス"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="出力動画ファイルのパス"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=None,
        help="設定ファイルのパス（省略時はデフォルト設定を使用）"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="使用するデバイス（cuda/cpu）"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="デバッグモードで実行"
    )
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not Path(args.input_path).exists():
        logger.error(f"❌ 入力ファイルが見つかりません: {args.input_path}")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("🚀 VideoPredictor デモを開始します...")
        
        # 設定ファイルの読み込み
        cfg = load_config(args.config_path)
        
        # デバイス設定の上書き
        if args.device:
            cfg.common.device = args.device
        
        # デバッグ設定の上書き
        if args.debug:
            cfg.predictors.streaming_overlayer.debug = True
        
        # VideoPredictor の設定・初期化
        video_predictor = setup_video_predictor(cfg, args.input_path, args.output_path)
        
        # 動画処理の実行
        logger.info(f"📹 動画処理を開始: {args.input_path} → {args.output_path}")
        video_predictor.run(args.input_path, args.output_path)
        
        # パフォーマンス結果の表示
        metrics = video_predictor.get_performance_metrics()
        logger.info("📊 処理完了！パフォーマンス結果:")
        logger.info(f"  • 総処理フレーム数: {metrics.get('total_frames_processed', 'N/A')}")
        logger.info(f"  • 総処理時間: {metrics.get('total_processing_time', 'N/A'):.2f} 秒")
        logger.info(f"  • 平均FPS: {metrics.get('frames_per_second', 'N/A'):.2f}")
        
        logger.info(f"✅ 処理が完了しました！出力ファイル: {args.output_path}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ ユーザーによって処理が中断されました")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ 処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 