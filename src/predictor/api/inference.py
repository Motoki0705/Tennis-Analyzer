#!/usr/bin/env python3
"""
Tennis Ball Detection Inference API
===================================

🎾 テニスボール検出推論統合エントリポイント

このスクリプトは、テニス動画からのボール検出、可視化、統計出力までを
ワンストップで実行するためのコマンドラインインターフェースです。

Features:
- 🤖 複数検出器対応 (LiteTrackNet, WASB-SBDT)
- 🚀 並列処理パイプライン
- 🎨 高品質可視化
- 📊 統計情報出力
- ⚡ リアルタイム〜バッチ処理

Usage:
    python -m src.predictor.api.inference \
        --video input.mp4 \
        --output output.mp4 \
        --model_path checkpoints/model.ckpt \
        --model_type lite_tracknet \
        --config high_performance

Examples:
    # 基本実行
    python -m src.predictor.api.inference \
        --video tennis_match.mp4 \
        --output annotated_match.mp4 \
        --model_path checkpoints/lite_tracknet.ckpt

    # 高性能並列処理
    python -m src.predictor.api.inference \
        --video long_match.mp4 \
        --output result.mp4 \
        --model_path models/wasb_sbdt.pth \
        --model_type wasb_sbdt \
        --config high_performance \
        --batch_size 16 \
        --num_workers 8

    # カスタム可視化
    python -m src.predictor.api.inference \
        --video input.mp4 \
        --output stylized.mp4 \
        --model_path model.ckpt \
        --ball_radius 15 \
        --trajectory_length 30 \
        --enable_prediction \
        --stats_output stats.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.predictor import (
    VideoPipeline, 
    create_ball_detector,
    VisualizationConfig,
    HIGH_PERFORMANCE_CONFIG,
    MEMORY_EFFICIENT_CONFIG,
    REALTIME_CONFIG,
    DEBUG_CONFIG
)
from src.utils.logging_utils import setup_logging


def setup_arguments() -> argparse.ArgumentParser:
    """コマンドライン引数設定"""
    parser = argparse.ArgumentParser(
        description="Tennis Ball Detection Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # === 入出力設定 ===
    io_group = parser.add_argument_group('Input/Output Settings')
    io_group.add_argument(
        '--video', '-v', type=str, required=True,
        help='入力動画ファイルパス'
    )
    io_group.add_argument(
        '--output', '-o', type=str, required=True,
        help='出力動画ファイルパス'
    )
    io_group.add_argument(
        '--stats_output', type=str,
        help='統計情報出力ファイルパス (JSON形式)'
    )
    
    # === モデル設定 ===
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument(
        '--model_path', '-m', type=str, required=True,
        help='モデルファイルパス (.ckpt または .pth)'
    )
    model_group.add_argument(
        '--model_type', type=str, choices=['lite_tracknet', 'wasb_sbdt', 'auto'],
        default='auto',
        help='モデルタイプ (auto: 拡張子から自動判定)'
    )
    model_group.add_argument(
        '--config_path', type=str,
        help='モデル設定ファイルパス'
    )
    model_group.add_argument(
        '--device', type=str, choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='実行デバイス (auto: GPU自動検出)'
    )
    
    # === パイプライン設定 ===
    pipeline_group = parser.add_argument_group('Pipeline Settings')
    pipeline_group.add_argument(
        '--config', type=str, 
        choices=['high_performance', 'memory_efficient', 'realtime', 'debug'],
        default='high_performance',
        help='パイプライン設定プリセット'
    )
    pipeline_group.add_argument(
        '--batch_size', type=int, default=8,
        help='バッチサイズ'
    )
    pipeline_group.add_argument(
        '--num_workers', type=int, default=4,
        help='ワーカースレッド数'
    )
    pipeline_group.add_argument(
        '--queue_size', type=int, default=100,
        help='キューサイズ'
    )
    
    # === 可視化設定 ===
    vis_group = parser.add_argument_group('Visualization Settings')
    vis_group.add_argument(
        '--ball_radius', type=int, default=8,
        help='ボール描画半径'
    )
    vis_group.add_argument(
        '--trajectory_length', type=int, default=20,
        help='軌跡表示フレーム数'
    )
    vis_group.add_argument(
        '--enable_smoothing', action='store_true',
        help='位置スムージング有効化'
    )
    vis_group.add_argument(
        '--enable_prediction', action='store_true',
        help='位置予測表示有効化'
    )
    vis_group.add_argument(
        '--confidence_threshold', type=float, default=0.5,
        help='信頼度閾値'
    )
    
    # === システム設定 ===
    sys_group = parser.add_argument_group('System Settings')
    sys_group.add_argument(
        '--log_level', type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='ログレベル'
    )
    sys_group.add_argument(
        '--async_processing', action='store_true',
        help='非同期処理モード'
    )
    sys_group.add_argument(
        '--progress_interval', type=float, default=1.0,
        help='進捗表示間隔（秒）'
    )
    
    return parser


def get_pipeline_config(config_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    """パイプライン設定取得"""
    # ベース設定選択
    if config_name == 'high_performance':
        base_config = HIGH_PERFORMANCE_CONFIG
    elif config_name == 'memory_efficient':
        base_config = MEMORY_EFFICIENT_CONFIG
    elif config_name == 'realtime':
        base_config = REALTIME_CONFIG
    elif config_name == 'debug':
        base_config = DEBUG_CONFIG
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    # カスタマイズ適用
    config = base_config.copy()
    if args.batch_size != 8:
        config['batch_size'] = args.batch_size
    if args.num_workers != 4:
        config['num_workers'] = args.num_workers
    if args.queue_size != 100:
        config['queue_size'] = args.queue_size
        
    return config


def get_detector_config(args: argparse.Namespace) -> Dict[str, Any]:
    """検出器設定取得"""
    # デバイス自動検出
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # モデルタイプ自動判定
    if args.model_type == 'auto':
        if args.model_path.endswith('.ckpt'):
            model_type = 'lite_tracknet'
        elif args.model_path.endswith('.pth'):
            model_type = 'wasb_sbdt'
        else:
            raise ValueError(f"Cannot auto-detect model type from: {args.model_path}")
    else:
        model_type = args.model_type
    
    config = {
        'model_path': args.model_path,
        'model_type': model_type,
        'device': device,
    }
    
    if args.config_path:
        config['config_path'] = args.config_path
        
    return config


def get_visualization_config(args: argparse.Namespace) -> VisualizationConfig:
    """可視化設定取得"""
    return VisualizationConfig(
        ball_radius=args.ball_radius,
        trajectory_length=args.trajectory_length,
        enable_smoothing=args.enable_smoothing,
        enable_prediction=args.enable_prediction,
        confidence_threshold=args.confidence_threshold
    )


def calculate_statistics(result: Dict[str, Any]) -> Dict[str, Any]:
    """統計情報計算"""
    stats = {
        'processing_info': {
            'total_frames': result.get('total_frames', 0),
            'processed_frames': result.get('processed_frames', 0),
            'processing_time': result.get('processing_time', 0.0),
            'average_fps': result.get('average_fps', 0.0),
        },
        'detection_stats': {
            'total_detections': 0,
            'frames_with_detection': 0,
            'average_confidence': 0.0,
            'detection_rate': 0.0,
        },
        'performance_stats': {
            'gpu_utilization': result.get('gpu_utilization', 0.0),
            'memory_usage': result.get('memory_usage', 0.0),
            'queue_efficiency': result.get('queue_efficiency', 0.0),
        }
    }
    
    # 検出統計計算
    detections = result.get('detections', {})
    if detections:
        total_detections = sum(len(det_list) for det_list in detections.values())
        frames_with_detection = sum(1 for det_list in detections.values() if det_list)
        
        all_confidences = []
        for det_list in detections.values():
            all_confidences.extend([det[2] for det in det_list if len(det) > 2])
        
        stats['detection_stats'].update({
            'total_detections': total_detections,
            'frames_with_detection': frames_with_detection,
            'average_confidence': sum(all_confidences) / len(all_confidences) if all_confidences else 0.0,
            'detection_rate': frames_with_detection / len(detections) if detections else 0.0,
        })
    
    return stats


def save_statistics(stats: Dict[str, Any], output_path: str):
    """統計情報保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logging.info(f"統計情報を保存: {output_path}")


def display_progress(result, progress_interval: float):
    """進捗表示"""
    while not result.is_completed():
        progress = result.get_progress()
        current_fps = result.get_current_fps()
        logging.info(f"進捗: {progress*100:.1f}% | 現在のFPS: {current_fps:.1f}")
        time.sleep(progress_interval)


def main():
    """メインエントリポイント"""
    # 引数解析
    parser = setup_arguments()
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(level=args.log_level)
    logging.info("🎾 Tennis Ball Detection Inference System 開始")
    
    try:
        # 入力検証
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"入力動画が見つかりません: {args.video}")
        
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {args.model_path}")
        
        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # 設定取得
        pipeline_config = get_pipeline_config(args.config, args)
        detector_config = get_detector_config(args)
        vis_config = get_visualization_config(args)
        
        logging.info(f"入力動画: {args.video}")
        logging.info(f"出力動画: {args.output}")
        logging.info(f"モデル: {detector_config['model_type']} ({args.model_path})")
        logging.info(f"デバイス: {detector_config['device']}")
        logging.info(f"パイプライン設定: {args.config}")
        
        # パイプライン作成
        pipeline = VideoPipeline(pipeline_config)
        
        # 処理実行
        start_time = time.time()
        
        if args.async_processing:
            # 非同期処理
            logging.info("非同期処理モードで実行中...")
            result = pipeline.process_video_async(
                video_path=args.video,
                detector_config=detector_config,
                output_path=args.output,
                vis_config=vis_config
            )
            
            # 進捗表示
            display_progress(result, args.progress_interval)
            final_result = result.get_result()
            
        else:
            # 同期処理
            logging.info("同期処理モードで実行中...")
            final_result = pipeline.process_video(
                video_path=args.video,
                detector_config=detector_config,
                output_path=args.output,
                vis_config=vis_config
            )
        
        processing_time = time.time() - start_time
        
        # 結果表示
        logging.info("🎯 処理完了!")
        logging.info(f"処理時間: {processing_time:.2f}秒")
        logging.info(f"平均FPS: {final_result.get('average_fps', 0):.2f}")
        logging.info(f"総フレーム数: {final_result.get('total_frames', 0)}")
        logging.info(f"検出フレーム数: {sum(1 for det_list in final_result.get('detections', {}).values() if det_list)}")
        
        # 統計情報保存
        if args.stats_output:
            stats = calculate_statistics(final_result)
            save_statistics(stats, args.stats_output)
        
        logging.info(f"出力動画: {args.output}")
        
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 