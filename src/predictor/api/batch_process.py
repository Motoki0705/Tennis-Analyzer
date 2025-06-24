#!/usr/bin/env python3
"""
Tennis Ball Detection Batch Processing API
==========================================

🎾 テニスボール検出バッチ処理システム

複数のテニス動画を効率的に一括処理するためのコマンドラインインターフェースです。
並列処理、進捗管理、エラー回復機能を提供します。

Features:
- 📁 ディレクトリ一括処理
- 🔄 並列バッチ処理
- 📊 統一統計情報
- 🛡️ エラー回復・続行
- 📈 詳細進捗レポート

Usage:
    python -m src.predictor.api.batch_process \
        --input_dir videos/ \
        --output_dir results/ \
        --model_path checkpoints/model.ckpt \
        --parallel_jobs 4

Examples:
    # ディレクトリ一括処理
    python -m src.predictor.api.batch_process \
        --input_dir tennis_videos/ \
        --output_dir annotated_videos/ \
        --model_path models/wasb_sbdt.pth \
        --model_type wasb_sbdt

    # 並列処理 + 統計レポート
    python -m src.predictor.api.batch_process \
        --input_dir videos/ \
        --output_dir results/ \
        --model_path model.ckpt \
        --parallel_jobs 4 \
        --report_path batch_report.json \
        --config high_performance

    # ファイルリスト処理
    python -m src.predictor.api.batch_process \
        --input_list video_list.txt \
        --output_dir results/ \
        --model_path model.pth \
        --continue_on_error
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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
        description="Tennis Ball Detection Batch Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # === 入出力設定 ===
    io_group = parser.add_argument_group('Input/Output Settings')
    input_group = io_group.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir', type=str,
        help='入力動画ディレクトリパス'
    )
    input_group.add_argument(
        '--input_list', type=str,
        help='入力動画リストファイル (1行1ファイル)'
    )
    
    io_group.add_argument(
        '--output_dir', type=str, required=True,
        help='出力ディレクトリパス'
    )
    io_group.add_argument(
        '--report_path', type=str,
        help='バッチ処理レポート出力パス (JSON形式)'
    )
    io_group.add_argument(
        '--file_patterns', type=str, nargs='+',
        default=['*.mp4', '*.avi', '*.mov'],
        help='処理対象ファイルパターン'
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
    
    # === バッチ処理設定 ===
    batch_group = parser.add_argument_group('Batch Processing Settings')
    batch_group.add_argument(
        '--parallel_jobs', type=int, default=1,
        help='並列処理ジョブ数'
    )
    batch_group.add_argument(
        '--config', type=str, 
        choices=['high_performance', 'memory_efficient', 'realtime', 'debug'],
        default='memory_efficient',
        help='パイプライン設定プリセット'
    )
    batch_group.add_argument(
        '--continue_on_error', action='store_true',
        help='エラー発生時も他ファイル処理を続行'
    )
    batch_group.add_argument(
        '--overwrite', action='store_true',
        help='既存出力ファイルを上書き'
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
        '--progress_interval', type=float, default=5.0,
        help='進捗表示間隔（秒）'
    )
    
    return parser


def collect_video_files(args: argparse.Namespace) -> List[str]:
    """処理対象動画ファイル収集"""
    video_files = []
    
    if args.input_dir:
        # ディレクトリから収集
        input_path = Path(args.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが見つかりません: {args.input_dir}")
        
        for pattern in args.file_patterns:
            video_files.extend(input_path.glob(pattern))
            
    elif args.input_list:
        # リストファイルから収集
        if not os.path.exists(args.input_list):
            raise FileNotFoundError(f"入力リストファイルが見つかりません: {args.input_list}")
        
        with open(args.input_list, 'r', encoding='utf-8') as f:
            for line in f:
                video_path = line.strip()
                if video_path and os.path.exists(video_path):
                    video_files.append(Path(video_path))
    
    # 文字列に変換してソート
    video_files = sorted([str(f) for f in video_files])
    
    logging.info(f"処理対象動画ファイル: {len(video_files)}件")
    return video_files


def get_output_path(video_path: str, output_dir: str) -> str:
    """出力パス生成"""
    video_name = Path(video_path).stem
    return os.path.join(output_dir, f"{video_name}_annotated.mp4")


def process_single_video(
    video_path: str,
    output_path: str,
    pipeline: VideoPipeline,
    detector_config: Dict[str, Any],
    vis_config: VisualizationConfig,
    overwrite: bool = False
) -> Dict[str, Any]:
    """単一動画処理"""
    result = {
        'video_path': video_path,
        'output_path': output_path,
        'status': 'pending',
        'start_time': time.time(),
        'processing_time': 0.0,
        'error_message': None,
        'stats': {}
    }
    
    try:
        # 出力ファイル存在チェック
        if os.path.exists(output_path) and not overwrite:
            result['status'] = 'skipped'
            result['error_message'] = '出力ファイルが既に存在します'
            return result
        
        # 処理実行
        logging.info(f"処理開始: {video_path}")
        processing_result = pipeline.process_video(
            video_path=video_path,
            detector_config=detector_config,
            output_path=output_path,
            vis_config=vis_config
        )
        
        result['processing_time'] = time.time() - result['start_time']
        result['status'] = 'completed'
        result['stats'] = processing_result
        
        logging.info(f"処理完了: {video_path} ({result['processing_time']:.2f}秒)")
        
    except Exception as e:
        result['processing_time'] = time.time() - result['start_time']
        result['status'] = 'failed'
        result['error_message'] = str(e)
        
        logging.error(f"処理失敗: {video_path} - {e}")
    
    return result


def get_pipeline_config(config_name: str) -> Dict[str, Any]:
    """パイプライン設定取得"""
    if config_name == 'high_performance':
        return HIGH_PERFORMANCE_CONFIG
    elif config_name == 'memory_efficient':
        return MEMORY_EFFICIENT_CONFIG
    elif config_name == 'realtime':
        return REALTIME_CONFIG
    elif config_name == 'debug':
        return DEBUG_CONFIG
    else:
        raise ValueError(f"Unknown config: {config_name}")


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


def generate_batch_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """バッチ処理レポート生成"""
    total_files = len(results)
    completed_files = sum(1 for r in results if r['status'] == 'completed')
    failed_files = sum(1 for r in results if r['status'] == 'failed')
    skipped_files = sum(1 for r in results if r['status'] == 'skipped')
    
    total_processing_time = sum(r['processing_time'] for r in results)
    completed_results = [r for r in results if r['status'] == 'completed']
    
    # 統計計算
    total_frames = sum(r['stats'].get('total_frames', 0) for r in completed_results)
    total_detections = 0
    total_detection_frames = 0
    
    for r in completed_results:
        detections = r['stats'].get('detections', {})
        if detections:
            total_detections += sum(len(det_list) for det_list in detections.values())
            total_detection_frames += sum(1 for det_list in detections.values() if det_list)
    
    report = {
        'summary': {
            'total_files': total_files,
            'completed_files': completed_files,
            'failed_files': failed_files,
            'skipped_files': skipped_files,
            'success_rate': completed_files / total_files if total_files > 0 else 0.0,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / completed_files if completed_files > 0 else 0.0,
        },
        'detection_summary': {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'frames_with_detection': total_detection_frames,
            'detection_rate': total_detection_frames / total_frames if total_frames > 0 else 0.0,
        },
        'detailed_results': results
    }
    
    return report


def save_batch_report(report: Dict[str, Any], output_path: str):
    """バッチレポート保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logging.info(f"バッチレポートを保存: {output_path}")


def main():
    """メインエントリポイント"""
    # 引数解析
    parser = setup_arguments()
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(level=args.log_level)
    logging.info("🎾 Tennis Ball Detection Batch Processing System 開始")
    
    try:
        # 入力検証
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {args.model_path}")
        
        # 出力ディレクトリ作成
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 動画ファイル収集
        video_files = collect_video_files(args)
        if not video_files:
            raise ValueError("処理対象の動画ファイルが見つかりません")
        
        # 設定取得
        pipeline_config = get_pipeline_config(args.config)
        detector_config = get_detector_config(args)
        vis_config = get_visualization_config(args)
        
        logging.info(f"モデル: {detector_config['model_type']} ({args.model_path})")
        logging.info(f"デバイス: {detector_config['device']}")
        logging.info(f"パイプライン設定: {args.config}")
        logging.info(f"並列ジョブ数: {args.parallel_jobs}")
        
        # 処理実行
        start_time = time.time()
        results = []
        
        if args.parallel_jobs == 1:
            # シーケンシャル処理
            pipeline = VideoPipeline(pipeline_config)
            
            for i, video_path in enumerate(video_files, 1):
                output_path = get_output_path(video_path, args.output_dir)
                logging.info(f"処理中 ({i}/{len(video_files)}): {video_path}")
                
                result = process_single_video(
                    video_path, output_path, pipeline,
                    detector_config, vis_config, args.overwrite
                )
                results.append(result)
                
                if result['status'] == 'failed' and not args.continue_on_error:
                    logging.error("エラー発生により処理を中断します")
                    break
                    
        else:
            # 並列処理
            logging.info(f"並列処理開始 ({args.parallel_jobs}並列)")
            
            def process_wrapper(video_path: str) -> Dict[str, Any]:
                pipeline = VideoPipeline(pipeline_config)
                output_path = get_output_path(video_path, args.output_dir)
                return process_single_video(
                    video_path, output_path, pipeline,
                    detector_config, vis_config, args.overwrite
                )
            
            with ThreadPoolExecutor(max_workers=args.parallel_jobs) as executor:
                # タスク投入
                future_to_video = {
                    executor.submit(process_wrapper, video_path): video_path
                    for video_path in video_files
                }
                
                # 結果収集
                for future in as_completed(future_to_video):
                    result = future.result()
                    results.append(result)
                    
                    completed = len(results)
                    logging.info(f"進捗: {completed}/{len(video_files)} ({completed/len(video_files)*100:.1f}%)")
                    
                    if result['status'] == 'failed' and not args.continue_on_error:
                        logging.error("エラー発生により処理を中断します")
                        # 残りのタスクをキャンセル
                        for remaining_future in future_to_video:
                            remaining_future.cancel()
                        break
        
        total_time = time.time() - start_time
        
        # 結果集計
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        
        # 結果表示
        logging.info("🎯 バッチ処理完了!")
        logging.info(f"総処理時間: {total_time:.2f}秒")
        logging.info(f"処理結果: 完了={completed}, 失敗={failed}, スキップ={skipped}")
        logging.info(f"成功率: {completed/len(results)*100:.1f}%")
        
        # レポート保存
        if args.report_path:
            report = generate_batch_report(results)
            save_batch_report(report, args.report_path)
        
        # 失敗したファイル一覧表示
        failed_files = [r for r in results if r['status'] == 'failed']
        if failed_files:
            logging.warning(f"失敗ファイル ({len(failed_files)}件):")
            for result in failed_files:
                logging.warning(f"  {result['video_path']}: {result['error_message']}")
        
    except Exception as e:
        logging.error(f"バッチ処理でエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 