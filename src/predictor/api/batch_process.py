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
        --config-path ../../configs/infer \
        --config-name batch_process \
        io.input_dir=videos/ \
        io.output_dir=results/ \
        model.model_path=checkpoints/model.ckpt \
        batch.parallel_jobs=4

Examples:
    # ディレクトリ一括処理
    python -m src.predictor.api.batch_process \
        --config-path ../../configs/infer \
        --config-name batch_process \
        io.input_dir=tennis_videos/ \
        io.output_dir=annotated_videos/ \
        model.model_path=models/wasb_sbdt.pth \
        model=wasb_sbdt

    # 並列処理 + 統計レポート
    python -m src.predictor.api.batch_process \
        --config-path ../../configs/infer \
        --config-name batch_process \
        io.input_dir=videos/ \
        io.output_dir=results/ \
        model.model_path=model.ckpt \
        batch.parallel_jobs=4 \
        io.report_path=batch_report.json \
        pipeline=high_performance

    # ファイルリスト処理
    python -m src.predictor.api.batch_process \
        --config-path ../../configs/infer \
        --config-name batch_process \
        io.input_list=video_list.txt \
        io.output_dir=results/ \
        model.model_path=model.pth \
        batch.continue_on_error=true
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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


def collect_video_files(cfg: DictConfig) -> List[str]:
    """処理対象動画ファイル収集"""
    video_files = []
    
    if cfg.io.input_dir:
        # ディレクトリから収集
        input_path = Path(cfg.io.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが見つかりません: {cfg.io.input_dir}")
        
        for pattern in cfg.io.file_patterns:
            video_files.extend(input_path.glob(pattern))
            
    elif cfg.io.input_list:
        # リストファイルから収集
        if not os.path.exists(cfg.io.input_list):
            raise FileNotFoundError(f"入力リストファイルが見つかりません: {cfg.io.input_list}")
        
        with open(cfg.io.input_list, 'r', encoding='utf-8') as f:
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


def get_pipeline_config(cfg: DictConfig) -> Dict[str, Any]:
    """パイプライン設定取得"""
    if cfg.pipeline.type == 'high_performance':
        base_config = HIGH_PERFORMANCE_CONFIG
    elif cfg.pipeline.type == 'memory_efficient':
        base_config = MEMORY_EFFICIENT_CONFIG
    elif cfg.pipeline.type == 'realtime':
        base_config = REALTIME_CONFIG
    elif cfg.pipeline.type == 'debug':
        base_config = DEBUG_CONFIG
    else:
        raise ValueError(f"Unknown pipeline type: {cfg.pipeline.type}")
    
    # カスタマイズ適用
    config = base_config.copy()
    config.update({
        'batch_size': cfg.pipeline.batch_size,
        'num_workers': cfg.pipeline.num_workers,
        'queue_size': cfg.pipeline.queue_size,
    })
        
    return config


def get_detector_config(cfg: DictConfig) -> Dict[str, Any]:
    """検出器設定取得"""
    # デバイス自動検出
    if cfg.model.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.model.device
    
    # モデルタイプ自動判定
    if cfg.model.type == 'auto':
        if cfg.model.model_path.endswith('.ckpt'):
            model_type = 'lite_tracknet'
        elif cfg.model.model_path.endswith('.pth'):
            model_type = 'wasb_sbdt'
        else:
            raise ValueError(f"Cannot auto-detect model type from: {cfg.model.model_path}")
    else:
        model_type = cfg.model.type
    
    config = {
        'model_path': cfg.model.model_path,
        'model_type': model_type,
        'device': device,
    }
    
    if cfg.model.config_path:
        config['config_path'] = cfg.model.config_path
        
    return config


def get_visualization_config(cfg: DictConfig) -> VisualizationConfig:
    """可視化設定取得"""
    return VisualizationConfig(
        ball_radius=cfg.visualization.ball_radius,
        trajectory_length=cfg.visualization.trajectory_length,
        enable_smoothing=cfg.visualization.enable_smoothing,
        enable_prediction=cfg.visualization.enable_prediction,
        confidence_threshold=cfg.visualization.confidence_threshold
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


@hydra.main(version_base=None, config_path="../../configs/infer", config_name="batch_process")
def validate_batch_config(cfg: DictConfig) -> None:
    """バッチ処理設定検証"""
    # 入力ソース検証
    if not cfg.io.input_dir and not cfg.io.input_list:
        raise ValueError("Either input_dir or input_list must be specified")
    
    # 必須フィールド検証
    required_fields = [
        ('io.output_dir', 'output directory'),
        ('model.model_path', 'model file path')
    ]
    
    for field_path, description in required_fields:
        field_value = cfg
        for key in field_path.split('.'):
            field_value = getattr(field_value, key, None)
        if not field_value:
            raise ValueError(f"{field_path} ({description}) is required but not provided")
    
    # ファイル存在確認
    if not os.path.exists(cfg.model.model_path):
        raise FileNotFoundError(f"Model file not found: {cfg.model.model_path}")
    
    if cfg.io.input_list and not os.path.exists(cfg.io.input_list):
        raise FileNotFoundError(f"Input list file not found: {cfg.io.input_list}")
    
    if cfg.io.input_dir and not os.path.exists(cfg.io.input_dir):
        raise FileNotFoundError(f"Input directory not found: {cfg.io.input_dir}")


def main(cfg: DictConfig) -> None:
    """メインエントリポイント"""
    # ログ設定
    logging.basicConfig(
        level=getattr(logging, cfg.system.log_level),
        format='[%(levelname)s] %(message)s'
    )
    logging.info("🎾 Tennis Ball Detection Batch Processing System 開始")
    
    try:
        # 設定検証
        validate_batch_config(cfg)
        
        # 出力ディレクトリ作成
        os.makedirs(cfg.io.output_dir, exist_ok=True)
        
        # 動画ファイル収集
        video_files = collect_video_files(cfg)
        if not video_files:
            raise ValueError("処理対象の動画ファイルが見つかりません")
        
        # 設定取得
        pipeline_config = get_pipeline_config(cfg)
        detector_config = get_detector_config(cfg)
        vis_config = get_visualization_config(cfg)
        
        logging.info(f"設定: {OmegaConf.to_yaml(cfg)}")
        logging.info(f"モデル: {detector_config['model_type']} ({cfg.model.model_path})")
        logging.info(f"デバイス: {detector_config['device']}")
        logging.info(f"パイプライン設定: {cfg.pipeline.type}")
        logging.info(f"並列ジョブ数: {cfg.batch.parallel_jobs}")
        
        # 処理実行
        start_time = time.time()
        results = []
        
        if cfg.batch.parallel_jobs == 1:
            # シーケンシャル処理
            pipeline = VideoPipeline(pipeline_config)
            
            for i, video_path in enumerate(video_files, 1):
                output_path = get_output_path(video_path, cfg.io.output_dir)
                logging.info(f"処理中 ({i}/{len(video_files)}): {video_path}")
                
                result = process_single_video(
                    video_path, output_path, pipeline,
                    detector_config, vis_config, cfg.batch.overwrite
                )
                results.append(result)
                
                if result['status'] == 'failed' and not cfg.batch.continue_on_error:
                    logging.error("エラー発生により処理を中断します")
                    break
                    
        else:
            # 並列処理
            logging.info(f"並列処理開始 ({cfg.batch.parallel_jobs}並列)")
            
            def process_wrapper(video_path: str) -> Dict[str, Any]:
                pipeline = VideoPipeline(pipeline_config)
                output_path = get_output_path(video_path, cfg.io.output_dir)
                return process_single_video(
                    video_path, output_path, pipeline,
                    detector_config, vis_config, cfg.batch.overwrite
                )
            
            with ThreadPoolExecutor(max_workers=cfg.batch.parallel_jobs) as executor:
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
                    
                    if result['status'] == 'failed' and not cfg.batch.continue_on_error:
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
        if cfg.io.report_path:
            report = generate_batch_report(results)
            save_batch_report(report, cfg.io.report_path)
        
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