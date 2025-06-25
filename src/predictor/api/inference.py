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
        --config-path ../../configs/infer \
        --config-name inference \
        io.video=input.mp4 \
        io.output=output.mp4 \
        model.model_path=checkpoints/model.ckpt \
        model=lite_tracknet \
        pipeline=high_performance

Examples:
    # 基本実行
    python -m src.predictor.api.inference \
        --config-path ../../configs/infer \
        --config-name inference \
        io.video=tennis_match.mp4 \
        io.output=annotated_match.mp4 \
        model.model_path=checkpoints/lite_tracknet.ckpt

    # 高性能並列処理
    python -m src.predictor.api.inference \
        --config-path ../../configs/infer \
        --config-name inference \
        io.video=long_match.mp4 \
        io.output=result.mp4 \
        model.model_path=models/wasb_sbdt.pth \
        model=wasb_sbdt \
        pipeline=high_performance \
        pipeline.batch_size=16 \
        pipeline.num_workers=8

    # カスタム可視化
    python -m src.predictor.api.inference \
        --config-path ../../configs/infer \
        --config-name inference \
        io.video=input.mp4 \
        io.output=stylized.mp4 \
        model.model_path=model.ckpt \
        visualization=enhanced \
        io.stats_output=stats.json
"""

import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

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
from src.predictor.pipeline.config import PipelineConfig


def get_pipeline_config(cfg: DictConfig) -> PipelineConfig:
    """パイプライン設定取得"""
    try:
        # ベース設定選択
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
        
        # PipelineConfigオブジェクトをコピー
        config = copy.copy(base_config)
        
        # カスタム設定の値を取得（デフォルト値を提供）
        batch_size = getattr(cfg.pipeline, 'batch_size', 8)
        num_workers = getattr(cfg.pipeline, 'num_workers', 4)
        queue_size = getattr(cfg.pipeline, 'queue_size', 100)
        
        # PipelineConfigオブジェクトのパラメータを更新
        if hasattr(config, 'gpu_batch_size'):
            config.gpu_batch_size = batch_size
        if hasattr(config, 'num_workers'):
            config.num_workers = num_workers
        if hasattr(config, 'frame_buffer_size'):
            config.frame_buffer_size = queue_size
            
        # PipelineConfigオブジェクトを直接返す
        return config
    
    except Exception as e:
        # エラーが発生した場合のフォールバック設定
        logging.warning(f"Pipeline config error: {e}. Using default PipelineConfig.")
        fallback_config = PipelineConfig()
        # デフォルト設定をカスタマイズ
        fallback_config.gpu_batch_size = getattr(cfg.pipeline, 'batch_size', 8)
        fallback_config.num_workers = getattr(cfg.pipeline, 'num_workers', 4)
        fallback_config.frame_buffer_size = getattr(cfg.pipeline, 'queue_size', 100)
        return fallback_config


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


def validate_config(cfg: DictConfig) -> None:
    """設定検証"""
    required_fields = [
        ('io.video', 'input video file'),
        ('io.output', 'output video file'),
        ('model.model_path', 'model file path')
    ]
    
    for field_path, description in required_fields:
        field_value = cfg
        for key in field_path.split('.'):
            field_value = getattr(field_value, key, None)
        if not field_value:
            raise ValueError(f"{field_path} ({description}) is required but not provided")
    
    # ファイル存在確認
    if not os.path.exists(cfg.io.video):
        raise FileNotFoundError(f"Input video not found: {cfg.io.video}")
    
    if not os.path.exists(cfg.model.model_path):
        raise FileNotFoundError(f"Model file not found: {cfg.model.model_path}")


@hydra.main(version_base=None, config_path="../../configs/infer", config_name="inference")
def main(cfg: DictConfig) -> None:
    """メインエントリポイント"""
    # ログ設定
    logging.basicConfig(
        level=getattr(logging, cfg.system.log_level),
        format='[%(levelname)s] %(message)s'
    )
    logging.info("🎾 Tennis Ball Detection Inference System 開始")
    
    try:
        # 設定検証
        validate_config(cfg)
        
        # 出力ディレクトリ作成
        output_dir = os.path.dirname(cfg.io.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 設定取得
        pipeline_config = get_pipeline_config(cfg)
        detector_config = get_detector_config(cfg)
        vis_config = get_visualization_config(cfg)
        
        logging.info(f"設定: {OmegaConf.to_yaml(cfg)}")
        logging.info(f"入力動画: {cfg.io.video}")
        logging.info(f"出力動画: {cfg.io.output}")
        logging.info(f"モデル: {detector_config['model_type']} ({cfg.model.model_path})")
        logging.info(f"デバイス: {detector_config['device']}")
        logging.info(f"パイプライン設定: {cfg.pipeline.type}")
        
        # パイプライン作成
        pipeline = VideoPipeline(pipeline_config)
        
        # 処理実行
        start_time = time.time()
        
        if cfg.system.async_processing:
            # 非同期処理
            logging.info("非同期処理モードで実行中...")
            result = pipeline.process_video_async(
                video_path=cfg.io.video,
                detector_config=detector_config,
                output_path=cfg.io.output,
                visualization_config=vis_config
            )
            
            # 進捗表示
            display_progress(result, cfg.system.progress_interval)
            final_result = result.get_result()
            
        else:
            # 同期処理
            logging.info("同期処理モードで実行中...")
            final_result = pipeline.process_video(
                video_path=cfg.io.video,
                detector_config=detector_config,
                output_path=cfg.io.output,
                visualization_config=vis_config
            )
        
        processing_time = time.time() - start_time
        
        # 結果表示
        logging.info("🎯 処理完了!")
        logging.info(f"処理時間: {processing_time:.2f}秒")
        logging.info(f"平均FPS: {final_result.get('average_fps', 0):.2f}")
        logging.info(f"総フレーム数: {final_result.get('total_frames', 0)}")
        logging.info(f"検出フレーム数: {sum(1 for det_list in final_result.get('detections', {}).values() if det_list)}")
        
        # 統計情報保存
        if cfg.io.stats_output:
            stats = calculate_statistics(final_result)
            save_statistics(stats, cfg.io.stats_output)
        
        logging.info(f"出力動画: {cfg.io.output}")
        
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 