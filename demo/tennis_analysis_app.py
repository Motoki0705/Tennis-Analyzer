#!/usr/bin/env python3
"""
Tennis Analysis Gradio Application
=================================

🎾 テニス動画・画像総合解析システム

このアプリケーションは、テニス動画・画像から包括的な解析を行う
統合Gradioインターフェースです。最新のpredictor APIを活用し、
ボール検出、コート解析、プレーヤー追跡、イベント分析までを
ワンストップで提供します。

Features:
- 🎯 ボール検出・軌跡可視化
- 🏟️ コート認識・キーポイント検出
- 👥 プレーヤー検出・姿勢推定
- 📊 統計分析・レポート生成
- ⚡ リアルタイム・バッチ処理
- 🎨 高品質可視化・カスタマイズ

Author: Tennis Analysis System Team
License: MIT
Version: 1.0.0
"""

import gradio as gr
import torch
import numpy as np
import cv2
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.predictor import (
        VideoPipeline,
        create_ball_detector,
        VisualizationConfig,
        HIGH_PERFORMANCE_CONFIG,
        MEMORY_EFFICIENT_CONFIG,
        REALTIME_CONFIG,
        DEBUG_CONFIG
    )
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    PREDICTOR_ERROR = str(e)
    print(f"Warning: Predictor system not available: {e}")

# グローバル設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATHS = {
    "ball_lite_tracknet": "checkpoints/ball/lite_tracknet.ckpt",
    "ball_wasb_sbdt": "checkpoints/ball/wasb_sbdt.pth", 
    "court": "checkpoints/court/lite_tracknet.ckpt",
    "player": "checkpoints/player/rt_detr.ckpt"
}

# ====================================================================
# ユーティリティ関数
# ====================================================================

def get_available_models() -> Dict[str, List[str]]:
    """利用可能なモデルファイルを検索"""
    available = {"ball": [], "court": [], "player": []}
    
    for model_type, default_path in DEFAULT_MODEL_PATHS.items():
        if os.path.exists(default_path):
            category = model_type.split("_")[0]
            available[category].append(default_path)
    
    # checkpointsディレクトリを検索
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for category in available.keys():
            category_dir = checkpoints_dir / category
            if category_dir.exists():
                for ext in ["*.ckpt", "*.pth"]:
                    available[category].extend([str(p) for p in category_dir.glob(ext)])
    
    # 重複除去
    for category in available:
        available[category] = list(set(available[category]))
    
    return available

def create_pipeline_config(config_name: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """パイプライン設定作成"""
    base_configs = {
        "high_performance": HIGH_PERFORMANCE_CONFIG,
        "memory_efficient": MEMORY_EFFICIENT_CONFIG,
        "realtime": REALTIME_CONFIG,
        "debug": DEBUG_CONFIG
    }
    
    config = base_configs.get(config_name, MEMORY_EFFICIENT_CONFIG).copy()
    
    if custom_params:
        config.update(custom_params)
    
    return config

def create_visualization_config(**kwargs) -> VisualizationConfig:
    """可視化設定作成"""
    return VisualizationConfig(**kwargs)

def generate_statistics_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """統計レポート生成"""
    report = {
        "processing_info": {
            "total_frames": results.get("total_frames", 0),
            "processing_time": results.get("processing_time", 0.0),
            "average_fps": results.get("average_fps", 0.0)
        },
        "detection_stats": {},
        "analysis_summary": {}
    }
    
    # ボール検出統計
    detections = results.get("detections", {})
    if detections:
        total_detections = sum(len(det_list) for det_list in detections.values())
        frames_with_detection = sum(1 for det_list in detections.values() if det_list)
        
        report["detection_stats"] = {
            "total_detections": total_detections,
            "frames_with_detection": frames_with_detection,
            "detection_rate": frames_with_detection / len(detections) if detections else 0.0
        }
    
    return report

def create_trajectory_plot(detections: Dict[str, List]) -> go.Figure:
    """軌跡プロット作成"""
    fig = go.Figure()
    
    x_coords = []
    y_coords = []
    frame_nums = []
    
    for frame_id, det_list in detections.items():
        if det_list:
            # フレーム番号を抽出
            frame_num = int(frame_id.replace("frame_", ""))
            for det in det_list:
                if len(det) >= 2:
                    x_coords.append(det[0])
                    y_coords.append(det[1])
                    frame_nums.append(frame_num)
    
    if x_coords:
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            name='Ball Trajectory',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Tennis Ball Trajectory",
            xaxis_title="X Coordinate (normalized)",
            yaxis_title="Y Coordinate (normalized)",
            template="plotly_white"
        )
    
    return fig

# ====================================================================
# ボール検出タブ
# ====================================================================

def analyze_ball_detection(
    video_file: str,
    model_path: str,
    model_type: str,
    config_preset: str,
    ball_radius: int,
    trajectory_length: int,
    enable_smoothing: bool,
    enable_prediction: bool,
    confidence_threshold: float,
    progress=gr.Progress()
) -> Tuple[str, str, go.Figure]:
    """ボール検出解析実行"""
    if not PREDICTOR_AVAILABLE:
        raise gr.Error(f"Predictor system not available: {PREDICTOR_ERROR}")
    
    if not video_file:
        raise gr.Error("動画ファイルをアップロードしてください")
    
    if not os.path.exists(model_path):
        raise gr.Error(f"モデルファイルが見つかりません: {model_path}")
    
    progress(0.1, desc="初期化中...")
    
    try:
        # パイプライン設定
        pipeline_config = create_pipeline_config(config_preset)
        pipeline = VideoPipeline(pipeline_config)
        
        # 検出器設定
        detector_config = {
            "model_path": model_path,
            "model_type": model_type,
            "device": DEVICE
        }
        
        # 可視化設定
        vis_config = create_visualization_config(
            ball_radius=ball_radius,
            trajectory_length=trajectory_length,
            enable_smoothing=enable_smoothing,
            enable_prediction=enable_prediction,
            confidence_threshold=confidence_threshold
        )
        
        progress(0.2, desc="処理開始...")
        
        # 動画処理実行
        output_filename = f"ball_detection_{int(time.time())}.mp4"
        result = pipeline.process_video(
            video_path=video_file,
            detector_config=detector_config,
            output_path=output_filename,
            vis_config=vis_config
        )
        
        progress(0.9, desc="統計生成中...")
        
        # 統計レポート生成
        stats_report = generate_statistics_report(result)
        stats_json = json.dumps(stats_report, indent=2, ensure_ascii=False)
        
        # 軌跡プロット作成
        trajectory_plot = create_trajectory_plot(result.get("detections", {}))
        
        progress(1.0, desc="完了!")
        
        return output_filename, stats_json, trajectory_plot
        
    except Exception as e:
        raise gr.Error(f"処理中にエラーが発生しました: {str(e)}")

def create_ball_detection_tab() -> gr.Tab:
    """ボール検出タブ作成"""
    with gr.Tab("🎯 Ball Detection", id="ball_detection") as tab:
        gr.Markdown("""
        ## テニスボール検出・軌跡解析
        
        動画をアップロードして、AIによるボール検出と軌跡可視化を実行します。
        高精度な検出と滑らかな軌跡表示、統計分析を提供します。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Settings")
                
                video_input = gr.File(
                    label="動画ファイル",
                    file_types=["video"],
                    type="filepath"
                )
                
                available_models = get_available_models()
                model_path = gr.Dropdown(
                    label="モデルファイル",
                    choices=available_models["ball"] if available_models["ball"] else ["No models found"],
                    value=available_models["ball"][0] if available_models["ball"] else None
                )
                
                model_type = gr.Radio(
                    label="モデルタイプ",
                    choices=["auto", "lite_tracknet", "wasb_sbdt"],
                    value="auto"
                )
                
                gr.Markdown("### ⚙️ Processing Settings")
                
                config_preset = gr.Radio(
                    label="処理設定プリセット",
                    choices=["high_performance", "memory_efficient", "realtime", "debug"],
                    value="memory_efficient"
                )
                
                gr.Markdown("### 🎨 Visualization Settings")
                
                ball_radius = gr.Slider(
                    label="ボール描画半径",
                    minimum=4, maximum=20, value=8, step=1
                )
                
                trajectory_length = gr.Slider(
                    label="軌跡表示フレーム数",
                    minimum=5, maximum=50, value=20, step=1
                )
                
                enable_smoothing = gr.Checkbox(
                    label="位置スムージング有効化",
                    value=False
                )
                
                enable_prediction = gr.Checkbox(
                    label="位置予測表示有効化",
                    value=False
                )
                
                confidence_threshold = gr.Slider(
                    label="信頼度閾値",
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05
                )
                
                submit_btn = gr.Button(
                    "🚀 解析開始",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                with gr.Row():
                    output_video = gr.Video(
                        label="処理済み動画",
                        height=400
                    )
                
                with gr.Row():
                    with gr.Column():
                        stats_output = gr.JSON(
                            label="統計情報"
                        )
                    
                    with gr.Column():
                        trajectory_plot = gr.Plot(
                            label="軌跡プロット"
                        )
        
        # イベントハンドラ
        submit_btn.click(
            fn=analyze_ball_detection,
            inputs=[
                video_input, model_path, model_type, config_preset,
                ball_radius, trajectory_length, enable_smoothing,
                enable_prediction, confidence_threshold
            ],
            outputs=[output_video, stats_output, trajectory_plot]
        )
    
    return tab

# ====================================================================
# バッチ処理タブ
# ====================================================================

def run_batch_processing(
    input_files: List[str],
    model_path: str,
    model_type: str,
    config_preset: str,
    parallel_jobs: int,
    continue_on_error: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """バッチ処理実行"""
    if not PREDICTOR_AVAILABLE:
        raise gr.Error(f"Predictor system not available: {PREDICTOR_ERROR}")
    
    if not input_files:
        raise gr.Error("処理対象ファイルを選択してください")
    
    progress(0.1, desc="バッチ処理準備中...")
    
    try:
        # 一時ディレクトリ作成
        output_dir = f"batch_output_{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        # バッチ処理実行（簡単な実装）
        results = []
        total_files = len(input_files)
        
        for i, video_file in enumerate(input_files):
            progress((i + 1) / total_files, desc=f"処理中 ({i+1}/{total_files}): {os.path.basename(video_file)}")
            
            try:
                # 個別ファイル処理
                pipeline_config = create_pipeline_config(config_preset)
                pipeline = VideoPipeline(pipeline_config)
                
                detector_config = {
                    "model_path": model_path,
                    "model_type": model_type,
                    "device": DEVICE
                }
                
                output_filename = os.path.join(
                    output_dir, 
                    f"{Path(video_file).stem}_annotated.mp4"
                )
                
                result = pipeline.process_video(
                    video_path=video_file,
                    detector_config=detector_config,
                    output_path=output_filename
                )
                
                results.append({
                    "file": video_file,
                    "status": "success",
                    "output": output_filename,
                    "stats": result
                })
                
            except Exception as e:
                error_msg = str(e)
                results.append({
                    "file": video_file,
                    "status": "error",
                    "error": error_msg
                })
                
                if not continue_on_error:
                    break
        
        # レポート生成
        batch_report = {
            "summary": {
                "total_files": len(input_files),
                "processed_files": sum(1 for r in results if r["status"] == "success"),
                "failed_files": sum(1 for r in results if r["status"] == "error"),
            },
            "detailed_results": results
        }
        
        report_json = json.dumps(batch_report, indent=2, ensure_ascii=False)
        
        # ZIPファイル作成（簡単な実装）
        import shutil
        zip_filename = f"{output_dir}.zip"
        shutil.make_archive(output_dir, 'zip', output_dir)
        
        progress(1.0, desc="完了!")
        
        return zip_filename, report_json
        
    except Exception as e:
        raise gr.Error(f"バッチ処理中にエラーが発生しました: {str(e)}")

def create_batch_processing_tab() -> gr.Tab:
    """バッチ処理タブ作成"""
    with gr.Tab("📁 Batch Processing", id="batch_processing") as tab:
        gr.Markdown("""
        ## バッチ処理システム
        
        複数の動画ファイルを一括で処理します。
        並列処理による高速化とエラー回復機能を提供します。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input Settings")
                
                input_files = gr.File(
                    label="動画ファイル（複数選択可）",
                    file_count="multiple",
                    file_types=["video"],
                    type="filepath"
                )
                
                available_models = get_available_models()
                model_path = gr.Dropdown(
                    label="モデルファイル",
                    choices=available_models["ball"] if available_models["ball"] else ["No models found"],
                    value=available_models["ball"][0] if available_models["ball"] else None
                )
                
                model_type = gr.Radio(
                    label="モデルタイプ",
                    choices=["auto", "lite_tracknet", "wasb_sbdt"],
                    value="auto"
                )
                
                gr.Markdown("### ⚙️ Batch Settings")
                
                config_preset = gr.Radio(
                    label="処理設定プリセット",
                    choices=["high_performance", "memory_efficient", "realtime", "debug"],
                    value="memory_efficient"
                )
                
                parallel_jobs = gr.Slider(
                    label="並列ジョブ数",
                    minimum=1, maximum=8, value=2, step=1
                )
                
                continue_on_error = gr.Checkbox(
                    label="エラー時も処理継続",
                    value=True
                )
                
                batch_submit_btn = gr.Button(
                    "🚀 バッチ処理開始",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                output_zip = gr.File(
                    label="処理済みファイル（ZIP）"
                )
                
                batch_report = gr.JSON(
                    label="バッチ処理レポート"
                )
        
        # イベントハンドラ
        batch_submit_btn.click(
            fn=run_batch_processing,
            inputs=[
                input_files, model_path, model_type, config_preset,
                parallel_jobs, continue_on_error
            ],
            outputs=[output_zip, batch_report]
        )
    
    return tab

# ====================================================================
# 設定・情報タブ
# ====================================================================

def get_system_info() -> Dict[str, Any]:
    """システム情報取得"""
    return {
        "device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "predictor_available": PREDICTOR_AVAILABLE,
        "available_models": get_available_models()
    }

def create_settings_tab() -> gr.Tab:
    """設定・情報タブ作成"""
    with gr.Tab("⚙️ Settings & Info", id="settings") as tab:
        gr.Markdown("""
        ## システム設定・情報
        
        システム状態の確認とモデル管理を行います。
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💻 System Information")
                
                system_info = gr.JSON(
                    label="システム情報",
                    value=get_system_info()
                )
                
                refresh_btn = gr.Button("🔄 情報更新")
                
                gr.Markdown("### 📋 Available Models")
                
                models_info = gr.JSON(
                    label="利用可能モデル",
                    value=get_available_models()
                )
            
            with gr.Column():
                gr.Markdown("### 📚 Usage Guide")
                
                gr.Markdown("""
                #### 基本的な使用方法
                
                1. **Ball Detection**タブで単一動画解析
                2. **Batch Processing**タブで複数動画一括処理
                3. 各タブで適切なモデルファイルを選択
                4. 処理設定を調整（高性能 vs メモリ効率）
                5. 可視化設定をカスタマイズ
                
                #### 推奨設定
                
                - **高速処理**: high_performance preset
                - **メモリ節約**: memory_efficient preset  
                - **リアルタイム**: realtime preset
                - **デバッグ**: debug preset
                
                #### トラブルシューティング
                
                - GPU使用時はCUDA環境を確認
                - モデルファイルパスが正しいことを確認
                - メモリ不足時はバッチサイズを削減
                """)
        
        # イベントハンドラ
        refresh_btn.click(
            fn=lambda: (get_system_info(), get_available_models()),
            outputs=[system_info, models_info]
        )
    
    return tab

# ====================================================================
# メインアプリケーション
# ====================================================================

def create_main_app() -> gr.Blocks:
    """メインアプリケーション作成"""
    
    # カスタムCSS
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .tab-nav {
        border-bottom: 2px solid #e1e5e9;
    }
    .tennis-header {
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    """
    
    with gr.Blocks(
        title="Tennis Analysis System",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as app:
        
        # ヘッダー
        gr.Markdown("""
        <h1 class="tennis-header">🎾 Tennis Analysis System</h1>
        <p style="text-align: center; font-size: 1.2rem; color: #666;">
        AI-Powered Tennis Video Analysis Platform
        </p>
        """)
        
        # システム状態確認
        if not PREDICTOR_AVAILABLE:
            gr.Markdown(f"""
            <div style="background-color: #ffe6e6; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff4444;">
            <strong>⚠️ Warning:</strong> Predictor system is not available.<br>
            <strong>Error:</strong> {PREDICTOR_ERROR}<br>
            Some features may be limited.
            </div>
            """)
        
        # タブ作成
        with gr.Tabs():
            ball_tab = create_ball_detection_tab()
            batch_tab = create_batch_processing_tab()
            settings_tab = create_settings_tab()
        
        # フッター
        gr.Markdown("""
        ---
        <p style="text-align: center; color: #888; font-size: 0.9rem;">
        Tennis Analysis System v1.0.0 | Powered by PyTorch, Gradio, and WASB-SBDT
        </p>
        """)
    
    return app

# ====================================================================
# アプリケーション起動
# ====================================================================

def main():
    """メインエントリポイント"""
    print("🎾 Tennis Analysis System Starting...")
    print(f"Device: {DEVICE}")
    print(f"Predictor Available: {PREDICTOR_AVAILABLE}")
    
    app = create_main_app()
    
    # アプリケーション起動
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main() 