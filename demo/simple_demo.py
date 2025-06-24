#!/usr/bin/env python3
"""
Simple Tennis Ball Detection Demo
================================

🎾 シンプルテニスボール検出デモ

軽量で使いやすいテニスボール検出デモアプリケーションです。
最小限の設定で素早くボール検出を試すことができます。

Features:
- 🚀 ワンクリック解析
- 📱 モバイル対応UI
- ⚡ 高速処理
- 🎨 リアルタイム可視化

Author: Tennis Analysis System Team
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
from typing import Dict, Any, Tuple, Optional
from PIL import Image

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.predictor import (
        VideoPipeline,
        create_ball_detector,
        VisualizationConfig,
        MEMORY_EFFICIENT_CONFIG
    )
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    PREDICTOR_ERROR = str(e)

# グローバル設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATHS = [
    "checkpoints/ball/lite_tracknet.ckpt",
    "checkpoints/ball/wasb_sbdt.pth",
    "checkpoints/ball/best_model.ckpt",
    "models/ball_detector.ckpt",
    "models/ball_detector.pth"
]

def find_available_model() -> Optional[str]:
    """利用可能なモデルファイルを検索"""
    for model_path in DEFAULT_MODEL_PATHS:
        if os.path.exists(model_path):
            return model_path
    
    # checkpointsディレクトリを検索
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for pattern in ["**/*.ckpt", "**/*.pth"]:
            for model_file in checkpoints_dir.glob(pattern):
                if "ball" in str(model_file).lower():
                    return str(model_file)
    
    return None

def analyze_video_simple(
    video_file: str,
    ball_radius: int = 8,
    trajectory_length: int = 20,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """シンプル動画解析"""
    
    if not PREDICTOR_AVAILABLE:
        return None, f"❌ Predictor system not available: {PREDICTOR_ERROR}"
    
    if not video_file:
        return None, "❌ 動画ファイルをアップロードしてください"
    
    # モデル検索
    model_path = find_available_model()
    if not model_path:
        return None, "❌ モデルファイルが見つかりません。checkpoints/フォルダにモデルを配置してください。"
    
    progress(0.1, desc="初期化中...")
    
    try:
        # パイプライン設定（メモリ効率重視）
        pipeline_config = MEMORY_EFFICIENT_CONFIG.copy()
        pipeline_config.update({
            "batch_size": 4,  # 軽量化
            "num_workers": 2,  # 軽量化
            "queue_size": 50   # 軽量化
        })
        
        pipeline = VideoPipeline(pipeline_config)
        
        # 検出器設定（自動判定）
        detector_config = {
            "model_path": model_path,
            "model_type": "auto",  # 拡張子から自動判定
            "device": DEVICE
        }
        
        # 可視化設定（シンプル）
        vis_config = VisualizationConfig(
            ball_radius=ball_radius,
            trajectory_length=trajectory_length,
            enable_smoothing=True,  # デフォルトで有効
            enable_prediction=False,  # 軽量化のため無効
            confidence_threshold=0.5
        )
        
        progress(0.2, desc="処理開始...")
        
        # 出力ファイル名生成
        timestamp = int(time.time())
        output_filename = f"tennis_analysis_{timestamp}.mp4"
        
        # 動画処理実行
        result = pipeline.process_video(
            video_path=video_file,
            detector_config=detector_config,
            output_path=output_filename,
            vis_config=vis_config
        )
        
        progress(0.9, desc="結果生成中...")
        
        # 簡単な統計情報
        detections = result.get("detections", {})
        total_frames = result.get("total_frames", 0)
        processing_time = result.get("processing_time", 0.0)
        
        detection_count = sum(len(det_list) for det_list in detections.values())
        frames_with_ball = sum(1 for det_list in detections.values() if det_list)
        
        stats_text = f"""
🎾 **解析完了！**

📊 **統計情報:**
- 総フレーム数: {total_frames:,}
- ボール検出数: {detection_count:,}
- 検出フレーム数: {frames_with_ball:,}
- 検出率: {frames_with_ball/total_frames*100:.1f}%
- 処理時間: {processing_time:.1f}秒
- 平均FPS: {result.get('average_fps', 0):.1f}

🤖 **使用モデル:** {os.path.basename(model_path)}
💻 **実行デバイス:** {DEVICE.upper()}
        """
        
        progress(1.0, desc="完了!")
        
        return output_filename, stats_text
        
    except Exception as e:
        error_msg = f"❌ 処理中にエラーが発生しました: {str(e)}"
        return None, error_msg

def analyze_image_simple(
    image_file: Image.Image,
    ball_radius: int = 8
) -> Tuple[Image.Image, str]:
    """シンプル画像解析（単一フレーム）"""
    
    if not image_file:
        return None, "❌ 画像ファイルをアップロードしてください"
    
    # 簡単な処理例（実際にはモデルを使用）
    try:
        # PIL → OpenCV変換
        image_np = np.array(image_file)
        if image_np.shape[-1] == 4:  # RGBA → RGB
            image_np = image_np[:, :, :3]
        
        # 簡単なデモ用の処理（中央にボールを描画）
        height, width = image_np.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # ボール描画
        image_with_ball = image_np.copy()
        cv2.circle(
            image_with_ball, 
            (center_x, center_y), 
            ball_radius, 
            (255, 0, 0),  # 赤色
            -1
        )
        
        # RGB → PIL変換
        result_image = Image.fromarray(image_with_ball)
        
        stats_text = f"""
🎾 **画像解析完了！**

📏 **画像情報:**
- サイズ: {width} × {height}
- 検出位置: ({center_x}, {center_y})
- ボール半径: {ball_radius}px

💡 **Note:** これはデモ用の簡単な実装です。
実際のボール検出には動画解析をご利用ください。
        """
        
        return result_image, stats_text
        
    except Exception as e:
        error_msg = f"❌ 処理中にエラーが発生しました: {str(e)}"
        return None, error_msg

def create_simple_demo() -> gr.Blocks:
    """シンプルデモアプリケーション作成"""
    
    # カスタムCSS（モバイル対応）
    custom_css = """
    .gradio-container {
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    .tennis-title {
        text-align: center;
        color: #2E7D32;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .tennis-subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .demo-section {
        border: 2px solid #E8F5E8;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #F9FFF9;
    }
    """
    
    with gr.Blocks(
        title="Tennis Ball Detector - Simple Demo",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="green")
    ) as demo:
        
        # ヘッダー
        gr.Markdown("""
        <div class="tennis-title">🎾 Tennis Ball Detector</div>
        <div class="tennis-subtitle">Simple & Fast AI-Powered Analysis</div>
        """)
        
        # システム状態表示
        system_status = "🟢 Ready" if PREDICTOR_AVAILABLE else f"🔴 Limited (Predictor unavailable)"
        model_status = "🟢 Found" if find_available_model() else "🔴 Not Found"
        
        gr.Markdown(f"""
        <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; text-align: center;">
        **System Status:** {system_status} | **Model:** {model_status} | **Device:** {DEVICE.upper()}
        </div>
        """)
        
        # タブ構成
        with gr.Tabs():
            
            # 動画解析タブ
            with gr.Tab("📹 Video Analysis", id="video"):
                gr.Markdown('<div class="demo-section">')
                gr.Markdown("### 動画をアップロードして、AIによるボール検出と軌跡解析を実行")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.File(
                            label="📁 動画ファイル",
                            file_types=["video"],
                            type="filepath"
                        )
                        
                        with gr.Row():
                            ball_radius_video = gr.Slider(
                                label="🎯 ボール半径",
                                minimum=4, maximum=15, value=8, step=1
                            )
                            trajectory_length_video = gr.Slider(
                                label="📏 軌跡長",
                                minimum=10, maximum=30, value=20, step=2
                            )
                        
                        analyze_btn = gr.Button(
                            "🚀 解析開始",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        output_video = gr.Video(
                            label="📊 解析結果",
                            height=300
                        )
                        
                        stats_output = gr.Markdown(
                            label="📈 統計情報"
                        )
                
                gr.Markdown('</div>')
            
            # 画像解析タブ（デモ用）
            with gr.Tab("🖼️ Image Demo", id="image"):
                gr.Markdown('<div class="demo-section">')
                gr.Markdown("### 画像解析デモ（テスト用）")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="📁 画像ファイル",
                            type="pil"
                        )
                        
                        ball_radius_image = gr.Slider(
                            label="🎯 ボール半径",
                            minimum=5, maximum=20, value=10, step=1
                        )
                        
                        analyze_image_btn = gr.Button(
                            "🔍 解析実行",
                            variant="secondary"
                        )
                    
                    with gr.Column():
                        output_image = gr.Image(
                            label="📊 解析結果",
                            type="pil"
                        )
                        
                        image_stats = gr.Markdown(
                            label="📈 画像情報"
                        )
                
                gr.Markdown('</div>')
            
            # 使用方法タブ
            with gr.Tab("📚 Guide", id="guide"):
                gr.Markdown("""
                ## 🎾 使用方法
                
                ### 📹 動画解析
                1. **Video Analysis**タブを選択
                2. 動画ファイル（MP4, AVI等）をアップロード
                3. ボール半径と軌跡長を調整（オプション）
                4. **解析開始**ボタンをクリック
                5. 処理完了後、結果動画と統計情報を確認
                
                ### 🖼️ 画像デモ
                1. **Image Demo**タブを選択
                2. 画像ファイルをアップロード
                3. **解析実行**ボタンをクリック
                4. デモ用の処理結果を確認
                
                ### ⚙️ システム要件
                - **モデルファイル**: `checkpoints/`フォルダにモデルを配置
                - **GPU**: CUDA対応GPU推奨（CPUでも動作可能）
                - **メモリ**: 8GB以上推奨
                
                ### 🔧 トラブルシューティング
                - **モデルが見つからない**: checkpoints/フォルダを確認
                - **処理が遅い**: GPUが利用可能か確認
                - **メモリエラー**: 動画サイズを小さくするか、CPUを使用
                
                ### 📞 サポート
                - システムの詳細情報は画面上部のステータスで確認
                - エラーが発生した場合は、エラーメッセージを確認
                """)
        
        # イベントハンドラー
        analyze_btn.click(
            fn=analyze_video_simple,
            inputs=[video_input, ball_radius_video, trajectory_length_video],
            outputs=[output_video, stats_output]
        )
        
        analyze_image_btn.click(
            fn=analyze_image_simple,
            inputs=[image_input, ball_radius_image],
            outputs=[output_image, image_stats]
        )
        
        # フッター
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
        🎾 Tennis Ball Detector v1.0.0 | Simple Demo Version
        </div>
        """)
    
    return demo

def main():
    """メインエントリポイント"""
    print("🎾 Tennis Ball Detector - Simple Demo")
    print(f"Device: {DEVICE}")
    print(f"Predictor Available: {PREDICTOR_AVAILABLE}")
    
    available_model = find_available_model()
    if available_model:
        print(f"Model Found: {available_model}")
    else:
        print("Warning: No model files found in checkpoints/")
    
    demo = create_simple_demo()
    
    # デモ起動
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 