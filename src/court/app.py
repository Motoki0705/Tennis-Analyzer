# app.py (プロジェクトのルートに配置)

import gradio as gr
from PIL import Image
import numpy as np
import cv2
from typing import Tuple

# 作成したパイプラインをインポート
from src.court.pipeline import CourtKeypointDetectorPipeline

# ------------------------------------------------------------------------
# グローバル設定とパイプラインの初期化
# ------------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/court/lit_lite_tracknet/epoch=002-val_loss=nan.ckpt"
pipeline = None

try:
    # パイプラインを一度だけ初期化
    pipeline = CourtKeypointDetectorPipeline(CHECKPOINT_PATH)
    print("Court detection pipeline initialized successfully.")
except Exception as e:
    print(f"Failed to initialize pipeline: {e}")
    # Gradioアプリを起動する前にエラーを表示したい場合
    # raise SystemExit(f"Error: {e}") from e

# ------------------------------------------------------------------------
# 画像解析と描画の実行関数
# ------------------------------------------------------------------------
def analyze_and_draw_keypoints(input_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    画像からキーポイントを検出し、元の画像に描画する。
    ヒートマップも同時に返す。
    """
    if pipeline is None:
        raise gr.Error("パイプラインが初期化されていません。コンソールのエラーを確認してください。")
    if input_image is None:
        raise gr.Error("画像をアップロードしてください。")

    # パイプラインを実行して結果を取得
    results = pipeline(input_image)

    keypoints = results['keypoints']
    scores = results['scores']
    heatmap_image = results['heatmap_image']

    # 元の画像にキーポイントを描画
    output_image_np = np.array(input_image.convert("RGB"))
    for i, (x, y) in enumerate(keypoints):
        # 座標を整数に変換
        px, py = int(round(x)), int(round(y))
        score = scores[i]
        
        # スコアに基づいて色を決定 (例: 高いほど緑、低いほど赤)
        color = (0, 255, 0) if score > 0.5 else (255, 0, 0)
        
        cv2.circle(output_image_np, (px, py), radius=5, color=color, thickness=-1)
        cv2.putText(output_image_np, f"{i}", (px + 7, py + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_image_pil = Image.fromarray(output_image_np)

    return output_image_pil, heatmap_image

# ------------------------------------------------------------------------
# Gradioインターフェースの構築
# ------------------------------------------------------------------------
title = "Tennis Court Heatmap Visualizer"
description = """
**テニスコートが写った画像をアップロードすると、AIがコートのキーポイントが存在する確率を示したヒートマップを生成します。**

赤に近いほど確率が高く、青に近いほど確率が低いことを示します。
"""
with gr.Blocks() as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column():
            # 出力コンポーネントのラベルを修正
            image_output = gr.Image(label="Output Heatmap")
    
    # 呼び出す関数名を修正
    submit_button.click(
        fn=analyze_and_draw_keypoints,
        inputs=[image_input],
        outputs=[image_output]
    )
    
    gr.Examples(
        examples=[],
        inputs=image_input,
        outputs=image_output,
        fn=analyze_and_draw_keypoints,
        cache_examples=True
    )

if __name__ == "__main__":
    demo.launch(share=True)