import gradio as gr
import torch
import torch.nn as nn # nnをインポート
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
from collections import OrderedDict

from src.court.models.lite_tracknet import LiteTrackNet
from src.court.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocal

# ------------------------------------------------------------------------
# グローバル設定と新しいモデルロード方式
# ------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/court/lit_lite_tracknet/epoch=002-val_loss=nan.ckpt"
INPUT_SIZE = (360, 640)
NUM_KEYPOINTS = 15
PEAK_SUPPRESSION_RADIUS = 10
model = None

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
    print("--- Top-level keys in the checkpoint ---")
    print(checkpoint.keys())
    print("-" * 40)
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")

try:
    # --- 新しいモデルロード方式 ---
    # 1. モデルの「骨格」を先に作成
    # コートモデルの入力チャンネルは3 (RGB)
    model = LitLiteTracknetFocal.load_from_checkpoint(CHECKPOINT_PATH, map_location=device).model    
    model.to(device)
    model.eval() # 評価モードに設定
    print("Model weights loaded successfully by extracting from checkpoint.")

except FileNotFoundError:
    print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")


# Transformの定義 (変更なし)
transform = A.Compose( # 単一画像なのでComposeで十分
    [
        A.Resize(height=INPUT_SIZE[0], width=INPUT_SIZE[1]),
        A.Normalize(),
        A.pytorch.ToTensorV2(),
    ]
)

# (analyze_court_image関数とGradio UIのコードは変更ないため、省略します。前の回答と同じものを使用してください)
# ------------------------------------------------------------------------
# 画像解析と描画の実行関数 (★ここを修正)
# ------------------------------------------------------------------------
def analyze_image_and_show_heatmap(input_image: Image.Image) -> Image.Image:
    """
    単一の画像からヒートマップを生成し、カラーマップを適用して可視化する。
    """
    if model is None:
        raise gr.Error("モデルがロードされていません。コンソールのエラーを確認してください。")
    if input_image is None:
        raise gr.Error("画像をアップロードしてください。")

    # --- 1. 前処理 ---
    original_image_np = np.array(input_image)
    transformed = transform(image=original_image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # --- 2. 推論 ---
    with torch.no_grad():
        heatmap_pred = model(input_tensor)
        heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()

    # --- 3. ヒートマップの可視化 ---
    # ヒートマップを0-255の範囲に正規化
    heatmap_normalized = cv2.normalize(heatmap_prob, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # カラーマップ(JET)を適用して、グレースケールのヒートマップを色付け
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # OpenCVのBGR形式から、PILで扱えるRGB形式に変換
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return Image.fromarray(heatmap_rgb)

# ------------------------------------------------------------------------
# Gradioインターフェースの構築 (★ここを修正)
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
        fn=analyze_image_and_show_heatmap,
        inputs=[image_input],
        outputs=[image_output]
    )
    
    gr.Examples(
        examples=[],
        inputs=image_input,
        outputs=image_output,
        fn=analyze_image_and_show_heatmap,
        cache_examples=True
    )

if __name__ == "__main__":
    demo.launch(share=True)