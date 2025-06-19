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

# (analyze_court_image関数とGradio UIのコードは変更ないため、省略します。前の回答と同じものを使用してください)
# ------------------------------------------------------------------------
# 画像解析と描画の実行関数
# ------------------------------------------------------------------------
def find_peaks(heatmap: np.ndarray, num_peaks: int, suppression_radius: int) -> list:
    peaks = []
    heatmap_copy = heatmap.copy()
    for _ in range(num_peaks):
        pos = np.unravel_index(np.argmax(heatmap_copy), heatmap_copy.shape)
        peaks.append(pos)
        cv2.circle(heatmap_copy, (pos[1], pos[0]), suppression_radius, 0, -1)
    return peaks

def analyze_court_image(input_image: Image.Image) -> Image.Image:
    if model is None:
        raise gr.Error("モデルがロードされていません。コンソールのエラーを確認してください。")
    if input_image is None:
        raise gr.Error("画像をアップロードしてください。")

    original_image_np = np.array(input_image)
    output_image = original_image_np.copy()
    
    transformed = transform(image=original_image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap_pred = model(input_tensor)
        heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()

    detected_peaks_resized = find_peaks(heatmap_prob, NUM_KEYPOINTS, PEAK_SUPPRESSION_RADIUS)

    orig_h, orig_w, _ = original_image_np.shape
    resized_h, resized_w = heatmap_prob.shape

    for i, peak in enumerate(detected_peaks_resized):
        y_resized, x_resized = peak
        x_orig = int(x_resized * orig_w / resized_w)
        y_orig = int(y_resized * orig_h / resized_h)
        cv2.circle(output_image, (x_orig, y_orig), radius=10, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(output_image, str(i+1), (x_orig + 15, y_orig + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    return Image.fromarray(output_image)

# ------------------------------------------------------------------------
# Gradioインターフェースの構築
# ------------------------------------------------------------------------
title = "Tennis Court Keypoint Detection"
description = """
**テニスコートが写った画像をアップロードすると、AIがコートのライン上の主要なキーポイントを15箇所検出します。**
"""
with gr.Blocks() as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Result Image with Keypoints")
    submit_button.click(
        fn=analyze_court_image,
        inputs=[image_input],
        outputs=[image_output]
    )
    gr.Examples(
        examples=[],
        inputs=image_input,
        outputs=image_output,
        fn=analyze_court_image,
        cache_examples=True
    )
if __name__ == "__main__":
    demo.launch(share=True)