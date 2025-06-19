import gradio as gr
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from collections import deque
import time
import os

# ユーザー指定のインポート文を利用
try:
    from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(e)

# ------------------------------------------------------------------------
# グローバル設定: デバイス、モデル、Transform、および新パラメータの準備
# ------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
INPUT_SIZE = (360, 640)

# ===== 新機能のパラメータ =====
# 異常値とみなすピクセル距離の閾値。この値より大きく動いたらジャンプと判断。
# 動画の解像度やボールの速度に応じて調整してください。
DISTANCE_THRESHOLD = 150  

model = None
MODEL_LOAD_ERROR = ""

# モデルのロード
if IMPORT_SUCCESS:
    if os.path.exists(CHECKPOINT_PATH):
        lit_model = LitLiteTracknetFocalLoss.load_from_checkpoint(CHECKPOINT_PATH, map_location=device)
        model = lit_model.model
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from '{CHECKPOINT_PATH}'")
    else:
        MODEL_LOAD_ERROR = f"Checkpoint file not found at '{CHECKPOINT_PATH}'"
        print(f"Error: {MODEL_LOAD_ERROR}")
else:
    MODEL_LOAD_ERROR = f"Could not import LitLiteTracknetFocalLoss: {IMPORT_ERROR_MESSAGE}"
    print(f"Error: {MODEL_LOAD_ERROR}")

# Transformの定義
transform = A.ReplayCompose(
    [
        A.Resize(height=INPUT_SIZE[0], width=INPUT_SIZE[1]),
        A.Normalize(),
        A.pytorch.ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format="xy"),
)

# ------------------------------------------------------------------------
# 動画解析と描画の実行関数 (修正箇所)
# ------------------------------------------------------------------------

def analyze_video(video_path: str, progress=gr.Progress(track_tqdm=True)) -> str:
    """
    動画を解析し、異常値除去を行った上でボールの軌跡を描画した新しい動画を生成する。
    """
    if model is None:
        raise gr.Error(f"モデルがロードされていません: {MODEL_LOAD_ERROR}")
    if video_path is None:
        raise gr.Error("動画ファイルをアップロードしてください。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("動画ファイルを開けませんでした。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- ステージ1/3: 全フレームのボール座標を推論 ---
    if progress:
        progress(0, desc="Stage 1/3: Analyzing frames...")
    
    frame_queue = deque(maxlen=3)
    ball_positions_raw = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_queue.append(frame_rgb)

        if len(frame_queue) == 3:
            frames_np = list(frame_queue)
            frames_transformed = []
            replay_data = transform(image=frames_np[0])
            frames_transformed.append(replay_data["image"])
            for f_np in frames_np[1:]:
                replayed = A.ReplayCompose.replay(replay_data["replay"], image=f_np)
                frames_transformed.append(replayed["image"])

            input_tensor = torch.cat(frames_transformed, dim=0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                heatmap_pred = model(input_tensor)
                heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()

            h_resized, w_resized = heatmap_prob.shape
            pos = np.unravel_index(np.argmax(heatmap_prob), heatmap_prob.shape)
            y_resized, x_resized = pos
            x_orig = int(x_resized * orig_width / w_resized)
            y_orig = int(y_resized * orig_height / h_resized)
            ball_positions_raw.append((x_orig, y_orig))
        
        if progress:
            progress((frame_idx + 1) / total_frames, desc=f"Stage 1/3: Analyzing frame {frame_idx + 1}/{total_frames}")

    cap.release()
    
    # --- ステージ2/3: 座標の後処理 (異常値除去) ---
    if progress:
        progress(0, desc="Stage 2/3: Post-processing coordinates...")

    # 最初の2フレームは推論できないためNoneでパディング
    full_positions = [None, None] + ball_positions_raw
    
    for i in range(2, len(full_positions)):
        # 前のフレームと現在のフレームの座標を取得
        prev_pos = full_positions[i-1]
        current_pos = full_positions[i]
        
        # 両方の座標が存在する場合のみ距離を計算
        if prev_pos is not None and current_pos is not None:
            # ユークリッド距離を計算
            dist = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
            
            # 距離が閾値を超えたらジャンプとみなし、前のフレームの座標で上書き(リジェクト)
            if dist > DISTANCE_THRESHOLD:
                full_positions[i] = prev_pos
        
        if progress:
             progress((i + 1) / len(full_positions), desc=f"Stage 2/3: Filtering frame {i + 1}/{len(full_positions)}")


    # --- ステージ3/3: 元の動画に座標と軌跡を描画 ---
    if progress:
        progress(0, desc="Stage 3/3: Rendering output video...")

    timestamp = int(time.time())
    output_filename = f"output_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (orig_width, orig_height))
    cap = cv2.VideoCapture(video_path)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        current_pos = full_positions[frame_idx] if frame_idx < len(full_positions) else None
        if current_pos:
            cv2.circle(frame, center=current_pos, radius=8, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        
        start_idx = max(2, frame_idx - 15)
        for i in range(start_idx, frame_idx):
            if i < len(full_positions) and (i + 1) < len(full_positions):
                p1, p2 = full_positions[i], full_positions[i+1]
                if p1 and p2:
                    cv2.line(frame, p1, p2, (0, 255, 255), 3, lineType=cv2.LINE_AA)

        out_writer.write(frame)
        
        if progress:
            progress((frame_idx + 1) / total_frames, desc=f"Stage 3/3: Rendering frame {frame_idx + 1}/{total_frames}")
            
    cap.release()
    out_writer.release()
    return output_filename

# ------------------------------------------------------------------------
# Gradioインターフェースの構築 (変更なし)
# ------------------------------------------------------------------------
title = "Video Ball Tracking System with Outlier Rejection"
description = """
**動画をアップロードすると、AIがボールの位置を追跡し、その軌跡を描画した新しい動画を生成します。**

**新機能:** 検出されたボールの座標が不自然にジャンプした場合、その点を異常値として棄却し、滑らかな軌跡を維持する機能が追加されました。

**使い方:**
1. `Upload Video` ボックスに動画ファイル（例: MP4）をドラッグ＆ドロップするか、クリックしてファイルを選択します。
2. `Submit` ボタンを押して解析を開始します。
3. 処理の進捗がプログレスバーに表示されます。処理は3つのステージ（推論→後処理→描画）で実行されます。
4. 解析が完了すると、下に結果の動画が表示されます。
"""

with gr.Blocks() as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Result Video")

    submit_button.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[video_output]
    )
    
    gr.Examples(
        examples=[],
        inputs=video_input,
        outputs=video_output,
        fn=analyze_video,
        cache_examples=True
    )

if __name__ == "__main__":
    demo.launch(share=True)