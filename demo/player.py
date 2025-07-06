import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import time
import os

# ------------------------------------------------------------------------
# グローバル設定: デバイス、モデル、プロセッサの準備
# ------------------------------------------------------------------------

# ユーザー指定のインポート文を利用
try:
    from src.player.lit_module.lit_rtdetr import LitRtdetr
    from transformers import RTDetrImageProcessor
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(e)

device = "cuda" if torch.cuda.is_available() else "cpu"
# --- ユーザーが指定したチェックポイントパス ---
CHECKPOINT_PATH = "checkpoints/player/lit_rt_detr/epoch=002-val_loss_total=2.38540339.ckpt"
# --- 検出結果として描画する信頼度の閾値 ---
CONFIDENCE_THRESHOLD = 0.5 

model = None
processor = None
MODEL_LOAD_ERROR = ""

# モデルとプロセッサのロード
if IMPORT_SUCCESS:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            # PyTorch Lightningモデルをロードし、評価モードに設定
            lit_model = LitRtdetr.load_from_checkpoint(CHECKPOINT_PATH, map_location=device)
            model = lit_model.to(device).eval()
            
            # 対応するRT-DETRのImage Processorをロード
            processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
            print(f"Model loaded successfully from '{CHECKPOINT_PATH}' and moved to {device}.")
        except Exception as e:
            MODEL_LOAD_ERROR = f"Failed to load model or processor: {e}"
            print(f"Error: {MODEL_LOAD_ERROR}")
    else:
        MODEL_LOAD_ERROR = f"Checkpoint file not found at '{CHECKPOINT_PATH}'"
        print(f"Error: {MODEL_LOAD_ERROR}")
else:
    MODEL_LOAD_ERROR = f"Could not import custom modules: {IMPORT_ERROR_MESSAGE}"
    print(f"Error: {MODEL_LOAD_ERROR}")

# ------------------------------------------------------------------------
# 動画解析と描画の実行関数
# ------------------------------------------------------------------------

def detect_players_in_video(video_path: str, progress=gr.Progress(track_tqdm=True)):
    """
    動画をフレームごとに解析し、RT-DETRでプレイヤーを検出し、
    バウンディングボックスを描画した新しい動画を生成する。
    """
    if model is None or processor is None:
        raise gr.Error(f"モデルまたはプロセッサがロードされていません: {MODEL_LOAD_ERROR}")
    if video_path is None:
        raise gr.Error("動画ファイルをアップロードしてください。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("動画ファイルを開けませんでした。")

    # 動画のプロパティ取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力動画の設定
    timestamp = int(time.time())
    output_filename = f"output_player_detection_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (orig_width, orig_height))

    # --- フレームごとの推論と描画 ---
    for _ in progress.tqdm(range(total_frames), desc="プレイヤーを検出中..."):
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCVのフレーム(BGR)をPIL Image(RGB)に変換
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        target_size = torch.tensor([(image.height, image.width)]).to(device)

        # 画像の前処理
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 推論の実行
        with torch.no_grad():
            outputs = model(**inputs)

        # 推論結果の後処理
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=CONFIDENCE_THRESHOLD
        )[0] # バッチの最初の結果を取得

        # 検出結果をフレームに描画
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i) for i in box.tolist()]
            
            # ラベル名を取得 (model.config.id2labelが存在することを仮定)
            try:
                label = model.config.id2label[label_id.item()]
            except (AttributeError, KeyError):
                # .config.id2label がない場合は、IDをそのまま表示
                label = f"ID:{label_id.item()}"

            # バウンディングボックスを描画
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (36, 255, 12), 2)
            
            # ラベルとスコアを描画
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

        # 処理済みフレームを動画ファイルに書き込み
        out_writer.write(frame)

    cap.release()
    out_writer.release()

    return output_filename

# ------------------------------------------------------------------------
# Gradioインターフェースの構築
# ------------------------------------------------------------------------
title = "Player Detection Demo with Fine-tuned RT-DETR"
description = """
**動画をアップロードすると、AI（ファインチューニング済みRT-DETR）が動画内のプレイヤーを検出し、その結果を描画した新しい動画を生成します。**

**使い方:**
1. `Upload Video` ボックスに動画ファイル（例: MP4）をドラッグ＆ドロップするか、クリックしてファイルを選択します。
2. `Submit` ボタンを押して解析を開始します。
3. 処理の進捗がプログレスバーに表示されます。
4. 解析が完了すると、右側に結果の動画が表示されます。
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            video_output = gr.Video(label="Result Video")

    submit_button.click(
        fn=detect_players_in_video,
        inputs=[video_input],
        outputs=[video_output]
    )
    
    gr.Examples(
        examples=[], # 例: [["path/to/your/sample_video.mp4"]]
        inputs=video_input,
        outputs=video_output,
        fn=detect_players_in_video,
        cache_examples=False 
    )

if __name__ == "__main__":
    # Gradioアプリを起動
    demo.launch(share=True)