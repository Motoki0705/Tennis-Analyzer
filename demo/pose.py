import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import time
import os
import requests

# ------------------------------------------------------------------------
# グローバル設定: デバイス、モデル、プロセッサの準備
# ------------------------------------------------------------------------

# ユーザー指定のインポート文を利用
try:
    from src.player.lit_module.lit_rtdetr import LitRtdetr
    from transformers import AutoProcessor, RTDetrImageProcessor, VitPoseForPoseEstimation
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(e)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Stage 1: Player Detector ---
PLAYER_DETECTOR_CHECKPOINT = "checkpoints/player/lit_rt_detr/epoch=002-val_loss_total=2.38540339.ckpt"
PLAYER_CONFIDENCE_THRESHOLD = 0.5
player_detector_model = None
player_detector_processor = None
PLAYER_MODEL_LOAD_ERROR = ""

# --- Stage 2: Pose Estimator ---
POSE_KEYPOINT_THRESHOLD = 0.3
pose_estimator_model = None
pose_estimator_processor = None
POSE_MODEL_LOAD_ERROR = ""

# モデルとプロセッサのロード
if IMPORT_SUCCESS:
    # Load Player Detector
    if os.path.exists(PLAYER_DETECTOR_CHECKPOINT):
        try:
            lit_model = LitRtdetr.load_from_checkpoint(PLAYER_DETECTOR_CHECKPOINT, map_location=device)
            player_detector_model = lit_model.to(device).eval()
            player_detector_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
            print(f"Player detector loaded successfully from '{PLAYER_DETECTOR_CHECKPOINT}'.")
        except Exception as e:
            PLAYER_MODEL_LOAD_ERROR = f"Failed to load player detector: {e}"
            print(f"Error: {PLAYER_MODEL_LOAD_ERROR}")
    else:
        PLAYER_MODEL_LOAD_ERROR = f"Player detector checkpoint not found at '{PLAYER_DETECTOR_CHECKPOINT}'"
        print(f"Error: {PLAYER_MODEL_LOAD_ERROR}")

    # Load Pose Estimator
    try:
        pose_estimator_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        pose_estimator_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device).eval()
        print("Pose estimator 'vitpose-base-simple' loaded successfully.")
    except Exception as e:
        POSE_MODEL_LOAD_ERROR = f"Failed to load pose estimator: {e}"
        print(f"Error: {POSE_MODEL_LOAD_ERROR}")

else:
    PLAYER_MODEL_LOAD_ERROR = f"Could not import custom modules: {IMPORT_ERROR_MESSAGE}"
    print(f"Error: {PLAYER_MODEL_LOAD_ERROR}")


# ------------------------------------------------------------------------
# 描画ヘルパー関数
# ------------------------------------------------------------------------

# ViT-Pose (COCO) の骨格定義
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [0, 5], [0, 6]
]
# キーポイントと骨格の色を定義
KEYPOINT_COLOR = (0, 255, 0)  # Green
SKELETON_COLOR = (255, 128, 0) # Orange

def draw_poses(frame, pose_results):
    """フレーム上に検出されたキーポイントと骨格を描画する"""
    for person_pose in pose_results:
        keypoints = person_pose['keypoints']
        keypoint_scores = person_pose['keypoint_scores']

        # キーポイントを描画
        for i, (point, score) in enumerate(zip(keypoints, keypoint_scores)):
            if score > POSE_KEYPOINT_THRESHOLD:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 5, KEYPOINT_COLOR, -1, cv2.LINE_AA)

        # 骨格を描画
        for joint_pair in SKELETON:
            idx1, idx2 = joint_pair
            if keypoint_scores[idx1] > POSE_KEYPOINT_THRESHOLD and keypoint_scores[idx2] > POSE_KEYPOINT_THRESHOLD:
                pt1 = tuple(map(int, keypoints[idx1]))
                pt2 = tuple(map(int, keypoints[idx2]))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)
    return frame

# ------------------------------------------------------------------------
# メインの動画処理関数
# ------------------------------------------------------------------------

def detect_and_estimate_pose_in_video(video_path: str, progress=gr.Progress(track_tqdm=True)):
    """
    動画を解析し、プレイヤー検出と姿勢推定を行い、結果を描画した新しい動画を生成する。
    """
    if player_detector_model is None or player_detector_processor is None:
        raise gr.Error(f"プレイヤー検出モデルがロードされていません: {PLAYER_MODEL_LOAD_ERROR}")
    if pose_estimator_model is None or pose_estimator_processor is None:
        raise gr.Error(f"姿勢推定モデルがロードされていません: {POSE_MODEL_LOAD_ERROR}")
    if video_path is None:
        raise gr.Error("動画ファイルをアップロードしてください。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("動画ファイルを開けませんでした。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamp = int(time.time())
    output_filename = f"output_pose_detection_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (orig_width, orig_height))

    for _ in progress.tqdm(range(total_frames), desc="動画を処理中..."):
        ret, frame = cap.read()
        if not ret: break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = frame.copy()

        # --- Stage 1: プレイヤー検出 ---
        inputs_det = player_detector_processor(images=image, return_tensors="pt")
        inputs_det = {k: v.to(device) for k, v in inputs_det.items()}

        with torch.no_grad():
            outputs_det = player_detector_model(**inputs_det)

        target_size = torch.tensor([(image.height, image.width)]).to(device)
        results_det = player_detector_processor.post_process_object_detection(
            outputs_det, target_sizes=target_size, threshold=PLAYER_CONFIDENCE_THRESHOLD
        )[0]
        
        player_boxes = results_det["boxes"]
        
        # --- Stage 2: 姿勢推定 (プレイヤーが検出された場合のみ) ---
        if len(player_boxes) > 0:
            # 姿勢推定器用のボックス形式 (x1, y1, w, h) に変換
            boxes_coco = player_boxes.cpu().numpy().copy()
            boxes_coco[:, 2] = boxes_coco[:, 2] - boxes_coco[:, 0]
            boxes_coco[:, 3] = boxes_coco[:, 3] - boxes_coco[:, 1]
            
            inputs_pose = pose_estimator_processor(image, boxes=[boxes_coco], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs_pose = pose_estimator_model(**inputs_pose)
            
            pose_results = pose_estimator_processor.post_process_pose_estimation(outputs_pose, boxes=[boxes_coco])
            
            # 結果を描画
            output_frame = draw_poses(output_frame, pose_results[0]) # 最初の画像の結果

        # バウンディングボックスも描画
        for score, label_id, box in zip(results_det["scores"], results_det["labels"], player_boxes):
            box_int = [round(i) for i in box.tolist()]
            label = player_detector_model.config.id2label[label_id.item()]
            label_text = f"{label}: {score:.2f}"
            cv2.rectangle(output_frame, (box_int[0], box_int[1]), (box_int[2], box_int[3]), (36, 255, 12), 2)
            cv2.putText(output_frame, label_text, (box_int[0], box_int[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

        out_writer.write(output_frame)

    cap.release()
    out_writer.release()
    return output_filename

# ------------------------------------------------------------------------
# Gradioインターフェースの構築
# ------------------------------------------------------------------------
title = "Player Detection & Pose Estimation Demo"
description = """
**動画をアップロードすると、AIが2段階のパイプラインで動画を解析し、結果を描画した新しい動画を生成します。**

1.  **プレイヤー検出**: ファインチューニング済みのRT-DETRモデルが動画内のプレイヤーを検出します。
2.  **姿勢推定**: 検出された各プレイヤーに対して、ViT-Poseモデルが骨格キーポイントを推定します。

**使い方:**
1. `Upload Video` ボックスに動画ファイルをドラッグ＆ドロップするか、クリックしてファイルを選択します。
2. `Submit` ボタンを押して解析を開始します。（処理には時間がかかる場合があります）
3. 解析が完了すると、右側に結果の動画が表示されます。
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
        fn=detect_and_estimate_pose_in_video,
        inputs=[video_input],
        outputs=[video_output]
    )

    gr.Examples(
        examples=[],
        inputs=video_input,
        outputs=video_output,
        fn=detect_and_estimate_pose_in_video,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()