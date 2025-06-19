import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import time
import os
from collections import deque, OrderedDict
import matplotlib.pyplot as plt
import io

# ------------------------------------------------------------------------
# 必要なクラス定義とインポート
# ------------------------------------------------------------------------
try:
    # ベースモデルのアーキテクチャを直接インポート
    from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss
    from src.court.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocal
    from src.player.lit_module.lit_rtdetr import LitRtdetr
    from transformers import AutoProcessor, RTDetrImageProcessor, VitPoseForPoseEstimation
    from src.event.model.transformer_v2 import EventTransformerV2
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    IMPORT_SUCCESS = True
    print("All required modules imported successfully.")
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(e)
    print(f"Failed to import modules: {e}")

# ------------------------------------------------------------------------
# グローバル設定と全モデルのロード
# ------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

BALL_CKPT = "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
COURT_CKPT = "checkpoints/court/lit_lite_tracknet/epoch=010-val_loss=0.76632285.ckpt"
PLAYER_CKPT = "checkpoints/player/lit_rt_detr/epoch=002-val_loss_total=2.38540339.ckpt"
EVENT_CKPT = "checkpoints/event/transformer_v2.py/epoch=18-step=532.ckpt"

ball_model, court_model, player_model, pose_model, event_model = None, None, None, None, None
player_processor, pose_processor = None, None
ball_transform, court_transform = None, None
LOAD_ERROR = ""

if not IMPORT_SUCCESS:
    LOAD_ERROR = f"Module import failed: {IMPORT_ERROR_MESSAGE}"
else:
    try:
        # --- 1. Ball Model ---
        # モデルのロード
        if IMPORT_SUCCESS:
            if os.path.exists(BALL_CKPT):
                lit_model = LitLiteTracknetFocalLoss.load_from_checkpoint(BALL_CKPT, map_location=device)
                ball_model = lit_model.model
                ball_model.to(device)
                ball_model.eval()
                print(f"Model loaded successfully from '{BALL_CKPT}'")
            else:
                MODEL_LOAD_ERROR = f"Checkpoint file not found at '{BALL_CKPT}'"
                print(f"Error: {MODEL_LOAD_ERROR}")
        else:
            MODEL_LOAD_ERROR = f"Could not import LitLiteTracknetFocalLoss: {IMPORT_ERROR_MESSAGE}"
            print(f"Error: {MODEL_LOAD_ERROR}")

        ball_transform = A.ReplayCompose([
            A.Resize(height=360, width=640), A.Normalize(), ToTensorV2()
        ], keypoint_params=A.KeypointParams(format="xy"))
        print("Ball model loaded manually.")

        # --- 2. Court Model ---
        try:
            # --- 新しいモデルロード方式 ---
            # 1. モデルの「骨格」を先に作成
            # コートモデルの入力チャンネルは3 (RGB)
            court_model = LitLiteTracknetFocal.load_from_checkpoint(COURT_CKPT, map_location=device).model    
            court_model.to(device)
            court_model.eval() # 評価モードに設定
            print("Model weights loaded successfully by extracting from checkpoint.")

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at '{COURT_CKPT}'")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")


        # Transformの定義 (変更なし)
        court_transform = A.Compose( # 単一画像なのでComposeで十分
            [
                A.Resize(height=360, width=640),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ]
        )
        print("Court model loaded manually.")

        # --- 3. Player Model ---
        # モデルとプロセッサのロード
        if IMPORT_SUCCESS:
            if os.path.exists(PLAYER_CKPT):
                try:
                    # PyTorch Lightningモデルをロードし、評価モードに設定
                    lit_model = LitRtdetr.load_from_checkpoint(PLAYER_CKPT, map_location=device)
                    player_model = lit_model.to(device).eval()
                    
                    # 対応するRT-DETRのImage Processorをロード
                    player_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
                    print(f"Model loaded successfully from '{PLAYER_CKPT}' and moved to {device}.")
                except Exception as e:
                    MODEL_LOAD_ERROR = f"Failed to load model or processor: {e}"
                    print(f"Error: {MODEL_LOAD_ERROR}")
            else:
                MODEL_LOAD_ERROR = f"Checkpoint file not found at '{PLAYER_CKPT}'"
                print(f"Error: {MODEL_LOAD_ERROR}")
        else:
            MODEL_LOAD_ERROR = f"Could not import custom modules: {IMPORT_ERROR_MESSAGE}"
            print(f"Error: {MODEL_LOAD_ERROR}")

        # --- 4. Pose Model ---
        pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device).eval()
        print("Pose estimator loaded.")
        
        # --- 5. Event Model (★ ご要望の箇所) ---
        # チェックポイントからハイパーパラメータと重みを直接ロード
        checkpoint_event = torch.load(EVENT_CKPT, map_location=device, weights_only=False)
        hparams_event = checkpoint_event['hyper_parameters']

        # ハイパーパラメータを使ってモデルの骨格を初期化
        event_model = EventTransformerV2(
            d_model=hparams_event.get('d_model', 128),
            nhead=hparams_event.get('nhead', 8),
            num_layers=hparams_event.get('num_layers', 4),
            dropout=hparams_event.get('dropout', 0.1),
            max_seq_len=hparams_event.get('max_seq_len', 512),
            pose_dim=hparams_event.get('pose_dim', 51)
        )

        # 重み(state_dict)を抽出してロード
        state_dict_event = checkpoint_event['state_dict']
        cleaned_state_dict_event = OrderedDict()
        for k, v in state_dict_event.items():
            if k.startswith("model."):
                cleaned_state_dict_event[k[len("model."):]] = v
        
        event_model.load_state_dict(cleaned_state_dict_event)
        event_model.to(device).eval()
        print("Event detection model loaded directly from checkpoint.")


    except Exception as e:
        LOAD_ERROR = f"Failed to load one or more models: {e}"
        print(f"ERROR: {LOAD_ERROR}")
        raise e

# --- 定数 ---
SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]
EVENT_THRESHOLD = 0.5

# ------------------------------------------------------------------------
# 特徴量抽出＆推論パイプライン (このセクションは変更ありません)
# ------------------------------------------------------------------------
def analyze_video_for_events(video_path: str, progress=gr.Progress(track_tqdm=True)):
    if LOAD_ERROR:
        raise gr.Error(LOAD_ERROR)
    if video_path is None:
        raise gr.Error("動画ファイルをアップロードしてください。")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("動画ファイルを開けませんでした。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ========== ステージ1: 全フレームから特徴量を抽出 ==========
    progress(0, desc="Stage 1/5: Extracting features from video...")
    
    raw_frames = []
    ball_feats_list, court_feats_list = [], []
    player_bbox_list, player_pose_list = [], []
    
    frame_queue = deque(maxlen=3)

    for frame_idx in progress.tqdm(range(total_frames), desc="Extracting features"):
        ret, frame = cap.read()
        if not ret: break
        
        raw_frames.append(frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # --- Ball Feature ---
        frame_queue.append(image_rgb)
        if len(frame_queue) == 3:
            frames_np = list(frame_queue)
            transformed = [ball_transform(image=f)["image"] for f in frames_np]
            input_tensor = torch.cat(transformed, dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                heatmap_pred = ball_model(input_tensor)
                heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()
            
            max_prob = np.max(heatmap_prob)
            pos = np.unravel_index(np.argmax(heatmap_prob), heatmap_prob.shape)
            y_b, x_b = pos
            ball_feats_list.append(torch.tensor([x_b / 640, y_b / 360, max_prob], dtype=torch.float32))
        else:
            ball_feats_list.append(torch.zeros(3))

        # --- Court Feature ---
        court_in = court_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            heatmap_pred = court_model(court_in)
            heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()
        
        peaks = []
        heatmap_copy = heatmap_prob.copy()
        for _ in range(15):
            pos = np.unravel_index(np.argmax(heatmap_copy), heatmap_copy.shape)
            peaks.append(pos)
            cv2.circle(heatmap_copy, (pos[1], pos[0]), 10, 0, -1)
        
        court_vec = []
        for y_c, x_c in peaks:
            court_vec.extend([x_c / 640, y_c / 360, 1.0]) # visibility=1 for simplicity
        court_feats_list.append(torch.tensor(court_vec, dtype=torch.float32))

        # --- Player & Pose Features ---
        inputs_det = player_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_det = player_model(**inputs_det)
        results_det = player_processor.post_process_object_detection(
            outputs_det, target_sizes=torch.tensor([(h, w)]), threshold=0.5
        )[0]
        
        frame_bboxes, frame_poses = [], []
        player_boxes = results_det["boxes"]
        if len(player_boxes) > 0:
            boxes_coco = player_boxes.cpu().numpy().copy()
            boxes_coco[:, 2] -= boxes_coco[:, 0]
            boxes_coco[:, 3] -= boxes_coco[:, 1]
            
            inputs_pose = pose_processor(pil_image, boxes=[boxes_coco], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_pose = pose_model(**inputs_pose)
            pose_results = pose_processor.post_process_pose_estimation(outputs_pose, boxes=[boxes_coco])[0]
            
            for p_idx, p_box in enumerate(results_det["boxes"]):
                x1, y1, x2, y2 = p_box.tolist()
                score = results_det["scores"][p_idx].item()
                frame_bboxes.append(torch.tensor([x1/w, y1/h, x2/w, y2/h, score]))

                kps = pose_results[p_idx]['keypoints']
                kps_scores = pose_results[p_idx]['scores']
                pose_vec = []
                for kp_idx in range(len(kps)):
                    kx, ky = kps[kp_idx]
                    kv = 2 if kps_scores[kp_idx] > 0.5 else 1
                    pose_vec.extend([kx/w, ky/h, kv])
                frame_poses.append(torch.tensor(pose_vec))
        
        player_bbox_list.append(frame_bboxes)
        player_pose_list.append(frame_poses)
    cap.release()
    
    # ========== ステージ2: 特徴量テンソルの準備 ==========
    progress(0, desc="Stage 2/5: Preparing feature tensors...")
    
    ball_tensor = torch.stack(ball_feats_list).unsqueeze(0)
    court_tensor = torch.stack(court_feats_list).unsqueeze(0)

    max_players = max(len(p) for p in player_bbox_list) if any(player_bbox_list) else 0
    padded_bbox_frames, padded_pose_frames = [], []
    for bboxes, poses in zip(player_bbox_list, player_pose_list):
        if bboxes:
            bbox_t = torch.stack(bboxes)
            pose_t = torch.stack(poses)
        else:
            bbox_t = torch.zeros((0, 5))
            pose_t = torch.zeros((0, 17*3))
        
        pad_n = max_players - len(bboxes)
        if pad_n > 0:
            bbox_t = torch.cat([bbox_t, torch.zeros((pad_n, 5))], dim=0)
            pose_t = torch.cat([pose_t, torch.zeros((pad_n, 17*3))], dim=0)
        padded_bbox_frames.append(bbox_t)
        padded_pose_frames.append(pose_t)

    player_bbox_tensor = torch.stack(padded_bbox_frames).unsqueeze(0)
    player_pose_tensor = torch.stack(padded_pose_frames).unsqueeze(0)

    # ========== ステージ3: イベント検出 ==========
    progress(0.25, desc="Stage 3/5: Detecting events...")
    with torch.no_grad():
        logits = event_model(
            ball_tensor.to(device),
            player_bbox_tensor.to(device),
            player_pose_tensor.to(device),
            court_tensor.to(device)
        )
    event_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # ========== ステージ4: 結果のプロット ==========
    progress(0.5, desc="Stage 4/5: Generating event plot...")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(event_probs[:, 0], label="Hit Probability", color="skyblue")
    ax.plot(event_probs[:, 1], label="Bounce Probability", color="salmon")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_title("Event Detection Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = Image.open(buf)
    plt.close(fig)

    # ========== ステージ5: 結果動画のレンダリング ==========
    progress(0.75, desc="Stage 5/5: Rendering output video...")
    
    output_filename = f"output_event_detection_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    for idx, frame in enumerate(progress.tqdm(raw_frames, desc="Rendering video")):
        if idx > 1:
            p1_norm, p2_norm = ball_feats_list[idx-1][:2], ball_feats_list[idx][:2]
            p1 = (int(p1_norm[0]*w), int(p1_norm[1]*h))
            p2 = (int(p2_norm[0]*w), int(p2_norm[1]*h))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
        hit_prob, bounce_prob = event_probs[idx]
        if hit_prob > EVENT_THRESHOLD:
            cv2.putText(frame, "HIT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
        if bounce_prob > EVENT_THRESHOLD:
            cv2.putText(frame, "BOUNCE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        out_writer.write(frame)
        
    out_writer.release()
    
    return plot_image, output_filename


# ------------------------------------------------------------------------
# Gradioインターフェース (このセクションは変更ありません)
# ------------------------------------------------------------------------
title = "Tennis Event Detection Demo"
description = """
**動画をアップロードすると、AIがボール、コート、プレイヤー、ポーズの4つの情報を統合し、ヒットとバウンドのイベントを検出します。**
1.  **特徴量抽出**: 動画の全フレームから、4つの上流モデルを用いて特徴量を抽出します。
2.  **イベント検出**: 抽出された時系列特徴量をTransformerモデルに入力し、イベント確率を計算します。
3.  **可視化**: イベント確率の時系列グラフと、イベント情報を描画した動画を生成します。
**注意:** 処理には非常に長い時間がかかる場合があります。
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)
    with gr.Row():
        video_input = gr.Video(label="Upload Tennis Video")
    with gr.Row():
        submit_button = gr.Button("Analyze Video", variant="primary")
    with gr.Row():
        with gr.Column(scale=1):
            plot_output = gr.Image(label="Event Probability Plot")
        with gr.Column(scale=1):
            video_output = gr.Video(label="Result Video with Events")

    submit_button.click(
        fn=analyze_video_for_events,
        inputs=[video_input],
        outputs=[plot_output, video_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)