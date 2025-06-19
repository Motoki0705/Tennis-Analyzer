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
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List
from scipy.interpolate import make_interp_spline

# ------------------------------------------------------------------------
# 必要なクラス定義とインポート (元のコードと同じ)
# ------------------------------------------------------------------------
try:
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
# 1. 設定の集中管理 (リファクタリング)
# ------------------------------------------------------------------------
@dataclass
class AppConfig:
    """アプリケーション全体の設定を管理するクラス"""
    # --- パス設定 ---
    BALL_CKPT: str = "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
    COURT_CKPT: str = "checkpoints/court/lit_lite_tracknet/epoch=010-val_loss=0.76632285.ckpt"
    PLAYER_CKPT: str = "checkpoints/player/lit_rt_detr/epoch=002-val_loss_total=2.38540339.ckpt"
    EVENT_CKPT: str = "checkpoints/event/transformer_v2.py/epoch=18-step=532.ckpt"
    POSE_HF_ID: str = "usyd-community/vitpose-base-simple"
    PLAYER_HF_ID: str = "PekingU/rtdetr_v2_r18vd"

    # --- モデル設定 ---
    IMG_WIDTH: int = 640
    IMG_HEIGHT: int = 360
    BALL_QUEUE_SIZE: int = 3
    POSE_DIM: int = 17 * 3  # 17 keypoints * (x, y, visibility)

    # --- 推論設定 ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PLAYER_THRESHOLD: float = 0.5
    EVENT_THRESHOLD: float = 0.5
    
    # --- 描画設定 ---
    SKELETON: List[List[int]] = field(default_factory=lambda: [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]])

# ------------------------------------------------------------------------
# 2. モデルロード処理のカプセル化 (リファクタリング)
# ------------------------------------------------------------------------
class ModelLoader:
    """
    全てのモデルとプロセッサ、変換パイプラインをロードし、保持するクラス。
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.models: Dict[str, Any] = {}
        self.transforms: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.load_error: str = ""

    def load_all(self):
        """全てのコンポーネントをロードする"""
        if not IMPORT_SUCCESS:
            self.load_error = f"Module import failed: {IMPORT_ERROR_MESSAGE}"
            print(f"ERROR: {self.load_error}")
            return
        
        try:
            print(f"Using device: {self.cfg.DEVICE}")
            self._load_ball_model()
            self._load_court_model()
            self._load_player_model()
            self._load_pose_model()
            self._load_event_model()
            print("\nAll models, processors, and transforms loaded successfully.")
        except Exception as e:
            self.load_error = f"Failed to load one or more models: {e}"
            print(f"ERROR: {self.load_error}")
            # エラーが発生した場合、Gradioアプリがクラッシュしないように例外を再送出
            raise e

    def _load_ball_model(self):
        lit_model = LitLiteTracknetFocalLoss.load_from_checkpoint(self.cfg.BALL_CKPT, map_location=self.cfg.DEVICE)
        self.models['ball'] = lit_model.model.to(self.cfg.DEVICE).eval()
        self.transforms['ball'] = A.ReplayCompose([
            A.Resize(height=self.cfg.IMG_HEIGHT, width=self.cfg.IMG_WIDTH), A.Normalize(), ToTensorV2()
        ], keypoint_params=A.KeypointParams(format="xy"))
        print("Ball model loaded.")

    def _load_court_model(self):
        self.models['court'] = LitLiteTracknetFocal.load_from_checkpoint(self.cfg.COURT_CKPT, map_location=self.cfg.DEVICE).model.to(self.cfg.DEVICE).eval()
        self.transforms['court'] = A.Compose([
            A.Resize(height=self.cfg.IMG_HEIGHT, width=self.cfg.IMG_WIDTH), A.Normalize(), A.pytorch.ToTensorV2()
        ])
        print("Court model loaded.")
        
    def _load_player_model(self):
        lit_model = LitRtdetr.load_from_checkpoint(self.cfg.PLAYER_CKPT, map_location=self.cfg.DEVICE)
        self.models['player'] = lit_model.to(self.cfg.DEVICE).eval()
        self.processors['player'] = RTDetrImageProcessor.from_pretrained(self.cfg.PLAYER_HF_ID)
        print("Player detection model loaded.")

    def _load_pose_model(self):
        self.models['pose'] = VitPoseForPoseEstimation.from_pretrained(self.cfg.POSE_HF_ID).to(self.cfg.DEVICE).eval()
        self.processors['pose'] = AutoProcessor.from_pretrained(self.cfg.POSE_HF_ID)
        print("Pose estimation model loaded.")
        
    def _load_event_model(self):
        self.models['event'] = EventTransformerV2.load_from_checkpoint(self.cfg.EVENT_CKPT, map_location=self.cfg.DEVICE).to(self.cfg.DEVICE).eval()  
        print("Event detection model loaded.")


# ------------------------------------------------------------------------
# 3. 機能ごとの関数分割 (リファクタリング)
# ------------------------------------------------------------------------
def extract_features(video_path: str, loader: ModelLoader, cfg: AppConfig, progress: gr.Progress) -> Dict[str, Any]:
    """
    動画から全てのフレームの特徴量を抽出する。
    ボール、コート、選手BBox、選手ポーズをフレームごとに検出し、リストに格納して返す。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("動画ファイルを開けませんでした。")

    # ビデオのプロパティを取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 結果を格納するための辞書
    results = {
        "raw_frames": [], "ball_feats": [], "court_feats": [],
        "player_bboxes": [], "player_poses": [],
        "original_size": (w, h), "fps": fps
    }
    
    ball_frame_queue = deque(maxlen=cfg.BALL_QUEUE_SIZE)

    # tqdmプログレスバーでフレームごとに処理
    for frame_idx in progress.tqdm(range(total_frames), desc="Stage 1: Extracting features"):
        ret, frame = cap.read()
        if not ret: break
        
        results["raw_frames"].append(frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # --- 1. ボール特徴量抽出 ---
        ball_frame_queue.append(image_rgb)
        if len(ball_frame_queue) == cfg.BALL_QUEUE_SIZE:
            transformed = [loader.transforms['ball'](image=f)["image"] for f in ball_frame_queue]
            input_tensor = torch.cat(transformed, dim=0).unsqueeze(0).to(cfg.DEVICE)
            with torch.no_grad():
                heatmap = torch.sigmoid(loader.models['ball'](input_tensor)).squeeze().cpu().numpy()
            pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y_b, x_b = pos
            max_prob = np.max(heatmap)
            results["ball_feats"].append(torch.tensor([x_b / cfg.IMG_WIDTH, y_b / cfg.IMG_HEIGHT, max_prob], dtype=torch.float32))
        else:
            results["ball_feats"].append(torch.zeros(3, dtype=torch.float32))

        # --- 2. コート特徴量抽出 ---
        court_in = loader.transforms['court'](image=image_rgb)["image"].unsqueeze(0).to(cfg.DEVICE)
        with torch.no_grad():
            heatmap_pred = loader.models['court'](court_in)
            heatmap_prob = torch.sigmoid(heatmap_pred).squeeze().cpu().numpy()
        
        peaks = []
        heatmap_copy = heatmap_prob.copy()
        # ヒートマップから最も確率の高い15点（キーポイント）を抽出
        for _ in range(15):
            pos = np.unravel_index(np.argmax(heatmap_copy), heatmap_copy.shape)
            peaks.append(pos)
            # 抽出した点の周辺を0にして、次のピークを見つけやすくする
            cv2.circle(heatmap_copy, (pos[1], pos[0]), radius=10, color=0, thickness=-1)
        
        court_vec = []
        for y_c, x_c in peaks:
            # 座標を正規化し、仮の可視性(1.0)と共に格納
            court_vec.extend([x_c / cfg.IMG_WIDTH, y_c / cfg.IMG_HEIGHT, 1.0])
        results["court_feats"].append(torch.tensor(court_vec, dtype=torch.float32))

        # --- 3. 選手 & ポーズ特徴量抽出 ---
        inputs_det = loader.processors['player'](images=pil_image, return_tensors="pt").to(cfg.DEVICE)
        with torch.no_grad():
            outputs_det = loader.models['player'](**inputs_det)
        
        # 検出結果を後処理
        res_det = loader.processors['player'].post_process_object_detection(
            outputs_det, target_sizes=torch.tensor([(h, w)]), threshold=cfg.PLAYER_THRESHOLD
        )[0]
        
        frame_bboxes, frame_poses = [], []
        player_boxes = res_det["boxes"]

        if len(player_boxes) > 0:
            # ViTPose用のBBox形式 (x, y, width, height) に変換
            boxes_coco = player_boxes.cpu().numpy().copy()
            boxes_coco[:, 2] -= boxes_coco[:, 0]
            boxes_coco[:, 3] -= boxes_coco[:, 1]
            
            # 姿勢推定の実行
            inputs_pose = loader.processors['pose'](images=pil_image, boxes=[boxes_coco], return_tensors="pt").to(cfg.DEVICE)
            with torch.no_grad():
                outputs_pose = loader.models['pose'](**inputs_pose)
            pose_results = loader.processors['pose'].post_process_pose_estimation(outputs_pose, boxes=[boxes_coco])[0]
            
            # 検出された各選手に対して処理
            for p_idx, p_box in enumerate(res_det["boxes"]):
                x1, y1, x2, y2 = p_box.tolist()
                score = res_det["scores"][p_idx].item()
                # BBoxを正規化して格納
                frame_bboxes.append(torch.tensor([x1/w, y1/h, x2/w, y2/h, score], dtype=torch.float32))

                # ポーズ情報を正規化して格納
                kps = pose_results[p_idx]['keypoints']
                kps_scores = pose_results[p_idx]['scores']
                pose_vec = []
                for kp_idx in range(len(kps)):
                    kx, ky = kps[kp_idx]
                    # 論文等で使われるCOCOの可視性フラグ (0:なし, 1:隠れている, 2:見える)
                    kv = 2 if kps_scores[kp_idx] > 0.5 else 1 
                    pose_vec.extend([kx/w, ky/h, kv])
                frame_poses.append(torch.tensor(pose_vec, dtype=torch.float32))
        
        results["player_bboxes"].append(frame_bboxes)
        results["player_poses"].append(frame_poses)

    cap.release()
    return results


def prepare_tensors(features: Dict[str, Any], cfg: AppConfig) -> Dict[str, torch.Tensor]:
    """
    特徴量リストをバッチ化・パディングし、モデル入力用のテンソルを作成する。
    """
    # --- 1. ボールとコートのテンソル化 ---
    # リストをスタックして (シーケンス長, 特徴量次元) のテンソルにし、バッチ次元を追加
    ball_tensor = torch.stack(features["ball_feats"]).unsqueeze(0)
    court_tensor = torch.stack(features["court_feats"]).unsqueeze(0)

    # --- 2. 選手関連テンソルのパディングとテンソル化 ---
    # ビデオ全体で同時に検出された選手の最大数を計算
    max_players = max(len(p) for p in features["player_bboxes"]) if any(features["player_bboxes"]) else 0
    
    padded_bbox_frames, padded_pose_frames = [], []

    # フレームごとにループし、選手数を最大値に揃える
    for bboxes, poses in zip(features["player_bboxes"], features["player_poses"]):
        # 現在のフレームで検出された選手数をテンソル化
        if bboxes:
            bbox_t = torch.stack(bboxes)
            pose_t = torch.stack(poses)
        else: # 選手が検出されなかった場合
            bbox_t = torch.zeros((0, 5), dtype=torch.float32) # (選手数=0, bbox次元=5)
            pose_t = torch.zeros((0, cfg.POSE_DIM), dtype=torch.float32) # (選手数=0, pose次元=51)
        
        # パディングが必要な数を計算
        pad_n = max_players - len(bboxes)
        if pad_n > 0:
            # 足りない分をゼロのテンソルで埋める
            bbox_pad = torch.zeros((pad_n, 5), dtype=torch.float32)
            pose_pad = torch.zeros((pad_n, cfg.POSE_DIM), dtype=torch.float32)
            bbox_t = torch.cat([bbox_t, bbox_pad], dim=0)
            pose_t = torch.cat([pose_t, pose_pad], dim=0)
            
        padded_bbox_frames.append(bbox_t)
        padded_pose_frames.append(pose_t)

    # パディング済みのフレームリストをスタックし、バッチ次元を追加
    player_bbox_tensor = torch.stack(padded_bbox_frames).unsqueeze(0)
    player_pose_tensor = torch.stack(padded_pose_frames).unsqueeze(0)
    
    return {
        "ball": ball_tensor,
        "court": court_tensor,
        "player_bbox": player_bbox_tensor,
        "player_pose": player_pose_tensor
    }


def detect_events(tensors: Dict[str, torch.Tensor], model: torch.nn.Module, device: str) -> np.ndarray:
    """イベント検出モデルを実行し、確率を返す"""
    with torch.no_grad():
        logits = model(
            tensors['ball'].to(device), tensors['player_bbox'].to(device),
            tensors['player_pose'].to(device), tensors['court'].to(device)
        )
    return torch.sigmoid(logits).squeeze().cpu().numpy()

def create_event_plot(event_probs: np.ndarray) -> Image.Image:
    """イベント確率の時系列プロットを生成する"""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(event_probs[:, 0], label="Hit Probability", color="skyblue")
    ax.plot(event_probs[:, 1], label="Bounce Probability", color="salmon")
    ax.set_xlabel("Frame"); ax.set_ylabel("Probability"); ax.set_title("Event Detection Over Time")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png'); buf.seek(0)
    plot_image = Image.open(buf)
    plt.close(fig)
    return plot_image

def create_ecg_style_plot(event_probs: np.ndarray, threshold: float = 0.5) -> Image.Image:
    """
    イベント確率を滑らかな心電図風のグラフとして描画する。
    
    Args:
        event_probs (np.ndarray): (フレーム数, 2) の形状の配列。[0]がヒット、[1]がバウンド確率。
        threshold (float): イベントとして検出する確率の閾値。

    Returns:
        Image.Image: 生成されたプロットのPIL画像オブジェクト。
    """
    try:
        # スタイルを適用してモダンな外観に
        plt.style.use('dark_background')

        # 2つのサブプロットを縦に並べて作成 (横12インチ, 縦6インチ)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.patch.set_facecolor('#121212') # Figure全体の背景色

        time_axis = np.arange(event_probs.shape[0])
        hit_probs = event_probs[:, 0]
        bounce_probs = event_probs[:, 1]
        
        # --- 共通の描画処理を関数化 ---
        def plot_single_event(ax, probs, title, line_color, fill_color, peak_color):
            ax.set_facecolor('#1E1E1E') # 各グラフの背景色
            ax.set_title(title, fontsize=16, color='white', pad=15)
            ax.set_ylabel("Probability", fontsize=12, color='gray')
            ax.set_ylim(0, 1.05)
            
            # 枠線の色と太さを調整
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_color('gray')
                ax.spines[spine].set_linewidth(0.5)
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')

            # 背景グリッド
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#444444')
            
            # --- 曲線の平滑化処理 ---
            if len(time_axis) > 3: # スプライン補間には3点より多く必要
                # 元のデータ点の3倍の解像度で滑らかなx軸を生成
                time_axis_smooth = np.linspace(time_axis.min(), time_axis.max(), len(time_axis) * 3)
                spl = make_interp_spline(time_axis, probs, k=3)  # k=3: 3次スプライン
                probs_smooth = spl(time_axis_smooth)
            else: # データ点が少ない場合はそのままプロット
                time_axis_smooth = time_axis
                probs_smooth = probs

            # --- グラフの描画 ---
            # 1. 滑らかな曲線
            ax.plot(time_axis_smooth, probs_smooth, color=line_color, linewidth=2.5, label='Probability')
            # 2. 曲線の下の塗りつぶし
            ax.fill_between(time_axis_smooth, probs_smooth, color=fill_color, alpha=0.3)
            
            # --- ピークの検出とマーキング ---
            peak_indices = np.where(probs > threshold)[0]
            if len(peak_indices) > 0:
                ax.scatter(peak_indices, probs[peak_indices], color=peak_color, s=60, zorder=5, edgecolor='white', linewidth=1, label=f'Event Detected')
            
            ax.legend(loc='upper right', frameon=True, facecolor='#333333', edgecolor='none', labelcolor='white')

        # --- Hitプロットを描画 ---
        plot_single_event(ax1, hit_probs, 'Hit Event Activity', 
                        line_color='#FF8C00', fill_color='#FF8C00', peak_color='#FFD700')
        
        # --- Bounceプロットを描画 ---
        plot_single_event(ax2, bounce_probs, 'Bounce Event Activity',
                        line_color='#00BFFF', fill_color='#00BFFF', peak_color='#7FFFD4')
        
        ax2.set_xlabel("Frame", fontsize=12, color='gray')
        
        plt.tight_layout(pad=2.0)
        
        # 描画したグラフを画像としてメモリに保存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
        buf.seek(0)
        plot_image = Image.open(buf)
        
        return plot_image

    finally:
        # matplotlibのスタイルが他の処理に影響しないようにリセット
        plt.close('all')
        plt.style.use('default')

def render_output_video(features: Dict[str, Any], event_probs: np.ndarray, cfg: AppConfig, progress: gr.Progress) -> str:
    """結果をオーバーレイ描画した動画を生成する"""
    output_filename = f"output_event_detection_{int(time.time())}.mp4"
    w, h = features["original_size"]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_filename, fourcc, features["fps"], (w, h))

    for idx, frame in enumerate(progress.tqdm(features["raw_frames"], desc="Rendering video")):
        # ボールの軌跡描画
        if idx > 1:
            p1_norm, p2_norm = features["ball_feats"][idx-1][:2], features["ball_feats"][idx][:2]
            p1 = (int(p1_norm[0] * w), int(p1_norm[1] * h))
            p2 = (int(p2_norm[0] * w), int(p2_norm[1] * h))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
        
        # イベントテキスト描画
        hit_prob, bounce_prob = event_probs[idx]
        if hit_prob > cfg.EVENT_THRESHOLD:
            cv2.putText(frame, "HIT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
        if bounce_prob > cfg.EVENT_THRESHOLD:
            cv2.putText(frame, "BOUNCE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        out_writer.write(frame)
        
    out_writer.release()
    return output_filename


# ------------------------------------------------------------------------
# 4. メインの処理フロー (リファクタリング)
# ------------------------------------------------------------------------
# グローバル変数としてアプリケーション設定とモデルローダーを初期化
CONFIG = AppConfig()
LOADER = ModelLoader(CONFIG)
# アプリケーション開始時に一度だけモデルをロードする
# Gradioのライフサイクルで管理するのがより望ましい
try:
    LOADER.load_all()
except Exception:
    # エラーが発生してもGradio UIは起動できるようにする
    pass

def full_analysis_pipeline(video_path: str, progress=gr.Progress(track_tqdm=True)):
    """
    全てのアナリシスパイプラインを統合して実行するメイン関数。
    """
    if LOADER.load_error:
        raise gr.Error(LOADER.load_error)
    if video_path is None:
        raise gr.Error("動画ファイルをアップロードしてください。")

    # --- ステージ1: 特徴量抽出 ---
    progress(0, desc="Stage 1/4: Extracting features...")
    # 注意: ここでは簡略化のため、`extract_features`の完全な実装は省略しています。
    # 実際には、元のコードからボール、コート、選手、ポーズの抽出ロジックをすべて移植する必要があります。
    features = extract_features(video_path, LOADER, CONFIG, progress)
    # 以下の行はダミーです。上記の行を実際のコードで有効化してください。


    # --- ステージ2: テンソル準備 ---
    progress(0.25, desc="Stage 2/4: Preparing feature tensors...")
    input_tensors = prepare_tensors(features, CONFIG)

    # --- ステージ3: イベント検出 ---
    progress(0.5, desc="Stage 3/4: Detecting events...")
    event_probs = detect_events(input_tensors, LOADER.models['event'], CONFIG.DEVICE)

    # --- ステージ4: 結果生成 ---
    progress(0.75, desc="Stage 4/4: Generating outputs...")
    plot_image = create_ecg_style_plot(event_probs, threshold=CONFIG.EVENT_THRESHOLD)
    output_video_path = render_output_video(features, event_probs, CONFIG, progress)

    return plot_image, output_video_path
    

# ------------------------------------------------------------------------
# Gradioインターフェース (変更は最小限)
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
    
    if LOADER.load_error:
        gr.Error(f"致命的なエラー: モデルのロードに失敗しました。アプリケーションを再起動してください。詳細: {LOADER.load_error}")
    
    with gr.Row():
        video_input = gr.Video(label="Upload Tennis Video")
    with gr.Row():
        submit_button = gr.Button("Analyze Video", variant="primary")
    with gr.Row():
        with gr.Column(scale=1):
            plot_output = gr.Image(label="Event Probability Plot")
        with gr.Column(scale=1):
            video_output = gr.Video(label="Result Video with Events")

    # `full_analysis_pipeline`を呼び出すように変更
    submit_button.click(
        fn=full_analysis_pipeline, # ここをリファクタリング後のメイン関数に変更
        inputs=[video_input],
        outputs=[plot_output, video_output]
    )

if __name__ == "__main__":
    # share=Trueはセキュリティリスクがあるため、開発時以外は注意
    demo.launch(share=True)