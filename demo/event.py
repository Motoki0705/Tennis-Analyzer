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
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List
from scipy.interpolate import make_interp_spline

# ------------------------------------------------------------------------
# 必要なクラス定義とインポート
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

class BallTrajectorySmoother:
    """
    ボールの検出結果（軌道）を平滑化し、信頼性を向上させるクラス。
    - 確率の低い検出を無効化
    - 物理的にありえない速度のジャンプ（外れ値）を除去
    - 欠損した軌道を補間
    """
    def __init__(self,
                 probability_threshold: float = 0.65,
                 max_speed_pixels: float = 150.0,
                 max_gap_frames: int = 10,
                 interpolation_method: str = 'linear'):
        """
        Args:
            probability_threshold (float): この確率未満の検出は無効とみなす。
            max_speed_pixels (float): 1フレームあたりの最大許容移動速度（ピクセル）。これを超えると外れ値とみなす。
            max_gap_frames (int): 補間を行う最大の連続フレーム数。これを超えてボールが見えない場合は補間しない。
            interpolation_method (str): 補間方法 ('linear', 'spline', 'polynomial' など)。
        """
        self.prob_threshold = probability_threshold
        self.max_speed_pixels = max_speed_pixels
        self.max_gap_frames = max_gap_frames
        self.interp_method = interpolation_method
        print(f"Ball smoother initialized with: "
              f"Threshold={self.prob_threshold}, Max Speed={self.max_speed_pixels}px/frame, "
              f"Max Gap={self.max_gap_frames} frames")

    def process(self, ball_feats: List[torch.Tensor], original_size: Tuple[int, int]) -> List[torch.Tensor]:
        """
        生のボール特徴量リストを受け取り、処理済みのリストを返す。

        Args:
            ball_feats (List[torch.Tensor]): (x, y, prob) のテンソルのリスト。
            original_size (Tuple[int, int]): 動画の元のサイズ (width, height)。

        Returns:
            List[torch.Tensor]: 平滑化されたボール特徴量のリスト。
        """
        if not ball_feats:
            return []

        w, h = original_size
        
        # --- 1. 確率に基づき信頼性の低い点を無効化 (NaNにする) ---
        coords = []
        for feat in ball_feats:
            prob = feat[2].item()
            if prob < self.prob_threshold:
                coords.append([np.nan, np.nan, prob])
            else:
                coords.append(feat.tolist())
        
        coords = np.array(coords)

        # --- 2. 異常な速度のジャンプ（外れ値）を無効化 ---
        last_valid_idx = -1
        for i in range(len(coords)):
            if np.isnan(coords[i, 0]):
                continue  # 既に無効な点はスキップ
            
            if last_valid_idx != -1:
                # 最後の有効点からのピクセル単位の距離と速度を計算
                prev_x, prev_y = coords[last_valid_idx, 0], coords[last_valid_idx, 1]
                curr_x, curr_y = coords[i, 0], coords[i, 1]
                
                dist_pixels = np.sqrt(((curr_x - prev_x) * w)**2 + ((curr_y - prev_y) * h)**2)
                frame_diff = i - last_valid_idx
                speed_pixels_per_frame = dist_pixels / frame_diff if frame_diff > 0 else 0
                
                if speed_pixels_per_frame > self.max_speed_pixels:
                    # 速度が速すぎるので外れ値とみなし、無効化
                    coords[i, :2] = np.nan
                    continue  # last_valid_idxを更新しないことで、この点をスキップ

            last_valid_idx = i

        # --- 3. 欠損データを補間 (pandasを使用) ---
        df = pd.DataFrame(coords, columns=['x', 'y', 'prob'])
        
        # limitで長すぎるギャップの補間を防ぐ
        df['x'] = df['x'].interpolate(method=self.interp_method, limit=self.max_gap_frames, limit_direction='forward')
        df['y'] = df['y'].interpolate(method=self.interp_method, limit=self.max_gap_frames, limit_direction='forward')
        
        # 補間後も残っているNaNは0で埋める（軌道の始点・終点など）
        df.fillna(0.0, inplace=True)

        # --- 4. 結果をtorch.Tensorのリストに再構成 ---
        smoothed_feats = [
            torch.tensor([row.x, row.y, row.prob], dtype=torch.float32)
            for row in df.itertuples(index=False)
        ]

        return smoothed_feats

# ------------------------------------------------------------------------
# 1. 設定の集中管理
# ------------------------------------------------------------------------
@dataclass
class AppConfig:
    """アプリケーション全体の設定を管理するクラス"""
    # --- パス設定 ---
    BALL_CKPT: str = "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
    COURT_CKPT: str = "checkpoints/court/lit_lite_tracknet/epoch=016-val_loss=0.00751106.ckpt"
    PLAYER_CKPT: str = "checkpoints/player/lit_rt_detr/epoch=002-val_loss_total=2.38540339.ckpt"
    EVENT_CKPT: str = "checkpoints/event/transformer_v2/epoch=18-step=532.ckpt"
    POSE_HF_ID: str = "usyd-community/vitpose-base-simple"
    PLAYER_HF_ID: str = "PekingU/rtdetr_v2_r18vd"

    # --- モデル設定 ---
    IMG_WIDTH: int = 640
    IMG_HEIGHT: int = 360
    BALL_QUEUE_SIZE: int = 3
    POSE_DIM: int = 17 * 3  # 17 keypoints * (x, y, visibility)

    # --- 推論設定 ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PLAYER_THRESHOLD: float = 0.85
    EVENT_THRESHOLD: float = 0.9
    BALL_PROB_THRESHOLD: float = 0.85  # この確率未満のボール検出を無視
    BALL_MAX_SPEED_PIXELS_PER_FRAME: float = 150.0  # 1フレームでの最大移動量(px)
    BALL_MAX_GAP_FRAMES: int = 10  # ボールを補間する最大の連続フレーム数
    
    # --- 描画設定 ---
    SKELETON: List[List[int]] = field(default_factory=lambda: [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]])

# ------------------------------------------------------------------------
# 2. モデルロード処理のカプセル化
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
        # PyTorch Lightningのバージョン差異による問題を避けるため、手動でロード
        checkpoint_event = torch.load(self.cfg.EVENT_CKPT, map_location=self.cfg.DEVICE, weights_only=False)
        
        # チェックポイントにハイパーパラメータが含まれているか確認
        if 'hyper_parameters' not in checkpoint_event:
            raise KeyError("Checkpoint for event model does not contain 'hyper_parameters'. Cannot initialize model.")
            
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

        # 重み(state_dict)を抽出し、キーから "model." プレフィックスを削除してロード
        state_dict_event = checkpoint_event['state_dict']
        cleaned_state_dict_event = OrderedDict()
        for k, v in state_dict_event.items():
            if k.startswith("model."):
                cleaned_state_dict_event[k[len("model."):]] = v
            else:
                cleaned_state_dict_event[k] = v # プレフィックスがない場合も考慮
        
        event_model.load_state_dict(cleaned_state_dict_event)
        self.models['event'] = event_model.to(self.cfg.DEVICE).eval()
        print("Event detection model loaded.")

# ------------------------------------------------------------------------
# 3. 機能ごとの関数分割 (★バッチ処理対応版★)
# ------------------------------------------------------------------------
def extract_features_batch(video_path: str, loader: ModelLoader, cfg: AppConfig, batch_sizes: Dict[str, int], progress: gr.Progress) -> Dict[str, Any]:
    """
    動画から全てのフレームの特徴量を【バッチ処理】で抽出する。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("動画ファイルを開けませんでした。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    results = {
        "raw_frames": [],
        "ball_feats": [torch.zeros(3, dtype=torch.float32) for _ in range(total_frames)],
        "court_feats": [],
        "player_bboxes": [],
        "player_poses": [],
        "original_size": (w, h),
        "fps": fps
    }
    
    frame_idx_list, pil_image_batch, rgb_image_batch = [], [], []
    ball_frame_queue = deque(maxlen=cfg.BALL_QUEUE_SIZE)
    ball_input_batch, ball_target_frame_indices = [], []

    pbar = progress.tqdm(range(total_frames), desc="Stage 1: Extracting features (Batch)")

    def process_player_and_pose_batch():
        """溜まったCourt/Player/Poseのバッチを処理する内部関数"""
        if not pil_image_batch: return

        # --- Courtモデル (バッチ推論) ---
        court_in_batch = torch.stack([loader.transforms['court'](image=img)["image"] for img in rgb_image_batch]).to(cfg.DEVICE)
        # ★★★★★ FIX 1 ★★★★★
        with torch.no_grad(), torch.autocast(device_type=cfg.DEVICE, dtype=torch.bfloat16):
            heatmap_preds = loader.models['court'](court_in_batch)
        # The result of autocast is float16, so convert back to float32 for CPU operations
        heatmap_probs = torch.sigmoid(heatmap_preds).squeeze(1).to(torch.float32).cpu().numpy()


        for heatmap_prob in heatmap_probs:
            peaks, heatmap_copy = [], heatmap_prob.copy()
            for _ in range(15):
                pos = np.unravel_index(np.argmax(heatmap_copy), heatmap_copy.shape)
                peaks.append(pos)
                cv2.circle(heatmap_copy, (pos[1], pos[0]), radius=10, color=0, thickness=-1)
            court_vec = [coord for y_c, x_c in peaks for coord in (x_c / cfg.IMG_WIDTH, y_c / cfg.IMG_HEIGHT, 1.0)]
            results["court_feats"].append(torch.tensor(court_vec, dtype=torch.float32))

        # --- Playerモデル (バッチ推論) ---
        target_sizes = [img.size[::-1] for img in pil_image_batch]
        inputs_det = loader.processors['player'](images=pil_image_batch, return_tensors="pt").to(cfg.DEVICE)
        # ★★★★★ FIX 2 ★★★★★
        with torch.no_grad(), torch.autocast(device_type=cfg.DEVICE, dtype=torch.bfloat16):
            outputs_det = loader.models['player'](**inputs_det)
        res_det_batch = loader.processors['player'].post_process_object_detection(
            outputs_det, target_sizes=torch.tensor(target_sizes), threshold=cfg.PLAYER_THRESHOLD
        )

        # --- Poseモデル (バッチ推論) ---
        pose_image_batch, pose_bbox_batch, pose_frame_map = [], [], {}
        temp_player_results = {idx: ([], []) for idx in frame_idx_list}
        
        for i, res_det in enumerate(res_det_batch):
            current_frame_idx = frame_idx_list[i]
            if len(res_det["boxes"]) > 0:
                pose_image_batch.append(pil_image_batch[i])
                boxes_coco = res_det["boxes"].cpu().numpy().copy()
                boxes_coco[:, 2] -= boxes_coco[:, 0]; boxes_coco[:, 3] -= boxes_coco[:, 1]
                pose_bbox_batch.append(boxes_coco)
                pose_frame_map[len(pose_image_batch)-1] = current_frame_idx
                
                frame_bboxes = [torch.tensor([*p_box.tolist(), score.item()], dtype=torch.float32) for p_box, score in zip(res_det["boxes"], res_det["scores"])]
                temp_player_results[current_frame_idx] = (frame_bboxes, [])

        if pose_image_batch:
            pose_batch_size = batch_sizes.get('pose', 16)
            for j in range(0, len(pose_image_batch), pose_batch_size):
                sub_images, sub_bboxes = pose_image_batch[j:j+pose_batch_size], pose_bbox_batch[j:j+pose_batch_size]
                inputs_pose = loader.processors['pose'](images=sub_images, boxes=sub_bboxes, return_tensors="pt").to(cfg.DEVICE)
                with torch.no_grad():
                    outputs_pose = loader.models['pose'](**inputs_pose)
                pose_results_batch = loader.processors['pose'].post_process_pose_estimation(outputs_pose, boxes=sub_bboxes)

                for k, pose_results in enumerate(pose_results_batch):
                    target_frame_idx = pose_frame_map[j+k]
                    frame_poses = []
                    for p_res in pose_results:
                        pose_vec = [c for kp, sc in zip(p_res['keypoints'], p_res['scores']) for c in (kp[0]/w, kp[1]/h, 2 if sc > 0.5 else 1)]
                        frame_poses.append(torch.tensor(pose_vec, dtype=torch.float32))
                    temp_player_results[target_frame_idx] = (temp_player_results[target_frame_idx][0], frame_poses)
        
        for idx in sorted(temp_player_results.keys()):
            bboxes, poses = temp_player_results[idx]
            # BBoxを正規化
            normalized_bboxes = [torch.tensor([b[0]/w, b[1]/h, b[2]/w, b[3]/h, b[4]], dtype=torch.float32) for b in bboxes]
            results["player_bboxes"].append(normalized_bboxes)
            results["player_poses"].append(poses)

        frame_idx_list.clear(); pil_image_batch.clear(); rgb_image_batch.clear()

    def process_ball_batch():
        """溜まったBallのバッチを処理する内部関数"""
        if not ball_input_batch: return
        input_tensors = torch.cat(ball_input_batch, dim=0).to(cfg.DEVICE)
        # ★★★★★ FIX 3 ★★★★★
        with torch.no_grad(), torch.autocast(device_type=cfg.DEVICE, dtype=torch.bfloat16):
            heatmaps_pred = loader.models['ball'](input_tensors)
        heatmaps = torch.sigmoid(heatmaps_pred).to(torch.float32).cpu().numpy()

        for i, heatmap in enumerate(heatmaps):
            pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y_b, x_b = pos
            max_prob = np.max(heatmap)
            target_idx = ball_target_frame_indices[i]
            results["ball_feats"][target_idx] = torch.tensor([x_b / cfg.IMG_WIDTH, y_b / cfg.IMG_HEIGHT, max_prob], dtype=torch.float32)
        ball_input_batch.clear(); ball_target_frame_indices.clear()

    for frame_idx in pbar:
        ret, frame = cap.read()
        if not ret: break
        results["raw_frames"].append(frame)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        frame_idx_list.append(frame_idx); pil_image_batch.append(pil_image); rgb_image_batch.append(image_rgb)
        ball_frame_queue.append(image_rgb)
        
        if len(ball_frame_queue) == cfg.BALL_QUEUE_SIZE:
            transformed = [loader.transforms['ball'](image=f)["image"] for f in ball_frame_queue]
            ball_input_batch.append(torch.cat(transformed, dim=0).unsqueeze(0))
            ball_target_frame_indices.append(frame_idx)
            
        if len(pil_image_batch) >= batch_sizes['player']: process_player_and_pose_batch()
        if len(ball_input_batch) >= batch_sizes['ball']: process_ball_batch()

    if pil_image_batch: process_player_and_pose_batch()
    if ball_input_batch: process_ball_batch()
    cap.release()
    return results
def prepare_tensors(features: Dict[str, Any], cfg: AppConfig) -> Dict[str, torch.Tensor]:
    """
    特徴量リストをバッチ化・パディングし、モデル入力用のテンソルを作成する。
    ★プレイヤーが検出されないエッジケースに対応済み★
    """
    ball_tensor = torch.stack(features["ball_feats"]).unsqueeze(0)
    court_tensor = torch.stack(features["court_feats"]).unsqueeze(0)

    max_players = max(len(p) for p in features["player_bboxes"]) if any(p for p in features["player_bboxes"]) else 0
    
    # === ★★★ 修正箇所 スタート ★★★ ===
    # max_playersが0（プレイヤー未検出）の場合の処理
    if max_players == 0:
        seq_len = len(features["player_bboxes"])
        # (B, T, N, D) の形状に合わせ、N=0 の空テンソルを作成
        player_bbox_tensor = torch.zeros((1, seq_len, 0, 5), dtype=torch.float32)
        player_pose_tensor = torch.zeros((1, seq_len, 0, cfg.POSE_DIM), dtype=torch.float32)
    else:
        padded_bbox_frames, padded_pose_frames = [], []
        for bboxes, poses in zip(features["player_bboxes"], features["player_poses"]):
            num_detected = len(bboxes)
            bbox_t = torch.stack(bboxes) if num_detected > 0 else torch.zeros((0, 5), dtype=torch.float32)
            pose_t = torch.stack(poses) if num_detected > 0 else torch.zeros((0, cfg.POSE_DIM), dtype=torch.float32)
            
            pad_n = max_players - num_detected
            if pad_n > 0:
                bbox_pad = torch.zeros((pad_n, 5), dtype=torch.float32)
                pose_pad = torch.zeros((pad_n, cfg.POSE_DIM), dtype=torch.float32)
                bbox_t = torch.cat([bbox_t, bbox_pad], dim=0)
                pose_t = torch.cat([pose_t, pose_pad], dim=0)
            padded_bbox_frames.append(bbox_t)
            padded_pose_frames.append(pose_t)

        player_bbox_tensor = torch.stack(padded_bbox_frames).unsqueeze(0)
        player_pose_tensor = torch.stack(padded_pose_frames).unsqueeze(0)
    # === ★★★ 修正箇所 エンド ★★★ ===

    return {
        "ball": ball_tensor, "court": court_tensor,
        "player_bbox": player_bbox_tensor, "player_pose": player_pose_tensor
    }
def create_ecg_style_plot(event_probs: np.ndarray, threshold: float = 0.5) -> Image.Image:
    """イベント確率を滑らかな心電図風のグラフとして描画する。"""
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.patch.set_facecolor('#121212')

        time_axis, hit_probs, bounce_probs = np.arange(event_probs.shape[0]), event_probs[:, 0], event_probs[:, 1]
        
        def plot_single_event(ax, probs, title, line_color, fill_color, peak_color):
            ax.set_facecolor('#1E1E1E')
            ax.set_title(title, fontsize=16, color='white', pad=15)
            ax.set_ylabel("Probability", fontsize=12, color='gray')
            ax.set_ylim(0, 1.05)
            for spine in ax.spines.values(): spine.set_color('gray'); spine.set_linewidth(0.5)
            ax.tick_params(axis='both', colors='gray')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#444444')
            
            if len(time_axis) > 3:
                time_axis_smooth = np.linspace(time_axis.min(), time_axis.max(), len(time_axis) * 3)
                probs_smooth = make_interp_spline(time_axis, probs, k=3)(time_axis_smooth)
            else:
                time_axis_smooth, probs_smooth = time_axis, probs

            ax.plot(time_axis_smooth, probs_smooth, color=line_color, linewidth=2.5, label='Probability')
            ax.fill_between(time_axis_smooth, probs_smooth, color=fill_color, alpha=0.3)
            peak_indices = np.where(probs > threshold)[0]
            if len(peak_indices) > 0:
                ax.scatter(peak_indices, probs[peak_indices], color=peak_color, s=60, zorder=5, edgecolor='white', linewidth=1, label=f'Event Detected')
            ax.legend(loc='upper right', frameon=True, facecolor='#333333', edgecolor='none', labelcolor='white')

        plot_single_event(ax1, hit_probs, 'Hit Event Activity', '#FF8C00', '#FF8C00', '#FFD700')
        plot_single_event(ax2, bounce_probs, 'Bounce Event Activity', '#00BFFF', '#00BFFF', '#7FFFD4')
        ax2.set_xlabel("Frame", fontsize=12, color='gray')
        plt.tight_layout(pad=2.0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
        buf.seek(0); plot_image = Image.open(buf)
        return plot_image
    finally:
        plt.close('all'); plt.style.use('default')

def render_output_video(features: Dict[str, Any], event_probs: np.ndarray, cfg: AppConfig, progress: gr.Progress) -> str:
    """結果をオーバーレイ描画した動画を生成する"""
    output_filename = f"output_event_detection_{int(time.time())}.mp4"
    w, h = features["original_size"]
    out_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), features["fps"], (w, h))

    for idx, frame in enumerate(progress.tqdm(features["raw_frames"], desc="Stage 5/5: Rendering video")):
        # ★★★ 描画条件を変更 ★★★
        # 確率が閾値を超えている点の間だけ線を描画する
        if idx > 0:
            prob1 = features["ball_feats"][idx-1][2]
            prob2 = features["ball_feats"][idx][2]
            
            if prob1 >= cfg.BALL_PROB_THRESHOLD and prob2 >= cfg.BALL_PROB_THRESHOLD:
                p1_norm, p2_norm = features["ball_feats"][idx-1][:2], features["ball_feats"][idx][:2]
                if all(c > 0 for c in p1_norm) and all(c > 0 for c in p2_norm): # 念の為0チェック
                    p1 = (int(p1_norm[0] * w), int(p1_norm[1] * h))
                    p2 = (int(p2_norm[0] * w), int(p2_norm[1] * h))
                    cv2.line(frame, p1, p2, (0, 255, 255), 2)
        
        # ボール自体の描画（点）
        ball_prob = features["ball_feats"][idx][2]
        if ball_prob >= cfg.BALL_PROB_THRESHOLD:
            ball_pos_norm = features["ball_feats"][idx][:2]
            ball_pos = (int(ball_pos_norm[0] * w), int(ball_pos_norm[1] * h))
            cv2.circle(frame, ball_pos, 5, (0, 255, 255), -1)

        hit_prob, bounce_prob = event_probs[idx]
        if hit_prob > cfg.EVENT_THRESHOLD: cv2.putText(frame, "HIT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
        if bounce_prob > cfg.EVENT_THRESHOLD: cv2.putText(frame, "BOUNCE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        out_writer.write(frame)
    out_writer.release()
    return output_filename
# ------------------------------------------------------------------------
# 4. メインの処理フロー (★バッチ処理＆チャンク処理対応★)
# ------------------------------------------------------------------------
CONFIG = AppConfig()
LOADER = ModelLoader(CONFIG)
try:
    LOADER.load_all()
except Exception as e:
    print(f"Stopping execution due to critical error during model loading: {e}")

def full_analysis_pipeline(video_path: str, progress=gr.Progress(track_tqdm=True)):
    """
    全てのアナリシスパイプラインを統合して実行するメイン関数。
    """
    if LOADER.load_error: raise gr.Error(LOADER.load_error)
    if not hasattr(LOADER, 'models') or not LOADER.models: raise gr.Error("Models are not loaded. Please restart the application.")
    if video_path is None: raise gr.Error("動画ファイルをアップロードしてください。")

    BATCH_SIZES = {"ball": 32, "court": 32, "player": 32, "pose": 16}

    # === ステージ1: 特徴量抽出 (バッチ処理) ===
    # Use the corrected function from above
    features = extract_features_batch(video_path, LOADER, CONFIG, BATCH_SIZES, progress)
    progress(0, desc="Stage 1.5: Smoothing ball trajectory...")
    ball_smoother = BallTrajectorySmoother(
        probability_threshold=CONFIG.BALL_PROB_THRESHOLD,
        max_speed_pixels=CONFIG.BALL_MAX_SPEED_PIXELS_PER_FRAME,
        max_gap_frames=CONFIG.BALL_MAX_GAP_FRAMES
    )
    # 元の特徴量を、平滑化された特徴量で上書きする
    features["ball_feats"] = ball_smoother.process(
        features["ball_feats"], features["original_size"]
    )

    # === ステージ2: テンソル準備 ===
    progress(0, desc="Stage 2/4: Preparing feature tensors...")
    # Make sure to use the corrected prepare_tensors if players might not be detected
    input_tensors = prepare_tensors(features, CONFIG)

    # === ステージ3: イベント検出 (チャンク処理) ===
    progress(0, desc="Stage 3/4: Detecting events (Chunking)...")
    CHUNK_SIZE, OVERLAP = 300, 30
    total_len = input_tensors['ball'].shape[1]
    all_event_probs = []
    
    chunk_pbar = progress.tqdm(range(0, total_len, CHUNK_SIZE - OVERLAP), desc="Detecting events in chunks")
    for start in chunk_pbar:
        end = min(start + CHUNK_SIZE, total_len)
        chunk_tensors = {k: t[:, start:end, ...].to(CONFIG.DEVICE) for k, t in input_tensors.items()}
        # ★★★★★ FIX 4 ★★★★★
        with torch.no_grad(), torch.autocast(device_type=CONFIG.DEVICE, dtype=torch.bfloat16):
            logits = LOADER.models['event'](
                chunk_tensors['ball'], chunk_tensors['player_bbox'],
                chunk_tensors['player_pose'], chunk_tensors['court']
            )
        probs = torch.sigmoid(logits).squeeze(0).to(torch.float32).cpu() # .squeeze(0) for batch dim

        # squeeze() can remove sequence dim if it's 1. Be safe.
        if probs.ndim == 1: probs = probs.unsqueeze(0)

        all_event_probs.append(probs[OVERLAP:] if all_event_probs else probs)
        if end == total_len: break
            
    event_probs = np.concatenate(all_event_probs, axis=0)

    # === ステージ4: 結果生成 ===
    min_len = min(len(features["raw_frames"]), len(event_probs))
    if len(event_probs) != len(features["raw_frames"]):
        print(f"Warning: Frame count mismatch. Raw: {len(features['raw_frames'])}, Probs: {len(event_probs)}. Trimming to {min_len}.")
        features["raw_frames"] = features["raw_frames"][:min_len]
        event_probs = event_probs[:min_len]

    plot_image = create_ecg_style_plot(event_probs, threshold=CONFIG.EVENT_THRESHOLD)
    
    # === ステージ5: 動画レンダリング ===
    output_video_path = render_output_video(features, event_probs, CONFIG, progress) # descは関数内で設定
    
    return plot_image, output_video_path
# ------------------------------------------------------------------------
# 5. Gradioインターフェース
# ------------------------------------------------------------------------
title = "Tennis Event Detection Demo"
description = """
**動画をアップロードすると、AIがボール、コート、プレイヤー、ポーズの4つの情報を統合し、ヒットとバウンドのイベントを検出します。**
1.  **特徴量抽出 (バッチ処理)**: 動画の全フレームから、4つの上流モデルを用いて特徴量を高速に抽出します。
2.  **イベント検出 (チャンク処理)**: 長い動画に対応するため、抽出した特徴量を小さな塊（チャンク）に分割し、Transformerモデルでイベント確率を計算します。
3.  **可視化**: イベント確率の時系列グラフと、イベント情報を描画した動画を生成します。
**注意:** 動画の長さやPCのスペックによっては処理に時間がかかる場合があります。
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

    submit_button.click(
        fn=full_analysis_pipeline,
        inputs=[video_input],
        outputs=[plot_output, video_output]
    )

if __name__ == "__main__":
    # Gradioの起動をtry-exceptで囲み、モデルロード失敗時にUIが起動しないようにする
    if not LOADER.load_error and hasattr(LOADER, 'models') and LOADER.models:
        demo.launch(share=True)
    else:
        print("Gradio application will not start due to model loading failure.")