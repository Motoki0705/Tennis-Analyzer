import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import collections

# 提供されたモデルのクラスをインポート
# このスクリプトと同じディレクトリに ball_det_net.py があることを想定
from src.ball.models.conv3d_tsm_fpn import BallDetNet
from src.ball.models import load_model_from_checkpoint

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class InferenceConfig:
    """
    推論設定を管理するデータクラス
    """
    video_path: Path
    checkpoint_path: Path
    output_video_path: Path
    output_csv_path: Path
    input_width: int
    input_height: int
    num_frames: int = 8  # モデルのconfigに合わせて設定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # ヒートマップの最大値がこの閾値を超えた場合のみ検出とみなす
    confidence_threshold: float = 0.5 

class BallDetector:
    """
    動画からボールを検出し、結果を出力する推論クラス
    """
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = self._load_model()
        logging.info(f"Using device: {self.config.device}")

    def _load_model(self) -> BallDetNet:
        """モデルを初期化し、学習済み重みをロードする"""
        try:
            logging.info(f"Loading model from {self.config.checkpoint_path}...")
            # モデルのnum_framesは設定と合わせる
            model = load_model_from_checkpoint(
                BallDetNet,
                self.config.checkpoint_path,
                map_location="cpu",
                prefix='model.',
                num_frames=self.config.num_frames,
                num_classes=1  # ボール検出なのでクラス数は1
            )
            model.to(self.config.device)
            model.eval()
            logging.info("Model loaded successfully.")
            return model
        except FileNotFoundError:
            logging.error(f"Checkpoint file not found at: {self.config.checkpoint_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def _preprocess_clip(self, clip_frames: List[np.ndarray]) -> torch.Tensor:
        """
        フレームのリスト（クリップ）をモデル入力用のテンソルに変換する
        """
        processed_frames = []
        for frame in clip_frames:
            # BGR -> RGB, リサイズ
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (self.config.input_width, self.config.input_height))
            # HWC -> CHW, [0, 255] -> [0.0, 1.0]
            img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float() / 255.0
            processed_frames.append(img_tensor)

        # フレーム数が足りない場合、最後のフレームでパディング
        while len(processed_frames) < self.config.num_frames:
            processed_frames.append(processed_frames[-1].clone())
            
        # [N, C, H, W] -> [1, N, C, H, W]
        clip_tensor = torch.stack(processed_frames).unsqueeze(0)
        return clip_tensor.to(self.config.device)

    @torch.no_grad()
    def _run_inference(self, clip_tensor: torch.Tensor) -> np.ndarray:
        """モデル推論を実行し、ヒートマップを返す"""
        heatmaps = self.model(clip_tensor)  # [1, N, H, W]
        return heatmaps.squeeze(0).cpu().numpy()

    def _postprocess_heatmaps(self, heatmaps: np.ndarray, original_shape: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """
        ヒートマップのバッチから座標と信頼度のリストを抽出する
        """
        orig_h, orig_w = original_shape
        input_h, input_w = self.config.input_height, self.config.input_width
        results = []

        for heatmap in heatmaps: # heatmap: [H, W]
            # 最も明るいピクセルの位置と値を取得
            confidence, flat_idx = torch.max(torch.from_numpy(heatmap).view(-1), 0)
            confidence = float(confidence.item())
            
            # 1D index -> 2D coordinate
            y, x = np.unravel_index(flat_idx.item(), heatmap.shape)

            # 元の画像の解像度に座標をスケールアップ
            orig_x = int(x * (orig_w / input_w))
            orig_y = int(y * (orig_h / input_h))
            
            results.append((orig_x, orig_y, confidence))
        return results

    def _draw_detection(self, frame: np.ndarray, x: int, y: int, conf: float) -> np.ndarray:
        """フレームに検出結果（円と信頼度）を描画する"""
        # 円を描画
        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
        # 信頼度テキストを描画
        text = f"{conf:.2f}"
        cv2.putText(frame, text, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    def run(self):
        """推論パイプラインを実行する"""
        # --- 1. 動画の読み込みと出力の準備 ---
        cap = cv2.VideoCapture(str(self.config.video_path))
        if not cap.isOpened():
            logging.error(f"Failed to open video: {self.config.video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 出力ディレクトリを作成
        self.config.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(self.config.output_video_path), fourcc, fps, (orig_w, orig_h))

        # --- 2. フレームごとの処理 ---
        frame_buffer = collections.deque(maxlen=self.config.num_frames)
        all_results_data = []
        
        try:
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                for frame_idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # オリジナルフレームをバッファに保持
                    frame_buffer.append(frame)
                    
                    # バッファが満たされたら推論を実行
                    if len(frame_buffer) == self.config.num_frames:
                        # バッファのフレーム（リストに変換）を前処理
                        clip_tensor = self._preprocess_clip(list(frame_buffer))
                        heatmaps = self._run_inference(clip_tensor)
                        coords = self._postprocess_heatmaps(heatmaps, (orig_h, orig_w))

                        # クリップの最初のフレームを処理対象とする
                        # TSMの性質上、中央のフレームが最も精度が良い可能性があるが、
                        # シンプルな実装のため、FIFOで処理する
                        processed_frame = frame_buffer.popleft()
                        x, y, conf = coords[0]
                        current_frame_idx = frame_idx - self.config.num_frames + 1

                        # 結果を記録
                        detection_info = {
                            "frame": current_frame_idx,
                            "x": x if conf > self.config.confidence_threshold else -1,
                            "y": y if conf > self.config.confidence_threshold else -1,
                            "confidence": conf
                        }
                        all_results_data.append(detection_info)

                        # 検出結果を描画して書き込み
                        if conf > self.config.confidence_threshold:
                            processed_frame = self._draw_detection(processed_frame.copy(), x, y, conf)
                        writer.write(processed_frame)
                        pbar.update(1)

            # --- 3. 残りのフレームの処理 ---
            # ループを抜けた後、バッファに残っているフレームを処理
            logging.info("Processing remaining frames...")
            remaining_frames = list(frame_buffer)
            if remaining_frames:
                clip_tensor = self._preprocess_clip(remaining_frames)
                heatmaps = self._run_inference(clip_tensor)
                coords = self._postprocess_heatmaps(heatmaps, (orig_h, orig_w))
                
                for i, frame in enumerate(remaining_frames):
                    x, y, conf = coords[i]
                    current_frame_idx = total_frames - len(remaining_frames) + i
                    detection_info = {
                        "frame": current_frame_idx,
                        "x": x if conf > self.config.confidence_threshold else -1,
                        "y": y if conf > self.config.confidence_threshold else -1,
                        "confidence": conf
                    }
                    all_results_data.append(detection_info)
                    
                    if conf > self.config.confidence_threshold:
                        frame = self._draw_detection(frame.copy(), x, y, conf)
                    writer.write(frame)
                    pbar.update(1)

        finally:
            # --- 4. リソースの解放と結果の保存 ---
            cap.release()
            writer.release()
            logging.info(f"Output video saved to: {self.config.output_video_path}")

        # CSVに保存
        df = pd.DataFrame(all_results_data)
        df.to_csv(self.config.output_csv_path, index=False)
        logging.info(f"Detection results saved to: {self.config.output_csv_path}")


if __name__ == '__main__':
    VIDEO_PATH = Path("datasets/test/video_input2.mp4")
    CHECKPOINT_PATH = Path("checkpoints/ball/conv3d_tsm_fpn/best_model.ckpt")
    
    # ダミーの動画ファイルを作成 (640x360, 5秒, 30fps)
    if not VIDEO_PATH.exists():
        logging.info("Creating a dummy video file for demo...")
        dummy_writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 360))
        for i in range(150):
            dummy_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            dummy_writer.write(dummy_frame)
        dummy_writer.release()

    # ダミーのモデルチェックポイントを作成
    if not CHECKPOINT_PATH.exists():
        logging.info("Creating a dummy model checkpoint for demo...")
        dummy_model = BallDetNet(num_frames=8, num_classes=1)
        torch.save(dummy_model.state_dict(), CHECKPOINT_PATH)
        
    # --- 推論の実行 ---
    config = InferenceConfig(
        video_path=VIDEO_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        output_video_path=Path("outputs/ball/ball_det_net_1.mp4"),
        output_csv_path=Path("outputs/ball/ball_det_net_1.csv"),
        input_width=640,
        input_height=320,
        num_frames=8,  # モデルのアーキテクチャに合わせる
    )

    try:
        detector = BallDetector(config)
        detector.run()
    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")