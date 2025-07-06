# pipeline_module_pose.py

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, List

# Hugging Face Transformersとカスタムモジュールをインポート
try:
    from src.player.lit_module.lit_rtdetr import LitRtdetr
    from transformers import AutoProcessor, RTDetrImageProcessor, VitPoseForPoseEstimation
except ImportError as e:
    raise ImportError(f"Could not import required modules. Please ensure 'transformers' is installed and 'src.player' is in the Python path. Original error: {e}")

# --- Stage 1: Player Detection Modules (from previous implementation) ---

class PlayerPreprocessor:
    """プレイヤー検出モデル用の画像を前処理するクラス。"""
    def __init__(self):
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
        print("PlayerPreprocessor: RTDetrImageProcessor loaded.")

    def process_batch(self, frames: List[np.ndarray]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """複数の画像フレームを前処理し、モデル入力用のバッチを作成する。"""
        batch_meta = []
        pil_images = []
        for frame in frames:
            original_height, original_width, _ = frame.shape
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_image)
            batch_meta.append({"original_size": (original_height, original_width)})
            
        inputs = self.processor(images=pil_images, return_tensors="pt")
        return inputs, batch_meta

class PlayerDetector:
    """プレイヤー検出モデルをロードし、推論を実行するクラス。"""
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        try:
            self.model = LitRtdetr.load_from_checkpoint(checkpoint_path, map_location=device).eval()
            self.model.to(self.device)
            print("PlayerDetector: Model weights loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the PlayerDetector model: {e}")

    @torch.no_grad()
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """バッチ入力に対して推論を実行する。"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)

class PlayerPostprocessor:
    """モデルの出力（推論結果）を後処理し、バウンディングボックス等を抽出するクラス。"""
    def __init__(self, confidence_threshold: float = 0.5):
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
        self.confidence_threshold = confidence_threshold
        print("PlayerPostprocessor: RTDetrImageProcessor loaded.")

    def process_batch(self, outputs: Any, batch_meta: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
        """モデルの出力から検出結果を抽出する。"""
        target_sizes = torch.tensor([meta["original_size"] for meta in batch_meta])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )
        batch_results = []
        for res in results:
            batch_results.append({
                'scores': res['scores'].cpu().numpy(),
                'labels': res['labels'].cpu().numpy(),
                'boxes': res['boxes'].cpu().numpy(),
            })
        return batch_results

# --- Stage 2: Pose Estimation Modules (New implementation) ---

class PosePreprocessor:
    """姿勢推定モデル用のデータを前処理するクラス。"""
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        print("PosePreprocessor: AutoProcessor for ViT-Pose loaded.")

    def process_frame(self, frame: np.ndarray, detections: Dict[str, np.ndarray]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        単一フレームとその検出結果を前処理する。
        Gradioデモのロジックに合わせ、フレーム単位で処理するヘルパー。
        """
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes_xyxy = detections['boxes']
        
        if len(boxes_xyxy) == 0:
            return None, None # 検出がない場合は何もしない

        # ViT-Poseが要求する xywh 形式に変換
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0] # width
        boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1] # height

        inputs = self.processor(images=pil_image, boxes=[boxes_xywh], return_tensors="pt")
        meta = {"boxes_for_post": [boxes_xywh]} # 後処理で必要になるボックス情報
        return inputs, meta


class PoseEstimator:
    """姿勢推定モデルをロードし、推論を実行するクラス。"""
    def __init__(self, device: torch.device):
        self.device = device
        try:
            self.model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device).eval()
            print("PoseEstimator: VitPoseForPoseEstimation model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the PoseEstimator model: {e}")
    
    @torch.no_grad()
    def predict(self, inputs: Dict[str, torch.Tensor]) -> Any:
        """入力に対して姿勢推定を実行する。"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)

class PosePostprocessor:
    """姿勢推定モデルの出力を後処理するクラス。"""
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        print("PosePostprocessor: AutoProcessor for ViT-Pose loaded.")

    def process_frame(self, outputs: Any, meta: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
        """単一フレームの推論結果を後処理する。"""
        boxes = meta["boxes_for_post"]
        pose_results = self.processor.post_process_pose_estimation(outputs, boxes=boxes)
        # 結果はリストのリストになっているので、最初の要素を取得
        return pose_results[0]