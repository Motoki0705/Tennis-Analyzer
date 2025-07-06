# pipeline_module_player.py

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, List

# Hugging Face Transformersとカスタムモジュールをインポート
# これらがインストールされている環境で実行してください
try:
    from src.player.lit_module.lit_rtdetr import LitRtdetr
    from transformers import RTDetrImageProcessor
except ImportError as e:
    raise ImportError(f"Could not import required modules. Please ensure 'transformers' is installed and 'src.player' is in the Python path. Original error: {e}")


class PlayerPreprocessor:
    """
    プレイヤー検出モデル用の画像を前処理するクラス。
    """
    def __init__(self):
        """
        RT-DETRの画像プロセッサを初期化します。
        """
        # RT-DETRの公式プロセッサをロード
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
        print("PlayerPreprocessor: RTDetrImageProcessor loaded.")

    def process_batch(self, frames: List[np.ndarray]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """
        複数の画像フレームのリストを前処理し、モデル入力用のバッチを作成する。

        Args:
            frames (List[np.ndarray]): フレームのリスト。各要素は (H, W, C) 形式のBGR画像。

        Returns:
            Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
                - Dict: モデル入力用のテンソル辞書（例: 'pixel_values', 'pixel_mask'）。
                - List[Dict]: 各フレームの後処理に必要なメタデータのリスト。
        """
        batch_meta = []
        pil_images = []

        for frame in frames:
            original_height, original_width, _ = frame.shape
            
            # OpenCVのBGRからPillowのRGBに変換
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_image)
            
            batch_meta.append({
                "original_size": (original_height, original_width),
            })
            
        # プロセッサで画像リストを一括前処理
        inputs = self.processor(images=pil_images, return_tensors="pt")
        
        return inputs, batch_meta


class PlayerDetector:
    """
    プレイヤー検出モデルをロードし、推論を実行するクラス。
    """
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        
        try:
            # PyTorch Lightningモデルをロードし、評価モードに設定
            self.model = LitRtdetr.load_from_checkpoint(
                checkpoint_path, map_location=device
            )
            self.model.to(self.device)
            self.model.eval()
            print("PlayerDetector: Model weights loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the model: {e}")

    @torch.no_grad()
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        バッチ入力に対して推論を実行する。

        Args:
            inputs (Dict[str, torch.Tensor]): 前処理済みの入力テンソル辞書。

        Returns:
            torch.Tensor: モデルが出力した生の推論結果。
        """
        # 入力テンソルを適切なデバイスに移動
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs


class PlayerPostprocessor:
    """
    モデルの出力（推論結果）を後処理し、バウンディングボックス等を抽出するクラス。
    """
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold (float): 検出結果として採用する信頼度の閾値。
        """
        # 前処理と同じプロセッサを後処理にも使用
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
        self.confidence_threshold = confidence_threshold
        print("PlayerPostprocessor: RTDetrImageProcessor loaded for post-processing.")

    def process_batch(self, outputs: Any, batch_meta: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
        """
        モデルの生の出力から、各画像の検出結果（ボックス、スコア、ラベル）を抽出する。

        Args:
            outputs (Any): PlayerDetectorからのモデル出力。
            batch_meta (List[Dict[str, Any]]): Preprocessorから渡されたメタデータのリスト。

        Returns:
            List[Dict[str, np.ndarray]]:
                各画像の検出結果（'boxes', 'scores', 'labels'）を格納した辞書のリスト。
        """
        # メタデータから元の画像サイズのテンソルを作成
        target_sizes = [meta["original_size"] for meta in batch_meta]
        target_sizes_tensor = torch.tensor(target_sizes)

        # プロセッサの後処理機能を使用して結果をデコード
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes_tensor,
            threshold=self.confidence_threshold
        )
        
        # 結果をCPU上のnumpy配列に変換して整理
        batch_results = []
        for res in results:
            batch_results.append({
                'scores': res['scores'].cpu().numpy(),
                'labels': res['labels'].cpu().numpy(),
                'boxes': res['boxes'].cpu().numpy(),
            })
            
        return batch_results