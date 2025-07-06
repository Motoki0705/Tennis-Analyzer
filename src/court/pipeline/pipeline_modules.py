import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Any, List
from tqdm import tqdm # 進捗表示のために追加
from scipy.ndimage import maximum_filter

from src.court.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocal


class CourtPreprocessor:
    """
    コート検出モデル用の画像を前処理するクラス。
    """
    def __init__(self, input_size: Tuple[int, int] = (360, 640)):
        """
        Args:
            input_size (Tuple[int, int]): モデルの入力解像度 (height, width)。
        """
        self.input_height, self.input_width = input_size
        self._transform = A.Compose([
            A.Resize(height=self.input_height, width=self.input_width),
            A.Normalize(),
            ToTensorV2(),
        ])

    def process_batch(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        複数の画像フレームのリストを前処理し、バッチテンソルを作成する。

        Args:
            frames (List[np.ndarray]): フレームのリスト。各要素は (H, W, C) 形式のRGB画像。

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - torch.Tensor: モデル入力用のバッチテンソル (B, C, H, W)。
                - List[Dict]: 各フレームの後処理に必要なメタデータのリスト。
        """
        batch_meta = []
        transformed_frames = []

        for frame in frames:
            original_height, original_width, _ = frame.shape
            
            transformed = self._transform(image=frame)
            input_tensor = transformed["image"]
            transformed_frames.append(input_tensor)
            
            batch_meta.append({
                "original_size": (original_height, original_width),
                "input_size": (self.input_height, self.input_width),
            })

        # 複数のテンソルをスタックしてバッチを作成 (B, C, H, W)
        batch_tensor = torch.stack(transformed_frames, dim=0)
        
        return batch_tensor, batch_meta


class CourtDetector:
    """
    コート検出モデルをロードし、推論を実行するクラス。
    """
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        
        try:
            self.model = LitLiteTracknetFocal.load_from_checkpoint(
                checkpoint_path, map_location=device
            ).model
            self.model.to(self.device)
            self.model.eval()
            print("CourtDetector: Model weights loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the model: {e}")

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        バッチテンソルに対してヒートマップを推論する。

        Args:
            input_tensor (torch.Tensor): 前処理済みの入力テンソル (B, C, H, W)。

        Returns:
            torch.Tensor: モデルが出力した生のヒートマップのバッチ (B, K, H, W)。
        """
        input_tensor = input_tensor.to(self.device)
        heatmap_preds = self.model(input_tensor)
        return heatmap_preds


class CourtPostprocessor:
    """
    モデルの出力（ヒートマップ）を後処理し、キーポイントを抽出するクラス。
    """
    def __init__(self, num_keypoints: int = 15, peak_threshold: float = 0.1, min_distance: int = 10):
        """
        Args:
            num_keypoints (int): 検出するキーポイントの数。
            peak_threshold (float): ピーク検出の閾値。
            min_distance (int): ピーク間の最小距離（ピクセル）。
        """
        self.num_keypoints = num_keypoints
        self.peak_threshold = peak_threshold
        self.min_distance = min_distance

    def _find_peaks(self, heatmap: np.ndarray, num_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        単一のヒートマップから指定された数のピークを検出する。
        
        Args:
            heatmap (np.ndarray): 2Dヒートマップ (H, W)
            num_peaks (int): 検出するピークの数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - peaks: ピークの座標 (num_peaks, 2) [x, y]
                - scores: 各ピークのスコア (num_peaks,)
        """
        # ローカルマキシマを検出
        local_maxima = maximum_filter(heatmap, size=self.min_distance) == heatmap
        
        # 閾値以上の点のみを考慮
        above_threshold = heatmap >= self.peak_threshold
        
        # 両方の条件を満たす点を取得
        peaks_mask = local_maxima & above_threshold
        peak_coords = np.where(peaks_mask)
        
        if len(peak_coords[0]) == 0:
            # ピークが見つからない場合は、最も高い値の点を返す
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            peaks = np.array([[max_idx[1], max_idx[0]]], dtype=np.float32)  # [x, y]
            scores = np.array([heatmap[max_idx]], dtype=np.float32)
        else:
            # スコアでソートして上位num_peaks個を取得
            peak_scores = heatmap[peak_coords]
            sorted_indices = np.argsort(peak_scores)[::-1]  # 降順
            
            # 必要な数だけ取得（足りない場合は全て取得）
            num_to_take = min(num_peaks, len(sorted_indices))
            selected_indices = sorted_indices[:num_to_take]
            
            peaks = np.column_stack([
                peak_coords[1][selected_indices],  # x座標
                peak_coords[0][selected_indices]   # y座標
            ]).astype(np.float32)
            scores = peak_scores[selected_indices].astype(np.float32)
        
        # 足りない場合は低スコアのダミーピークで埋める
        if len(peaks) < num_peaks:
            remaining = num_peaks - len(peaks)
            dummy_peaks = np.zeros((remaining, 2), dtype=np.float32)
            dummy_scores = np.zeros(remaining, dtype=np.float32)
            
            peaks = np.vstack([peaks, dummy_peaks])
            scores = np.concatenate([scores, dummy_scores])
        
        return peaks, scores

    def process_batch(self, heatmap_preds: torch.Tensor, batch_meta: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
        """
        生のヒートマップのバッチから、各画像のキーポイント座標とスコアを抽出する。

        Args:
            heatmap_preds (torch.Tensor): モデルの出力テンソルのバッチ (B, 1, H, W)。
            batch_meta (List[Dict[str, Any]]): Preprocessorから渡されたメタデータのリスト。

        Returns:
            List[Dict[str, np.ndarray]]:
                各画像の検出結果（'keypoints'と'scores'）を格納した辞書のリスト。
        """
        # (B, 1, H, W) -> B x (H, W)
        heatmap_probs = torch.sigmoid(heatmap_preds).cpu().numpy()
        
        batch_results = []
        for i in range(heatmap_probs.shape[0]): # バッチサイズでループ
            # 単一のヒートマップを取得 (1, H, W) -> (H, W)
            heatmap_prob = heatmap_probs[i, 0, :, :]  # 最初のチャンネルのみ使用
            meta = batch_meta[i]
            
            original_h, original_w = meta["original_size"]
            input_h, input_w = meta["input_size"]
            
            # ピークを検出
            peaks, scores = self._find_peaks(heatmap_prob, self.num_keypoints)
            
            # 元の画像サイズに座標を変換
            keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
            for k in range(len(peaks)):
                original_x = (peaks[k, 0] / input_w) * original_w
                original_y = (peaks[k, 1] / input_h) * original_h
                keypoints[k] = [original_x, original_y]
            
            batch_results.append({"keypoints": keypoints, "scores": scores})

        return batch_results

    def visualize_heatmap(self, heatmap_pred: torch.Tensor) -> Image.Image:
        """単一のヒートマップ全体を視覚化して画像として返す（デバッグ用）"""
        # (1, 1, H, W) or (1, H, W) -> (H, W)
        if heatmap_pred.dim() == 4:
            heatmap_pred = heatmap_pred.squeeze(0).squeeze(0)
        elif heatmap_pred.dim() == 3:
            heatmap_pred = heatmap_pred.squeeze(0)
        
        heatmap_prob = torch.sigmoid(heatmap_pred).cpu().numpy()
        
        heatmap_normalized = cv2.normalize(heatmap_prob, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(heatmap_rgb)

    def visualize_peaks(self, heatmap_pred: torch.Tensor, show_peaks: bool = True) -> Image.Image:
        """
        ヒートマップと検出されたピークを視覚化する（デバッグ用）
        
        Args:
            heatmap_pred (torch.Tensor): モデルの出力テンソル
            show_peaks (bool): ピークを表示するかどうか
            
        Returns:
            Image.Image: 視覚化された画像
        """
        # ヒートマップの準備
        if heatmap_pred.dim() == 4:
            heatmap_pred = heatmap_pred.squeeze(0).squeeze(0)
        elif heatmap_pred.dim() == 3:
            heatmap_pred = heatmap_pred.squeeze(0)
        
        heatmap_prob = torch.sigmoid(heatmap_pred).cpu().numpy()
        
        # 基本的なヒートマップを作成
        heatmap_normalized = cv2.normalize(heatmap_prob, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if show_peaks:
            # ピークを検出して描画
            peaks, scores = self._find_peaks(heatmap_prob, self.num_keypoints)
            
            for i, (peak, score) in enumerate(zip(peaks, scores)):
                if score > 0:  # ダミーピークでない場合のみ描画
                    x, y = int(peak[0]), int(peak[1])
                    cv2.circle(heatmap_rgb, (x, y), 3, (255, 255, 255), -1)
                    cv2.putText(heatmap_rgb, f'{i}', (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return Image.fromarray(heatmap_rgb)
    """
    モデルの出力（ヒートマップ）を後処理し、キーポイントを抽出するクラス。
    """
    def __init__(self, num_keypoints: int = 15):
        """
        Args:
            num_keypoints (int): 検出するキーポイントの数。
        """
        self.num_keypoints = num_keypoints

    def process_batch(self, heatmap_preds: torch.Tensor, batch_meta: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
        """
        生のヒートマップのバッチから、各画像のキーポイント座標とスコアを抽出する。

        Args:
            heatmap_preds (torch.Tensor): モデルの出力テンソルのバッチ (B, K, H, W)。
            batch_meta (List[Dict[str, Any]]): Preprocessorから渡されたメタデータのリスト。

        Returns:
            List[Dict[str, np.ndarray]]:
                各画像の検出結果（'keypoints'と'scores'）を格納した辞書のリスト。
        """
        # (B, K, H, W) -> B x (K, H, W)
        heatmap_probs = torch.sigmoid(heatmap_preds).cpu().numpy()
        
        batch_results = []
        for i in range(heatmap_probs.shape[0]): # バッチサイズでループ
            heatmap_prob = heatmap_probs[i]
            meta = batch_meta[i]
            
            keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
            scores = np.zeros(self.num_keypoints, dtype=np.float32)

            original_h, original_w = meta["original_size"]
            input_h, input_w = meta["input_size"]

            for k in range(self.num_keypoints):
                hmap = heatmap_prob[k, :, :]
                peak_y, peak_x = np.unravel_index(np.argmax(hmap), hmap.shape)
                score = hmap[peak_y, peak_x]

                original_x = (peak_x / input_w) * original_w
                original_y = (peak_y / input_h) * original_h

                keypoints[k] = [original_x, original_y]
                scores[k] = score
            
            batch_results.append({"keypoints": keypoints, "scores": scores})

        return batch_results

    def visualize_heatmap(self, heatmap_pred: torch.Tensor) -> Image.Image:
        """単一のヒートマップ全体を視覚化して画像として返す（デバッグ用）"""
        # (1, K, H, W) or (K, H, W) -> (K, H, W)
        heatmap_pred = heatmap_pred.squeeze(0)
        heatmap_prob = torch.sigmoid(heatmap_pred).cpu().numpy()
        
        combined_heatmap = np.max(heatmap_prob, axis=0)
        
        heatmap_normalized = cv2.normalize(combined_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(heatmap_rgb)
