import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Any, List
from tqdm import tqdm # 進捗表示のために追加
from scipy.ndimage import maximum_filter

from src.court.lit_module.lit_vit_unet import LitViTUNet

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
            self.model = LitViTUNet.load_from_checkpoint(
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
    モデルの出力を後処理し、キーポイントを抽出するクラス。
    2つのモードをサポート：
    - multi_channel=True: 複数のヒートマップからそれぞれ1つのキーポイントを抽出 (15->15)
    - multi_channel=False: 1つのヒートマップから複数のキーポイントを抽出 (1->15)
    """

    def __init__(self,
                 multi_channel: bool = True,
                 num_keypoints: int = 15,
                 peak_threshold: float = 0.7,
                 min_distance: int = 10):
        """
        Args:
            multi_channel (bool): True: 複数チャンネルからそれぞれ1つのキーポイントを抽出 (15->15)
                                 False: 1つのチャンネルから複数のキーポイントを抽出 (1->15)
            num_keypoints (int): 検出するキーポイントの最大数。
            peak_threshold (float): ピークとして認識するための最小スコア（0.0-1.0）。
            min_distance (int): ピーク間の最小距離（ピクセル単位）。
        """
        if not (0.0 <= peak_threshold <= 1.0):
            raise ValueError("peak_thresholdは0.0から1.0の間の値でなければなりません。")
            
        self.multi_channel = multi_channel
        self.num_keypoints = num_keypoints
        self.peak_threshold = peak_threshold
        self.min_distance = min_distance
        
        mode_description = "multi-channel mode (15->15)" if multi_channel else "single-channel mode (1->15)"
        print(f"CourtPostprocessor initialized with: {mode_description}, num_keypoints={num_keypoints}, peak_threshold={peak_threshold}, min_distance={min_distance}")

    def _find_peaks(self, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        単一の2Dヒートマップから、指定された条件で上位N個のピークを検出する。

        Args:
            heatmap (np.ndarray): 2Dヒートマップ (H, W)。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - peaks: 検出されたピークの座標 (num_keypoints, 2) [x, y]形式。
                - scores: 各ピークのスコア (num_keypoints,)。
        """
        # 局所的最大値を見つける
        local_max = maximum_filter(heatmap, size=self.min_distance, mode='constant') == heatmap
        
        # 閾値を超える点のみを対象にする
        above_threshold = heatmap > self.peak_threshold
        
        # 両方の条件を満たす点をピーク候補とする
        peaks_mask = local_max & above_threshold
        
        # ピーク候補の座標とスコアを取得
        coords_y, coords_x = np.where(peaks_mask)
        scores = heatmap[coords_y, coords_x]

        # スコアでソートし、上位N件を取得
        if len(scores) > 0:
            sorted_indices = np.argsort(scores)[::-1]  # 降順
            num_to_take = min(self.num_keypoints, len(sorted_indices))
            top_indices = sorted_indices[:num_to_take]

            peaks_x = coords_x[top_indices]
            peaks_y = coords_y[top_indices]
            
            # [x, y] の形式に整形
            peaks = np.vstack((peaks_x, peaks_y)).T
            final_scores = scores[top_indices]
        else:
            # ピークが見つからなかった場合
            peaks = np.empty((0, 2), dtype=np.float32)
            final_scores = np.empty(0, dtype=np.float32)

        # 検出数が足りない場合、ゼロパディングする
        num_found = len(final_scores)
        if num_found < self.num_keypoints:
            padding_size = self.num_keypoints - num_found
            padded_peaks = np.zeros((padding_size, 2), dtype=np.float32)
            padded_scores = np.zeros(padding_size, dtype=np.float32)
            
            peaks = np.vstack((peaks, padded_peaks)) if num_found > 0 else padded_peaks
            final_scores = np.concatenate((final_scores, padded_scores)) if num_found > 0 else padded_scores
            
        return peaks.astype(np.float32), final_scores.astype(np.float32)

    def _find_single_peak(self, heatmap: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        単一の2Dヒートマップから最も信頼度の高い1つのピークを検出する。

        Args:
            heatmap (np.ndarray): 2Dヒートマップ (H, W)。

        Returns:
            Tuple[np.ndarray, float]:
                - peak: 検出されたピークの座標 (2,) [x, y]形式。
                - score: ピークのスコア。
        """
        # 最大値の位置を取得
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        max_score = heatmap[max_idx]
        
        # 座標を [x, y] 形式に変換
        peak_x, peak_y = max_idx[1], max_idx[0]  # (row, col) -> (x, y)
        
        # 閾値チェック
        if max_score < self.peak_threshold:
            return np.array([0.0, 0.0], dtype=np.float32), 0.0
        
        return np.array([peak_x, peak_y], dtype=np.float32), float(max_score)

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
        batch_results = []
        
        if not self.multi_channel:
            # 1つのヒートマップから複数のキーポイントを抽出 (1->15)
            if heatmap_preds.shape[1] != 1:
                print(f"Warning: single-channel mode (1->15)では1チャンネルのヒートマップを想定していますが、入力は{heatmap_preds.shape[1]}チャンネルです。最初のチャンネルのみを使用します。")
            
            # ヒートマップを確率に変換
            heatmap_probs = torch.sigmoid(heatmap_preds[:, 0, :, :]).cpu().numpy()
            
            for i in range(heatmap_probs.shape[0]):
                heatmap = heatmap_probs[i]
                meta = batch_meta[i]
                
                # 複数のピークを検出
                peaks_coords, scores = self._find_peaks(heatmap)
                
                # 元の画像サイズに座標を変換
                original_h, original_w = meta["original_size"]
                input_h, input_w = meta["input_size"]
                
                scale_x = original_w / input_w
                scale_y = original_h / input_h
                
                original_keypoints = peaks_coords * np.array([scale_x, scale_y])
                original_keypoints[scores == 0] = [0, 0]
                
                batch_results.append({
                    "keypoints": original_keypoints,
                    "scores": scores
                })
        
        else:
            # 複数のヒートマップからそれぞれ1つのキーポイントを抽出 (15->15)
            expected_channels = self.num_keypoints
            actual_channels = heatmap_preds.shape[1]
            
            if actual_channels != expected_channels:
                print(f"Warning: multi-channel mode (15->15)では{expected_channels}チャンネルのヒートマップを想定していますが、入力は{actual_channels}チャンネルです。")
            
            # 利用可能なチャンネル数を決定
            num_channels = min(actual_channels, expected_channels)
            
            # ヒートマップを確率に変換
            heatmap_probs = torch.sigmoid(heatmap_preds[:, :num_channels, :, :]).cpu().numpy()
            
            for i in range(heatmap_probs.shape[0]):
                keypoints_list = []
                scores_list = []
                
                for ch in range(num_channels):
                    heatmap = heatmap_probs[i, ch]
                    peak_coord, score = self._find_single_peak(heatmap)
                    keypoints_list.append(peak_coord)
                    scores_list.append(score)
                
                # 不足分をゼロパディング
                while len(keypoints_list) < self.num_keypoints:
                    keypoints_list.append(np.array([0.0, 0.0], dtype=np.float32))
                    scores_list.append(0.0)
                
                keypoints = np.array(keypoints_list)
                scores = np.array(scores_list)
                
                # 元の画像サイズに座標を変換
                meta = batch_meta[i]
                original_h, original_w = meta["original_size"]
                input_h, input_w = meta["input_size"]
                
                scale_x = original_w / input_w
                scale_y = original_h / input_h
                
                original_keypoints = keypoints * np.array([scale_x, scale_y])
                original_keypoints[scores == 0] = [0, 0]
                
                batch_results.append({
                    "keypoints": original_keypoints,
                    "scores": scores
                })
        
        return batch_results

    def visualize_heatmap(self, heatmap_pred: torch.Tensor, channel_idx: int = 0) -> Image.Image:
        """
        指定されたチャンネルのヒートマップを視覚化して画像として返す（デバッグ用）
        
        Args:
            heatmap_pred (torch.Tensor): モデルの出力テンソル
            channel_idx (int): 表示するチャンネルのインデックス
        """
        # 指定されたチャンネルを取得
        if heatmap_pred.dim() == 4:  # (B, C, H, W)
            heatmap = heatmap_pred[0, channel_idx, :, :]
        elif heatmap_pred.dim() == 3:  # (C, H, W)
            heatmap = heatmap_pred[channel_idx, :, :]
        else:  # (H, W)
            heatmap = heatmap_pred

        heatmap_prob = torch.sigmoid(heatmap).cpu().numpy()
        
        heatmap_normalized = cv2.normalize(heatmap_prob, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(heatmap_rgb)

    def visualize_peaks(self, heatmap_pred: torch.Tensor, channel_idx: int = 0, show_peaks: bool = True) -> Image.Image:
        """
        指定されたチャンネルのヒートマップと検出されたピークを視覚化する（デバッグ用）
        
        Args:
            heatmap_pred (torch.Tensor): モデルの出力テンソル
            channel_idx (int): 表示するチャンネルのインデックス
            show_peaks (bool): ピークを表示するかどうか
        """
        # 指定されたチャンネルを取得
        if heatmap_pred.dim() == 4:  # (B, C, H, W)
            heatmap = heatmap_pred[0, channel_idx, :, :]
        elif heatmap_pred.dim() == 3:  # (C, H, W)
            heatmap = heatmap_pred[channel_idx, :, :]
        else:  # (H, W)
            heatmap = heatmap_pred

        heatmap_prob = torch.sigmoid(heatmap).cpu().numpy()
        
        # 基本的なヒートマップを作成
        heatmap_normalized = cv2.normalize(heatmap_prob, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if show_peaks:
            if self.multi_channel:
                # multi-channel mode: 指定されたチャンネルから1つのピークを検出
                peak, score = self._find_single_peak(heatmap_prob)
                if score > self.peak_threshold:
                    x, y = int(peak[0]), int(peak[1])
                    cv2.circle(heatmap_rgb, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(heatmap_rgb, (x, y), 7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(heatmap_rgb, f'Ch{channel_idx}', (x + 8, y + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(heatmap_rgb, f'Ch{channel_idx}', (x + 8, y + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                # single-channel mode: 複数のピークを検出
                peaks, scores = self._find_peaks(heatmap_prob)
                for i, (peak, score) in enumerate(zip(peaks, scores)):
                    if score > self.peak_threshold:
                        x, y = int(peak[0]), int(peak[1])
                        cv2.circle(heatmap_rgb, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(heatmap_rgb, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(heatmap_rgb, f'{i}', (x + 5, y + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(heatmap_rgb, f'{i}', (x + 5, y + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        return Image.fromarray(heatmap_rgb)