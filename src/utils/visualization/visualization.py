# src/utils/visualization.py

import cv2
import numpy as np
import torch

def tensor_to_cv2_image(tensor: torch.Tensor) -> np.ndarray:
    """
    正規化された画像テンソル (C, H, W) をOpenCV画像 (H, W, C, BGR) に変換する。
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # チャンネルが最初に来ていることを確認 (C, H, W)
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError("Input tensor must have shape (3, H, W)")

    # 逆正規化 (albumentations.Normalizeのデフォルト値)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # テンソルでの逆正規化
    image_tensor = tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Numpyに変換し、チャンネル順を (H, W, C) に変更
    image_np = image_tensor.permute(1, 2, 0).numpy()
    
    # 0-255の整数値に変換し、RGBからBGR形式にする
    image_np = (image_np * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_bgr

def visualize_peak_valley_heatmap(
    image: np.ndarray, 
    heatmap: torch.Tensor, 
    alpha: float = 0.6
) -> np.ndarray:
    """
    単一のヒートマップ（ピークと谷を含む）を画像にオーバーレイ表示する。
    ピーク（+1に近い値）は赤、谷（-1に近い値）は青で表現する。
    
    Args:
        image (np.ndarray): ベースとなるOpenCV画像 (BGR, H, W, C)。
        heatmap (torch.Tensor): 単一チャンネルのヒートマップ (H, W)。
        alpha (float): オーバーレイの透明度。
    
    Returns:
        np.ndarray: オーバーレイされた画像。
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type np.uint8")
        
    img_copy = image.copy()
    
    if heatmap.is_cuda:
        heatmap = heatmap.cpu()
        
    # ヒートマップをNumpy配列に変換
    heatmap_np = heatmap.numpy()

    # ヒートマップの値を [-1, 1] から [0, 255] にスケーリング
    # 式: (val + 1) / 2 * 255
    heatmap_scaled = ((heatmap_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    # 発散的カラーマップ (Cool-Warm) を適用
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)

    # 元画像とヒートマップをブレンド
    overlay_image = cv2.addWeighted(img_copy, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay_image

def visualize_standard_heatmap(
    image: np.ndarray, 
    heatmap: torch.Tensor, 
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    単一のヒートマップ（[0, 1]範囲）を画像にオーバーレイ表示する。
    値が高いほど強い色（デフォルトでは赤）で表現する。
    
    Args:
        image (np.ndarray): ベースとなるOpenCV画像 (BGR, H, W, C)。
        heatmap (torch.Tensor): 単一チャンネルのヒートマップ (H, W)、値の範囲は [0, 1]。
        alpha (float): オーバーレイの透明度。
        colormap (int): OpenCVのカラーマップ（デフォルトはJET）。
    
    Returns:
        np.ndarray: オーバーレイされた画像。
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type np.uint8")
        
    img_copy = image.copy()
    
    if heatmap.is_cuda:
        heatmap = heatmap.cpu()
        
    # ヒートマップをNumpy配列に変換
    heatmap_np = heatmap.numpy()
    
    # 値の範囲を確認（デバッグ用）
    if heatmap_np.min() < 0 or heatmap_np.max() > 1:
        print(f"Warning: Heatmap values are outside [0, 1] range. Min: {heatmap_np.min()}, Max: {heatmap_np.max()}")
        # 値をクリップして [0, 1] に収める
        heatmap_np = np.clip(heatmap_np, 0, 1)
    
    # ヒートマップを画像サイズにリサイズ（必要に応じて）
    if heatmap_np.shape != img_copy.shape[:2]:
        heatmap_np = cv2.resize(heatmap_np, (img_copy.shape[1], img_copy.shape[0]))
    
    # ヒートマップの値を [0, 1] から [0, 255] にスケーリング
    heatmap_scaled = (heatmap_np * 255.0).astype(np.uint8)
    
    # カラーマップを適用
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, colormap)
    
    # 透明度を考慮してマスクを作成（値が0に近い部分は透明に）
    mask = heatmap_np > 0.1  # 閾値以下の部分は透明にする
    
    # マスクを3チャンネルに拡張
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    
    # マスクされた部分のみブレンド
    overlay_image = img_copy.copy()
    overlay_image[mask_3ch] = cv2.addWeighted(
        img_copy, 1 - alpha, heatmap_colored, alpha, 0
    )[mask_3ch]
    
    return overlay_image