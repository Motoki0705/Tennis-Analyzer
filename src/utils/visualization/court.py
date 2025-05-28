"""
コート検出用の可視化関数を提供するモジュール
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_court_overlay(
    image,
    heatmap,
    alpha: float = 0.5,
    cmap: str = "jet",
    channel_idx: int = None,
    figsize: tuple = (10, 10),
    save_path: str = None,
):
    """
    コート検出結果を可視化します。
    複数チャネルのヒートマップを処理できるようにしています。

    Args:
        image: torch.Tensor あるいは np.ndarray, shape = (C, H, W) or (H, W, C)
        heatmap: torch.Tensor あるいは np.ndarray, shape = (H, W), (1, H, W), or (N, H, W)
        alpha: float = ヒートマップの透過率 (0.0〜1.0)
        cmap: str = カラーマップ名
        channel_idx: int or None = heatmap が複数チャネルの場合、表示するチャンネル番号
        figsize: tuple = 図のサイズ(幅, 高さ)インチ
        save_path: str or None = ファイルパスを指定すると画像を保存し、表示はしません
    """
    # -------------------------
    # Tensor 判定用ヘルパー
    # -------------------------
    def is_tensor_like(x):
        return hasattr(x, "permute") and hasattr(x, "cpu") and hasattr(x, "numpy")

    # -------------------------
    # 画像を numpy に
    # -------------------------
    if is_tensor_like(image):
        img = image.permute(1, 2, 0).cpu().numpy()
    else:
        img = image
    # HWC に統一
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # -------------------------
    # デノーマライズ (albumentations Normalize の逆)
    # -------------------------
    mean = np.array([0.485, 0.456, 0.406], dtype=img.dtype)
    std = np.array([0.229, 0.224, 0.225], dtype=img.dtype)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)

    # -------------------------
    # ヒートマップを numpy に
    # -------------------------
    if is_tensor_like(heatmap):
        hm = heatmap.cpu().detach().numpy()
    else:
        hm = heatmap
    hm = np.squeeze(hm)

    # 複数チャネル対応
    if hm.ndim == 3:
        hm = hm[channel_idx] if channel_idx is not None else hm.sum(axis=0)

    # 0-1 正規化
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)

    # -------------------------
    # プロット
    # -------------------------
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.imshow(hm, alpha=alpha, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Overlay image saved to: {save_path}")
    else:
        plt.show()
    plt.close() 