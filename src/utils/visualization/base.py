"""
基本的な可視化関数を提供するモジュール
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF


def tensor_to_numpy(tensor):
    """
    テンソルをNumPy配列に変換します

    Args:
        tensor: torch.Tensor または numpy.ndarray

    Returns:
        numpy.ndarray: 変換された配列
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def normalize_image(img):
    """
    画像を0-1に正規化します

    Args:
        img: numpy.ndarray または torch.Tensor

    Returns:
        同じ型で正規化された画像
    """
    if isinstance(img, torch.Tensor):
        if img.max() > 1.0 and img.dtype != torch.uint8:
            return img / 255.0
    else:  # numpy
        if img.max() > 1.0 and img.dtype != np.uint8:
            return img / 255.0
    return img


def visualize_overlay(
    image,
    heatmap,
    alpha: float = 0.5,
    cmap: str = "jet",
    channel_idx: int = None,
    figsize: tuple = (10, 10),
    save_path: str = None,
):
    """
    画像とヒートマップを重ね合わせて表示または保存します

    Args:
        image: torch.Tensor あるいは np.ndarray, shape = (C, H, W) or (H, W, C)
        heatmap: torch.Tensor あるいは np.ndarray, shape = (H, W), (1, H, W), or (N, H, W)
        alpha: float = ヒートマップの透過率 (0.0〜1.0)
        cmap: str = カラーマップ名
        channel_idx: int or None = heatmap が複数チャネルの場合、表示するチャンネル番号
        figsize: tuple = 図のサイズ(幅, 高さ)インチ
        save_path: str or None = ファイルパスを指定すると画像を保存し、表示はしません
    """
    # Tensor 判定用ヘルパー
    def is_tensor_like(x):
        return hasattr(x, "permute") and hasattr(x, "cpu") and hasattr(x, "numpy")

    # 画像を numpy に
    if is_tensor_like(image):
        img = image.permute(1, 2, 0).cpu().numpy()
    else:
        img = image
    # HWC に統一
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # デノーマライズ (albumentations Normalize の逆)
    mean = np.array([0.485, 0.456, 0.406], dtype=img.dtype)
    std = np.array([0.229, 0.224, 0.225], dtype=img.dtype)
    img = img * std + mean
    img = np.clip(img, 0.0, 1.0)

    # ヒートマップを numpy に
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

    # プロット
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


def visualize_img_with_heatmap(frames, heatmap, is_cat_frames):
    """
    フレームとヒートマップをオーバーレイして可視化します

    Args:
        frames (torch.Tensor): [B, C*T, H, W] または [B, T, C, H, W]
        heatmap (torch.Tensor): ヒートマップ
        is_cat_frames (bool): フレームが連結されているかどうか

    Returns:
        None: 画像を保存します
    """
    # 最終画像を取得
    if is_cat_frames:
        third_frame_tensor = frames[-3:, :, :]  # [3, H, W]
    else:
        third_frame_tensor = frames[-1, :, :, :]  # [3, H, W]

    # denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    third_frame_tensor = third_frame_tensor * std + mean

    third_frame_np = TF.to_pil_image(third_frame_tensor).convert("RGB")
    third_frame_np = np.array(third_frame_np)

    # --- ヒートマップをスケーリング＆カラー化 ---
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_uint8 = np.uint8(255 * (heatmap_np / np.max(heatmap_np) + 1e-8))  # 正規化
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # --- サイズ合わせ（念のため） ---
    heatmap_color = cv2.resize(
        heatmap_color, (third_frame_np.shape[1], third_frame_np.shape[0])
    )

    # --- 合成（α合成） ---
    alpha = 0.5
    overlay = cv2.addWeighted(third_frame_np, 1 - alpha, heatmap_color, alpha, 0)

    # --- 保存 ---
    cv2.imwrite("heatmap_overlay.png", overlay)
    print("Saved overlay as heatmap_overlay.png") 