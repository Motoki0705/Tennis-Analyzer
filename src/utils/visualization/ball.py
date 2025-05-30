"""
ボール検出用の可視化関数を提供するモジュール
"""

import time
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image


def play_overlay_sequence(
    frames: np.ndarray, heatmaps: np.ndarray, batch_idx: int = 0, fps: float = 5.0
):
    """
    バッチ内の連続フレームとヒートマップを動画のように表示します。

    Args:
        frames: np.ndarray or torch.Tensor
            形状 [B, 3, N, H, W]
        heatmaps: np.ndarray or torch.Tensor
            形状 [B, 1, N, H, W]
        batch_idx: int
            表示するサンプルのインデックス
        fps: float
            フレームレート（1秒あたりのフレーム数）
    """
    # Tensor→numpy
    if hasattr(frames, "cpu"):
        frames = frames.cpu().numpy()
    if hasattr(heatmaps, "cpu"):
        heatmaps = heatmaps.cpu().numpy()

    B, C, N, H, W = frames.shape

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6 * H / W))
    ax.axis("off")
    im = ax.imshow(np.zeros((H, W, 3), dtype=np.float32))

    interval = 1.0 / fps
    cmap = plt.get_cmap("jet")

    for t in range(N):
        # 元画像
        img = frames[batch_idx, :, t]  # [3, H, W]
        img = np.transpose(img, (1, 2, 0))  # [H, W, 3]

        # ヒートマップ
        hm = heatmaps[batch_idx, 0, t]  # [H, W]
        hm_color = cmap(hm)[:, :, :3]  # [H, W, 3]

        # オーバーレイ（半透明）
        overlay = img * 0.6 + hm_color * 0.4
        overlay = np.clip(overlay, 0, 1)

        im.set_data(overlay)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(interval)

    plt.ioff()
    plt.show()


def play_overlay_sequence_xy(frames, coords, vis, fps=5.0, r=4):
    """
    フレームと座標点をオーバーレイして動画のように表示します

    Args:
        frames: [3,N,H,W] フレーム配列
        coords: [N,2] 正規化された(x,y)座標
        vis: [N] 可視性フラグ
        fps: フレームレート
        r: 円の半径
    """
    frames = frames.cpu().numpy() if hasattr(frames, "cpu") else frames
    coords = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
    vis = vis.cpu().numpy() if hasattr(vis, "cpu") else vis

    _, N, H, W = frames.shape
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6 * H / W))
    ax.axis("off")
    im = ax.imshow(np.zeros((H, W, 3), dtype=np.float32))
    circ = plt.Circle((0, 0), r, color="red")
    ax.add_patch(circ)

    for t in range(N):
        frame = np.transpose(frames[:, t], (1, 2, 0))
        im.set_data(frame)
        if vis[t] > 0:
            cx = coords[t, 0] * W
            cy = coords[t, 1] * H
            circ.center = (cx, cy)
            circ.set_visible(True)
        else:
            circ.set_visible(False)
        fig.canvas.draw()
        plt.pause(1.0 / fps)

    plt.ioff()
    plt.show()


def overlay_heatmaps_on_frames(
    frames: torch.Tensor,
    heatmaps: torch.Tensor,
    alpha: float = 0.5,
    cmap: str = "jet",
    output_dir: Union[str, Path] = None,
):
    """
    フレームとヒートマップをオーバーレイして可視化・保存する

    Args:
        frames: [B, C*T, H, W] or [B, T, C, H, W]
        heatmaps: [B, H, W] or [B, T, H, W]
        alpha: ヒートマップと元画像の合成比率
        cmap: matplotlib のカラーマップ
        output_dir: 保存先ディレクトリ（None の場合は保存しない）
    """
    import os

    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    is_cat = frames.dim() == 4 and frames.shape[1] % 3 == 0  # [B, C*T, H, W]
    is_stack = frames.dim() == 5  # [B, T, C, H, W]

    B = frames.shape[0]
    T = frames.shape[1] // 3 if is_cat else frames.shape[1]

    for b in range(B):
        for t in range(T):
            # 画像取得
            if is_cat:
                img = frames[b, t * 3 : (t + 1) * 3]  # [3, H, W]
            else:
                img = frames[b, t]  # [3, H, W]
            img_np = to_pil_image(img.cpu()).convert("RGB")
            img_np = np.array(img_np)

            # ヒートマップ取得
            if heatmaps.dim() == 3:
                hm = heatmaps[b]
            else:
                hm = heatmaps[b, t]

            hm_np = hm.cpu().numpy()
            hm_np = cv2.resize(hm_np, (img_np.shape[1], img_np.shape[0]))
            hm_colored = cv2.applyColorMap(
                (hm_np * 255).astype(np.uint8), getattr(cv2, f"COLORMAP_{cmap.upper()}")
            )

            overlay = cv2.addWeighted(img_np, 1 - alpha, hm_colored, alpha, 0)

            if output_dir:
                out_path = Path(output_dir) / f"b{b:02d}_t{t:02d}.png"
                cv2.imwrite(str(out_path), overlay[:, :, ::-1])  # BGR -> RGB
            else:
                plt.imshow(overlay)
                plt.axis("off")
                plt.title(f"Sample {b}, Frame {t}")
                plt.show() 