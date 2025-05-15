import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from typing import Optional, Dict, Union
import torch

def visualize_dataset(img_pil, target):
    label_map = {
        0: "player",
        1: "non_player_person"
    }

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img_pil)
    ax.axis("off")

    boxes = [ann["bbox"] for ann in target["annotations"]]
    labels = [ann["category_id"] for ann in target["annotations"]]

    for (x_min, y_min, rect_w, rect_h), lbl in zip(boxes, labels):
        rect = plt.Rectangle(
            (x_min, y_min), rect_w, rect_h,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min,
            label_map[int(lbl)],
            fontsize=12, color="white",
            bbox=dict(facecolor="red", alpha=0.5, pad=2)
        )

    plt.show()


def visualize_datamodule(
    img_tensor: Union[torch.Tensor, np.ndarray],
    boxes: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> None:
    """
    画像とバウンディングボックス、ラベルを可視化する。

    Args:
        img_tensor: torch.Tensor または numpy.ndarray, shape [3, H, W]。値域は [0,1] または [0,255]。
        boxes: numpy.ndarray, shape [N, 4]。正規化された [cx, cy, w, h] 形式。
        labels: numpy.ndarray, shape [N]。整数ラベルの配列。
        class_names: Optional[Dict[int, str]]。ラベルからクラス名へのマッピング。
    """
    # Tensor → NumPy 変換
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = img_tensor

    # [3, H, W] → [H, W, 3]
    img = np.transpose(img, (1, 2, 0))

    # 値域を [0,255] の uint8 に
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    H, W = img.shape[:2]

    # 描画準備
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')

    # 各ボックスを描画
    for (cx, cy, bw, bh), lbl in zip(boxes, labels):
        # 正規化 → ピクセル座標（左上起点）
        x0 = (cx - bw / 2) * W
        y0 = (cy - bh / 2) * H
        w_px = bw * W
        h_px = bh * H

        # 枠
        rect = patches.Rectangle(
            (x0, y0), w_px, h_px,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # ラベル
        text = class_names[lbl] if class_names and lbl in class_names else str(lbl)
        ax.text(
            x0, y0,
            text,
            fontsize=12,
            color='white',
            verticalalignment='top',
            bbox=dict(facecolor='red', alpha=0.5, pad=2)
        )

    plt.show()

def visualize_results(pixel_values, results):
    # ラベル ID → 名前 の対応辞書を用意
    label_map = {
        0: "player",
        1: "non_player_person"
    }

    # Tensor → NumPy 画像に変換（H, W, C）
    img = pixel_values.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")

    for score, lbl, box in zip(results["scores"], results["labels"], results["boxes"]):
        # normalized center→絶対座標に復元
        x_min = box[0]
        y_min = box[1]
        rect_w = box[2] - x_min
        rect_h = box[3] - y_min

        # バウンディングボックスの描画
        rect = plt.Rectangle(
            (x_min, y_min), rect_w, rect_h,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        # ラベル名を表示
        ax.text(
            x_min, y_min,
            f"{label_map[int(lbl)]}: score = {score:.2f}",
            fontsize=12, color="white",
            bbox=dict(facecolor="red", alpha=0.5, pad=2)
        )

    plt.show()