"""
ガウシアンヒートマップ生成関数を提供するモジュール
"""

import numpy as np
import torch


def gaussian2D(shape, sigma=1):
    """
    2Dガウシアン分布を生成します

    Args:
        shape (tuple): ガウシアン分布のサイズ (height, width)
        sigma (float): ガウシアン分布の標準偏差

    Returns:
        numpy.ndarray: 2Dガウシアン分布
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    ヒートマップにガウシアン分布を描画します（UMich実装スタイル）

    Args:
        heatmap (numpy.ndarray): 描画対象のヒートマップ
        center (tuple): ガウシアン分布の中心座標 (x, y)
        radius (int): ガウシアン分布の半径
        k (float): ガウシアン分布の強度

    Returns:
        numpy.ndarray: 更新されたヒートマップ
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    """
    バウンディングボックスに対する適切なガウシアン半径を計算します

    Args:
        det_size (tuple): バウンディングボックスのサイズ (height, width)
        min_overlap (float): 最小オーバーラップ率

    Returns:
        float: ガウシアン半径
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def generate_gaussian_heatmap(
    raw_label, input_size, output_size, sigma=3.0
):
    """
    キーポイント情報からガウスヒートマップを生成

    Args:
        raw_label (dict): "keypoints": [x, y, visibility]
        input_size (tuple): (in_H, in_W)
        output_size (tuple): (H_out, W_out)
        sigma (float): ガウス分布の標準偏差（出力解像度に対する固定値）

    Returns:
        torch.Tensor: ヒートマップテンソル（[1, H_out, W_out]）
    """
    heatmap = np.zeros(output_size, dtype=np.float32)

    # キーポイントが存在しない、または不可視である場合はゼロヒートマップを返す
    if (
        "keypoints" not in raw_label
        or raw_label["keypoints"] is None
        or len(raw_label["keypoints"]) != 3
    ):
        return torch.from_numpy(heatmap).unsqueeze(0)

    x, y, visibility = raw_label["keypoints"]
    in_H, in_W = input_size

    if visibility == 0 or x is None or y is None:
        return torch.from_numpy(heatmap).unsqueeze(0)

    # 座標を出力解像度にスケーリング
    x_scaled = x * output_size[1] / in_W
    y_scaled = y * output_size[0] / in_H

    # 2D ガウス分布の生成
    xx, yy = np.meshgrid(np.arange(output_size[1]), np.arange(output_size[0]))
    gaussian = np.exp(-((xx - x_scaled) ** 2 + (yy - y_scaled) ** 2) / (2 * sigma**2))

    heatmap = np.clip(gaussian.astype(np.float32), 0, 1)

    return torch.from_numpy(heatmap).unsqueeze(0)


def draw_gaussian(heatmap, center, sigma=3.0):
    """
    ヒートマップにガウシアン分布を描画します（PyTorch版）

    Args:
        heatmap (torch.Tensor): 描画対象のヒートマップ
        center (tuple): ガウシアン分布の中心座標 (x, y)
        sigma (float): ガウシアン分布の標準偏差

    Returns:
        torch.Tensor: 更新されたヒートマップ
    """
    tmp_size = sigma * 3
    mu_x, mu_y = center
    w, h = heatmap.shape[1], heatmap.shape[0]

    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = torch.arange(0, size, 1).float()
    y = x[:, None]
    x0 = y0 = size // 2
    g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)

    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = torch.max(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap 