"""
ガウシアンヒートマップ生成関数を提供するモジュール
"""

import numpy as np
import torch

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

def draw_negative_gaussian(heatmap, center, sigma=3.0):
    """
    ヒートマップに負のガウシアン分布（谷）を描画します。

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
    # 正のガウシアンを生成
    g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)

    # torch.min を使って、生成したガウシアンの「負の値」でヒートマップを更新する
    # これにより、谷が形成される
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = torch.min(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        -g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap