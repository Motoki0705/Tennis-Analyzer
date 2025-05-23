import numpy as np
import torch


def generate_gaussian_heatmap(
    raw_label, input_size, output_size, sigma=1.0
):
    """
    キーポイント情報からガウスヒートマップを生成（固定σ版）

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


if __name__ == "__main__":
    # 仮のラベルと画像サイズ
    raw_label = {"keypoints": [128, 300, 1]}  # 中央付近にキーポイントあり
    orig_size = (512, 512)
    output_size = (64, 64)

    heatmap = generate_gaussian_heatmap(raw_label, orig_size, output_size)
    print(heatmap.max())
    # 可視化
    import matplotlib.pyplot as plt

    plt.imshow(heatmap.squeeze(0).numpy(), cmap="hot")
    plt.title("Generated Heatmap")
    plt.show()
