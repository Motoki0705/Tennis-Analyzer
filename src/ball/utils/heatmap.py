import numpy as np
import torch

def generate_gaussian_heatmap(raw_label, input_size, output_size, base_sigma=1.5, base_size=128):
    """
    キーポイント情報からガウスヒートマップを生成（動的なσスケーリングあり）

    Args:
        raw_label: dict, "keypoints": [x, y, visibility]
        input_size: tuple, (in_H, in_W)
        output_size: tuple, (H_out, W_out)
        base_sigma: 出力サイズが base_size のときの基準 σ
        base_size: 比較用の出力基準サイズ（例: 56）

    Returns:
        torch.Tensor: [1, H_out, W_out]
    """

    heatmap = np.zeros(output_size, dtype=np.float32)

    # キーポイント存在チェック
    if "keypoints" not in raw_label or raw_label["keypoints"] is None or len(raw_label["keypoints"]) != 3:
        return torch.from_numpy(heatmap).unsqueeze(0)

    x, y, visibility = raw_label["keypoints"]
    in_H, in_W = input_size

    if visibility == 0 or x is None or y is None:
        return torch.from_numpy(heatmap).unsqueeze(0)

    # 座標スケーリング（floatで保持）
    x_scaled = x * output_size[1] / in_W
    y_scaled = y * output_size[0] / in_H

    # σ を動的に設定
    scale_factor = np.mean(output_size) / base_size
    sigma = base_sigma * scale_factor

    # 2D ガウス分布の生成
    xx, yy = np.meshgrid(np.arange(output_size[1]), np.arange(output_size[0]))
    gaussian = np.exp(-((xx - x_scaled) ** 2 + (yy - y_scaled) ** 2) / (2 * sigma ** 2))

    heatmap = np.clip(gaussian.astype(np.float32), 0, 1)

    return torch.from_numpy(heatmap) # [1, H_out, W_out]

if __name__ == '__main__':
    # 仮のラベルと画像サイズ
    raw_label = {"keypoints": [128, 300, 1]}  # 中央付近にキーポイントあり
    orig_size = (512, 512)
    output_size = (512, 512)

    heatmap = generate_gaussian_heatmap(raw_label, orig_size, output_size)

    # 可視化
    import matplotlib.pyplot as plt
    plt.imshow(heatmap.squeeze(0).numpy(), cmap='hot')
    plt.title("Generated Heatmap")
    plt.show()
