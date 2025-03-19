import time
import numpy as np
import torch
import matplotlib.pyplot as plt

class DatasetVisualizer:
    def __init__(self, dataset, delay=0.5):
        """
        Args:
            dataset: torch.utils.data.Dataset のインスタンス。
            transform: 画像変換が必要な場合に指定（__getitem__ 内で既に適用されているなら不要）。
            delay: 各サンプル間の表示時間（秒）。
        """
        self.dataset = dataset
        self.delay = delay

    def visualize(self):
        """
        データセット内の各サンプルを順次表示します。
        torch.cat により連結された3枚の画像のうち中央の画像を抽出し、
        ターゲットのヒートマップから算出したボールの位置に赤い点をプロットします。
        """
        plt.ion()  # インタラクティブモード ON
        fig, ax = plt.subplots()

        num_samples = len(self.dataset)
        for idx in range(num_samples):
            # __getitem__ から取得するサンプルは (input_tensor, target) の組
            input_tensor, target = self.dataset[idx]

            # input_tensor は [prev, curr, next] の順で連結されているため、
            # チャンネル数がそれぞれ 3 であることを想定すると、
            # 現在の画像はチャネル [3:6] に相当する
            num_channels = input_tensor.shape[0] // 3
            curr_img_tensor = input_tensor[num_channels:2*num_channels, :, :]

            # tensor を numpy 配列に変換（channels-last の形式に変換）
            # ※ transform による正規化等がある場合は、必要に応じて逆正規化を実施してください
            curr_img = curr_img_tensor.detach().cpu().numpy().transpose(1, 2, 0)

            # ターゲット（ヒートマップ）は、形状が (1, H, W) または (H, W) と想定
            if isinstance(target, torch.Tensor):
                target_np = target.squeeze().detach().cpu().numpy()
            else:
                target_np = np.squeeze(target)

            # ヒートマップ中の最大値の位置をボール位置とする
            ball_y, ball_x = np.unravel_index(np.argmax(target_np), target_np.shape)

            # 画像表示の更新
            ax.clear()
            ax.imshow(curr_img)
            ax.plot(ball_x, ball_y, 'ro', markersize=8)  # 赤い点でプロット
            ax.set_title(f"Sample {idx+1}/{num_samples}")
            plt.draw()
            plt.pause(self.delay)

        plt.ioff()
        plt.show()
