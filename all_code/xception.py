import torch
import torch.nn as nn
import time

# --- Helper Modules (変更なし) ---
class XceptionBlock(nn.Module):
    """
    Xception Block following the structure:
    Pointwise(C -> C*ratio)
    Depthwise(C*ratio -> C*ratio)
    Pointwise(C*ratio -> C) -> Residual Add -> BN -> GeLU
    """
    def __init__(self, channels, internal_channels_ratio=0.25):
        super().__init__()
        internal_channels = max(1, int(channels * internal_channels_ratio))
        self.conv1_pw_in = nn.Conv2d(channels, internal_channels, kernel_size=1, padding=0)
        self.conv2_dw = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, groups=internal_channels, bias=False)
        self.conv3_pw_out = nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)
        self.out_gelu = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.conv1_pw_in(x)
        x = self.conv2_dw(x)
        x = self.conv3_pw_out(x)
        out = x + shortcut
        out = self.out_gelu(self.out_bn(out))
        return out

class DownSample(nn.Module):
    """Downsamples spatial resolution by 2, increases channels by 2."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    """Upsamples spatial resolution by 2, decreases channels by 1/2."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        return self.conv_transpose(x)
    

class XceptionHeatmapNet(nn.Module):
    def __init__(self, in_channels=9, base_channels=32, out_channels=1,
                 num_xceptions=[1, 1, 1, 1, 1, 1]):
        """
        num_xceptions: [down1後, down2後, down3後, up1後, up2後, up3後] のxceptionの繰り返し回数
        """
        super().__init__()

        assert len(num_xceptions) == 6, "num_xceptionsは6要素で指定してください。"

        # Initial convolution
        self.in_conv_1 = nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, stride=1, padding=1)
        self.in_conv_2 = nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsampling path
        self.down1 = DownSample(base_channels, base_channels * 2)
        self.xcep_block1 = nn.Sequential(
            *[XceptionBlock(base_channels * 2) for _ in range(num_xceptions[0])]
        )

        self.down2 = DownSample(base_channels * 2, base_channels * 4)
        self.xcep_block2 = nn.Sequential(
            *[XceptionBlock(base_channels * 4) for _ in range(num_xceptions[1])]
        )

        self.down3 = DownSample(base_channels * 4, base_channels * 8)
        self.xcep_block3 = nn.Sequential(
            *[XceptionBlock(base_channels * 8) for _ in range(num_xceptions[2])]
        )

        self.up1 = UpSample(base_channels * 8, base_channels * 4)
        self.xcep_block4 = nn.Sequential(
            *[XceptionBlock(base_channels * 4) for _ in range(num_xceptions[3])]
        )

        # Upsampling path
        self.up2 = UpSample(base_channels * 4, base_channels * 2)
        self.xcep_block5 = nn.Sequential(
            *[XceptionBlock(base_channels * 2) for _ in range(num_xceptions[4])]
        )

        self.up3 = UpSample(base_channels * 2, base_channels)
        self.xcep_block6 = nn.Sequential(
            *[XceptionBlock(base_channels) for _ in range(num_xceptions[5])]
        )

        # Final convolutions to produce heatmap
        self.final_conv_1 = nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, stride=1, padding=1)
        self.final_conv_2 = nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)

        # residual parameta
        self.x_0_param = nn.Parameter(torch.zeros(1))
        self.x_2_param = nn.Parameter(torch.zeros(1))
        self.x_4_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Initial conv
        x = self.in_conv_1(x)
        x_0 = self.in_conv_2(x)

        # Downsampling path
        x = self.down1(x_0)
        x_2 = self.xcep_block1(x)

        x = self.down2(x_2)
        x_4 = self.xcep_block2(x)

        x = self.down3(x_4)
        x = self.xcep_block3(x)

        x = self.up1(x)
        x = self.xcep_block4(x)

        # Upsampling path (with residual connections)
        x = self.up2(x + x_4 * self.x_4_param)
        x = self.xcep_block5(x)

        x = self.up3(x + x_2 * self.x_2_param)
        x = self.xcep_block6(x)

        # Final output layers
        x = self.final_conv_1(x + x_0 * self.x_0_param)
        out = self.final_conv_2(x)

        return out


# --- 動作確認コード ---
if __name__ == '__main__':
    sequence_frames = 1
    inputs = torch.rand(1, 3 * sequence_frames, 256, 256)

    # xception層を複数回繰り返した例 (各層を2〜3回繰り返すケース)
    num_xceptions = [1, 1, 2, 2, 1, 1]

    model = XceptionHeatmapNet(in_channels=3 * sequence_frames, base_channels=64,
                               out_channels=sequence_frames, num_xceptions=num_xceptions)
    model.eval()

    # テスト実行（スピードチェック）
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            outputs = model(inputs)
    end = time.time()

    print('出力サイズ:', outputs.shape)
    print(f'テスト完了までの所要時間: {end - start:.4f}秒')
