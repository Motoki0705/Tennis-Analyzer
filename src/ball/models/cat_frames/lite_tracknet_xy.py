import torch
import torch.nn as nn


# SEブロックはそのまま利用
class SE(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class DSConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, se_ratio=0.25):
        super().__init__()
        self.use_residual = (in_c == out_c) and (stride == 1)
        padding = kernel_size // 2

        self.dw = nn.Conv2d(
            in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_c)

        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.act = nn.Hardswish(inplace=True)
        self.se = SE(out_c, se_ratio) if se_ratio else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        x = self.se(x)
        return x + residual if self.use_residual else x


def make_dsconv_block(in_c, out_c, stride=1, se_ratio=0.25, repeat=3):
    layers = []
    for i in range(repeat):
        s = stride if i == 0 else 1
        inp = in_c if i == 0 else out_c
        layers.append(DSConv(inp, out_c, stride=s, se_ratio=se_ratio))
    return nn.Sequential(*layers)


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, upscale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * (upscale_factor**2), 1, bias=False)
        self.ps = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.ps(self.conv(x))))


class LiteTrackNetXY(nn.Module):
    """
    Ball位置を (x, y) ∈ [0, 1] × [0, 1] で出力する座標回帰モデル
    入力: (B, 9, H, W)
    出力: (B, 2)
    """

    def __init__(self, in_channels: int = 9):
        super().__init__()
        # Encoder
        self.enc1 = make_dsconv_block(in_channels, 16, stride=1, se_ratio=0.0, repeat=2)
        self.enc2 = make_dsconv_block(16, 32, stride=2, repeat=3)
        self.enc3 = make_dsconv_block(32, 64, stride=2, repeat=3)
        self.enc4 = make_dsconv_block(64, 128, stride=2, repeat=5)
        self.enc5 = make_dsconv_block(128, 256, stride=2, repeat=5)
        # Bottleneck
        self.bottleneck = make_dsconv_block(256, 512)

        # Global pooling + 回帰ヘッド
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),  # (B, 256)
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Sigmoid(),  # [0, 1] 正規化出力
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.bottleneck(x)
        x = self.pool(x)
        return self.head(x)  # → (B, 2)


if __name__ == "__main__":
    model = LiteTrackNetXY()
    x = torch.randn(1, 9, 360, 640)
    pred = model(x)
    print("座標予測 (正規化):", pred)  # → (B, 2)
