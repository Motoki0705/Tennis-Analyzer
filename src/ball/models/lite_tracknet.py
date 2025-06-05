import torch
import torch.nn as nn


class SE(nn.Module):
    """
    Squeeze-and-Excitation block
    """

    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        return x * scale


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution + SE + Hardswish + Residual
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        self.use_residual = (in_c == out_c) and (stride == 1)
        padding = kernel_size // 2

        # Depthwise
        self.dw = nn.Conv2d(
            in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_c)

        # Pointwise
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        # Activation
        self.act = nn.Hardswish(inplace=True)

        # SE
        self.se = SE(out_c, se_ratio) if se_ratio else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.se(x)
        if self.use_residual:
            x = x + residual  # ResNet風 skip connection
        return x


def make_dsconv_block(in_c, out_c, stride=1, se_ratio=0.25, repeat=3):
    layers = []
    for i in range(repeat):
        s = stride if i == 0 else 1
        inp = in_c if i == 0 else out_c
        layers.append(DSConv(inp, out_c, stride=s, se_ratio=se_ratio))
    return nn.Sequential(*layers)


class PixelShuffleBlock(nn.Module):
    """
    Sub-pixel upsampling block:
    - 1×1 Conv でチャンネルを (out_c * r^2) に引き上げ
    - PixelShuffle(r) で空間解像度を r 倍
    - BN + Hardswish
    """

    def __init__(self, in_c: int, out_c: int, upscale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * (upscale_factor**2), 1, bias=False)
        self.ps = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LiteTrackNet(nn.Module):
    """
    改良版 BallTracker:
      - 入力: （B, 9, H, W）
      - 出力: （B, 1, H, W） ヒートマップ
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = make_dsconv_block(in_channels, 16, stride=1, se_ratio=0.0)
        self.enc2 = make_dsconv_block(16, 32, stride=2)
        self.enc3 = make_dsconv_block(32, 64, stride=2)
        self.enc4 = make_dsconv_block(64, 128, stride=2)

        # Bottleneck
        self.bottleneck = make_dsconv_block(128, 256, stride=1)

        # Decoder
        self.dec3_up = PixelShuffleBlock(256 + 128, 128, upscale_factor=2)
        self.dec3_conv = make_dsconv_block(128, 128, stride=1)

        self.dec2_up = PixelShuffleBlock(128 + 64, 64, upscale_factor=2)
        self.dec2_conv = make_dsconv_block(64, 64, stride=1)

        self.dec1_up = PixelShuffleBlock(64 + 32, 32, upscale_factor=2)
        self.dec1_conv = make_dsconv_block(32, 32, stride=1)

        # Head: ヒートマップ生成
        self.head = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder ---
        s1 = self.enc1(x)  # → (B, 16, H, W)
        s2 = self.enc2(s1)  # → (B, 24, H/2, W/2)
        s3 = self.enc3(s2)  # → (B, 40, H/4, W/4)
        s4 = self.enc4(s3)  # → (B, 80, H/8, W/8)

        # --- Bottleneck ---
        b = self.bottleneck(s4)  # → (B,112,H/8,W/8)

        # --- Decoder w/ Skip ---
        # Decoder w/ Skip
        d3 = self.dec3_up(torch.cat([b, s4], dim=1))
        d3 = self.dec3_conv(d3)

        d2 = self.dec2_up(torch.cat([d3, s3], dim=1))
        d2 = self.dec2_conv(d2)

        d1 = self.dec1_up(torch.cat([d2, s2], dim=1))
        d1 = self.dec1_conv(d1)

        # --- Head ---
        out = self.head(torch.cat([d1, s1], dim=1))  # → (B, 1, H, W)
        # Sigmoid をかけてヒートマップにする場合は以下を有効化:
        # out = torch.sigmoid(out)
        
        # バッチサイズが1の場合のみsqueeze
        if out.shape[1] == 1:
            return out.squeeze(1)
        return out


if __name__ == "__main__":
    # テスト実行例
    model = LiteTrackNet(in_channels=9, out_channels=1)
    x = torch.randn(1, 9, 360, 640)
    y = model(x)
    print(f"out shape: {y.shape}")  # → torch.Size([1,1,360,640])
