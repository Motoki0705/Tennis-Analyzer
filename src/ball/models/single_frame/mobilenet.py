from typing import List

import torch
import torch.nn as nn
import torch.quantization as quant


# ---------- 基本ユーティリティ ---------- #
def hard_swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.clamp(x + 3, 0, 6) / 6  # ≈ h-swish


class Activation(nn.Module):
    def __init__(self, act: str = "relu6"):
        super().__init__()
        if act == "relu6":
            self.fn = nn.ReLU6(inplace=True)
        elif act == "hswish":
            self.fn = hard_swish
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x):
        return self.fn(x)


# ---------- SE (MobileNet v3 スタイル) ---------- #
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, act: str = "relu6"):
        super().__init__()
        squeezed = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, squeezed, 1)
        self.act = Activation(act)
        self.conv2 = nn.Conv2d(squeezed, channels, 1)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.conv1(w))
        w = self.hsigmoid(self.conv2(w))
        return x * w


# ---------- MobileNet v2 / v3 MBConv ---------- #
class MBConv(nn.Module):
    """
    expansion -> depthwise -> projection (+SE) with optional residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        use_se: bool = False,
        act: str = "relu6",
    ):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_res = stride == 1 and in_channels == out_channels

        layers = [
            # 1) expansion (1×1 PW)
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Activation(act),
            # 2) depthwise (DW) conv
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            Activation(act),
        ]

        # 2.5) SE(optional)
        if use_se:
            layers.append(SEBlock(hidden_dim, act=act))

        # 3) projection (Linear 1×1)
        layers += [
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


# ---------- Up / Down サンプリング ---------- #
class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch, act="relu6"):
        super().__init__()
        # stride=2 depthwise separable
        self.op = nn.Sequential(
            MBConv(in_ch, out_ch, stride=2, expansion=1, act=act)  # depthwise stride 2
        )

    def forward(self, x):
        return self.op(x)


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, act="relu6"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = Activation(act)

    def forward(self, x):
        return self.act(self.bn(self.up(x)))


# ---------- MobileNet-U-HeatmapNet ---------- #
class MobileNetUHeatmapNet(nn.Module):
    """
    U-Net 風: Down3-Up3。skip は加算（× learnable α）で接続。
    repeats: [d1,d2,d3,u1,u2,u3]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 1,
        repeats: List[int] = (1, 1, 2, 2, 1, 1),
        expansion: int = 4,
        use_se: bool = True,
        act: str = "hswish",
    ):
        super().__init__()
        assert len(repeats) == 6, "repeats must be len=6"

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            Activation(act),
        )

        # ---- Encoder ---- #
        self.down1 = DownSample(base_channels, base_channels * 2, act)
        self.enc1 = nn.Sequential(
            *[
                MBConv(
                    base_channels * 2,
                    base_channels * 2,
                    stride=1,
                    expansion=expansion,
                    use_se=use_se,
                    act=act,
                )
                for _ in range(repeats[0])
            ]
        )

        self.down2 = DownSample(base_channels * 2, base_channels * 4, act)
        self.enc2 = nn.Sequential(
            *[
                MBConv(base_channels * 4, base_channels * 4, 1, expansion, use_se, act)
                for _ in range(repeats[1])
            ]
        )

        self.down3 = DownSample(base_channels * 4, base_channels * 8, act)
        self.enc3 = nn.Sequential(
            *[
                MBConv(base_channels * 8, base_channels * 8, 1, expansion, use_se, act)
                for _ in range(repeats[2])
            ]
        )

        # ---- Decoder ---- #
        self.up1 = UpSample(base_channels * 8, base_channels * 4, act)
        self.dec1 = nn.Sequential(
            *[
                MBConv(base_channels * 4, base_channels * 4, 1, expansion, use_se, act)
                for _ in range(repeats[3])
            ]
        )

        self.up2 = UpSample(base_channels * 4, base_channels * 2, act)
        self.dec2 = nn.Sequential(
            *[
                MBConv(base_channels * 2, base_channels * 2, 1, expansion, use_se, act)
                for _ in range(repeats[4])
            ]
        )

        self.up3 = UpSample(base_channels * 2, base_channels, act)
        self.dec3 = nn.Sequential(
            *[
                MBConv(base_channels, base_channels, 1, expansion, use_se, act)
                for _ in range(repeats[5])
            ]
        )

        # ---- Head ---- #
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            Activation(act),
            nn.Conv2d(base_channels // 2, out_channels, 1),
        )

        # learnable skip-scales（≒ MobileNet-EdgeTPU のアイデア）
        self.alpha0 = nn.Parameter(torch.zeros(1))
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        s0 = self.stem(x)  # (B, C, H, W)

        d1 = self.enc1(self.down1(s0))  # (B, 2C, H/2, W/2)
        d2 = self.enc2(self.down2(d1))  # (B, 4C, H/4, W/4)
        bottleneck = self.enc3(self.down3(d2))  # (B, 8C, H/8, W/8)

        u1 = self.dec1(self.up1(bottleneck) + self.alpha2 * d2)
        u2 = self.dec2(self.up2(u1) + self.alpha1 * d1)
        u3 = self.dec3(self.up3(u2) + self.alpha0 * s0)
        out = self.head(u3)
        return self.dequant(out)


# ---------- MobileNetV3Ball ---------- #
class MobileNetV3Ball(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        import torchvision.models as models
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        # 最終層を置き換え
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        
        
# ---------- 動作確認 ---------- #
if __name__ == "__main__":
    # 1️⃣ モデル構築
    model = MobileNetUHeatmapNet(
        in_channels=3,
        base_channels=32,
        out_channels=1,
        repeats=[1, 1, 2, 2, 1, 1],
        expansion=4,
        use_se=True,
        act="relu6",
    )
    # QuantStub / DeQuantStubを __init__ に追加済みの状態

    # 2️⃣ qconfig指定
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
    )
    model.qconfig = qconfig

    # 3️⃣ 量子化用に準備
    model_prepared = quant.prepare(model)

    # 4️⃣ キャリブレーション（ダミーデータでOK）
    model_prepared.eval()
    for _ in range(10):
        x = torch.randn(1, 3, 360, 640)
        with torch.no_grad():
            _ = model_prepared(x)

    # 5️⃣ 実際の量子化
    model_int8 = quant.convert(model_prepared)

    # 6️⃣ 推論してみる
    with torch.no_grad():
        out = model_int8(torch.randn(1, 3, 360, 640))
    print(out.shape)
