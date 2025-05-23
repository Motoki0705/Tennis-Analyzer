from typing import List

import torch
import torch.nn as nn


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

    def forward(self, x):
        s0 = self.stem(x)  # (B, C, H, W)

        d1 = self.enc1(self.down1(s0))  # (B, 2C, H/2, W/2)
        d2 = self.enc2(self.down2(d1))  # (B, 4C, H/4, W/4)
        bottleneck = self.enc3(self.down3(d2))  # (B, 8C, H/8, W/8)

        u1 = self.dec1(self.up1(bottleneck) + self.alpha2 * d2)
        u2 = self.dec2(self.up2(u1) + self.alpha1 * d1)
        u3 = self.dec3(self.up3(u2) + self.alpha0 * s0)

        return self.head(u3)


class MobileNetUHeatmapWrapper(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.stem = pretrained_model.stem
        self.down1 = pretrained_model.down1
        self.enc1 = pretrained_model.enc1
        self.down2 = pretrained_model.down2
        self.enc2 = pretrained_model.enc2
        self.down3 = pretrained_model.down3
        self.enc3 = pretrained_model.enc3

        self.up1 = pretrained_model.up1
        self.dec1 = pretrained_model.dec1
        self.up2 = pretrained_model.up2
        self.dec2 = pretrained_model.dec2
        self.up3 = pretrained_model.up3
        self.dec3 = pretrained_model.dec3
        self.head = pretrained_model.head

        self.alpha0 = pretrained_model.alpha0
        self.alpha1 = pretrained_model.alpha1
        self.alpha2 = pretrained_model.alpha2

        # freeze 全層
        for p in self.parameters():
            p.requires_grad = False

    def encode(self, x: torch.Tensor):
        s0 = self.stem(x)
        d1 = self.enc1(self.down1(s0))
        d2 = self.enc2(self.down2(d1))
        bottleneck = self.enc3(self.down3(d2))
        return bottleneck, [s0, d1, d2]

    def decode(self, feats: torch.Tensor, skips: List[torch.Tensor]):
        s0, d1, d2 = skips
        u1 = self.dec1(self.up1(feats) + self.alpha2 * d2)
        u2 = self.dec2(self.up2(u1) + self.alpha1 * d1)
        u3 = self.dec3(self.up3(u2) + self.alpha0 * s0)
        return self.head(u3)


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim * 2, kernel_size, padding=padding
        )
        self.conv_h = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding
        )

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        z_r = torch.sigmoid(self.conv_zr(combined))
        z, r = torch.chunk(z_r, 2, dim=1)
        combined_r = torch.cat([x, h_prev * r], dim=1)
        h_hat = torch.tanh(self.conv_h(combined_r))
        h = (1 - z) * h_prev + z * h_hat
        return h


class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvGRUCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        h = torch.zeros(B, C, H, W, device=x.device)
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, T, C, H, W)


class TemporalHeatmapModel(nn.Module):
    def __init__(self, backbone: MobileNetUHeatmapWrapper, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.temporal = ConvGRU(input_dim=hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)

        # --- Encode 全ステップ一括 ---
        with torch.no_grad():
            bottleneck, skips = self.backbone.encode(
                x_reshaped
            )  # (B*T, c, h, w), [(B*T, ...)]
        c, h, w = bottleneck.shape[1:]

        # スキップも reshape して T軸追加
        skips_t = [s.view(B, T, *s.shape[1:]) for s in skips]

        # --- ConvGRU 時系列処理 ---
        bottleneck_t = bottleneck.view(B, T, c, h, w)
        bottleneck_out = self.temporal(bottleneck_t)  # (B, T, c, h, w)

        # --- reshape して一括 Decode ---
        bottleneck_flat = bottleneck_out.contiguous().view(B * T, c, h, w)
        skips_flat = [s.contiguous().view(B * T, *s.shape[2:]) for s in skips_t]

        with torch.no_grad():
            out = self.backbone.decode(
                bottleneck_flat, skips_flat
            )  # (B*T, out_channels, H, W)

        # (B, T, H, W) に戻す
        out = out.view(B, T, out.shape[2], out.shape[3])
        return out


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 3, 128, 128)
    pretrained = MobileNetUHeatmapNet()
    wapper = MobileNetUHeatmapWrapper(pretrained)
    model = TemporalHeatmapModel(wapper, hidden_dim=32 * 8)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    print(outputs.shape)
