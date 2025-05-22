import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class ConvtransBnRelu(nn.Module):
    def __init__(
        self, out_size, in_channels, out_channels, kernel_size=3, stride=2, padding=1
    ):
        super().__init__()
        self.out_size = out_size
        self.convtrans_bn_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.convtrans_bn_relu(x)
        return F.interpolate(x, size=self.out_size, mode="bilinear")


class UpBlock(nn.Module):
    def __init__(self, out_size, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up_conv = ConvtransBnRelu(
            out_size, in_ch, in_ch, kernel_size=3, stride=2, padding=1
        )
        self.identity_conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1),
            ConvBnRelu(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat((x, skip), dim=1)
        return self.identity_conv(x)


class SwinCourtUNet(nn.Module):
    def __init__(
        self,
        num_keypoints=15,
        final_channels=[64, 32],
        swin_model="swin_base_patch4_window7_224",
        pretrained=True,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            swin_model, pretrained=pretrained, features_only=True
        )

        channels = [
            f["num_chs"] for f in self.backbone.feature_info
        ]  # [128, 256, 512, 1024]
        hidden_channels = [c // 2 for c in reversed(channels)]

        # 順に解像度: 56x56, 28x28, 14x14, 7x7

        # アップサンプリングブロック：U-Net風
        self.up3 = UpBlock(
            out_size=(14, 14),
            in_ch=channels[3],
            skip_ch=channels[2],
            out_ch=hidden_channels[0],
        )
        self.up2 = UpBlock(
            out_size=(28, 28),
            in_ch=hidden_channels[0],
            skip_ch=channels[1],
            out_ch=hidden_channels[1],
        )
        self.up1 = UpBlock(
            out_size=(56, 56),
            in_ch=hidden_channels[1],
            skip_ch=channels[0],
            out_ch=hidden_channels[2],
        )

        # 最終アップ：224x224まで復元
        self.final_up = nn.Sequential(
            ConvtransBnRelu(
                out_size=(112, 112),
                in_channels=hidden_channels[2],
                out_channels=hidden_channels[2],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvBnRelu(
                in_channels=hidden_channels[2],
                out_channels=final_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvtransBnRelu(
                out_size=(224, 224),
                in_channels=final_channels[0],
                out_channels=final_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvBnRelu(
                in_channels=final_channels[0],
                out_channels=final_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        # 入力画像分の + 3 (hidden_channels[4] + 3)
        self.final_conv = nn.Conv2d(
            in_channels=final_channels[1] + 3,
            out_channels=num_keypoints,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        identity = x
        features = self.backbone(x)  # len=4: [B, H_i, W_i, C_i]
        # NHWC → NCHW に変換
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        x = self.up3(features[3], features[2])  # 7x7 -> 14x14
        x = self.up2(x, features[1])  # 14x14 -> 28x28
        x = self.up1(x, features[0])  # 28x28 -> 56x56

        x = self.final_up(x)  # 56x56 -> 224x224
        x = torch.cat((x, identity), dim=1)  # channel方向にコンキャット
        heatmap = self.final_conv(x)
        return heatmap


if __name__ == "__main__":
    model = SwinCourtUNet(num_keypoints=15)
    x = torch.randn(1, 3, 224, 224)  # 画像入力
    out = model(x)  # => torch.Size([1, 15, 224, 224])
    print(out.shape)
