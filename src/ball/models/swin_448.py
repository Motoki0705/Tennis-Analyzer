import timm
import torch
import torch.nn as nn


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


class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * upscale**2, 3, 1, 1)
        self.ps = nn.PixelShuffle(upscale)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.ps(self.conv(x))))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, upscale=2):
        super().__init__()
        self.up_conv = PixelShuffleUp(in_ch, in_ch, upscale=upscale)
        self.identity_conv = ConvBnRelu(
            in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat((x, skip), dim=1)
        return self.identity_conv(x)


class SwinBallUNet(nn.Module):
    def __init__(
        self,
        in_channels=9,
        out_channels=1,
        final_channels=[64, 32],
        swin_model="swin_base_patch4_window7_224",
        pretrained=True,
        img_size=448,
    ):
        super().__init__()

        self.backbone = self.init_swin(
            in_channels=in_channels,
            swin_model=swin_model,
            pretrained=pretrained,
            img_size=img_size,
        )

        channels = [
            f["num_chs"] for f in self.backbone.feature_info
        ]  # [128, 256, 512, 1024]
        hidden_channels = [c // 2 for c in reversed(channels)]

        # アップサンプリングブロック：U-Net風
        self.up_res_1 = UpBlock(
            in_ch=channels[3], skip_ch=channels[2], out_ch=hidden_channels[0], upscale=2
        )
        self.up_res_2 = UpBlock(
            in_ch=hidden_channels[0],
            skip_ch=channels[1],
            out_ch=hidden_channels[1],
            upscale=2,
        )
        self.up_res_3 = UpBlock(
            in_ch=hidden_channels[1],
            skip_ch=channels[0],
            out_ch=hidden_channels[2],
            upscale=2,
        )

        # 最終アップ：入力解像度まで復元
        self.final_up = nn.Sequential(
            PixelShuffleUp(
                in_ch=hidden_channels[2], out_ch=hidden_channels[2], upscale=2
            ),
            ConvBnRelu(
                in_channels=hidden_channels[2],
                out_channels=final_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            PixelShuffleUp(
                in_ch=final_channels[0], out_ch=final_channels[0], upscale=2
            ),
            ConvBnRelu(
                in_channels=final_channels[0],
                out_channels=final_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        # 入力画像分の + 3 (hidden_channels[4] + input_channels)
        self.final_res_conv = nn.Conv2d(
            in_channels=final_channels[1] + in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        identity = x
        features = self.backbone(x)  # len=4: [B, H_i, W_i, C_i]

        # NHWC → NCHW に変換
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        x = self.up_res_1(features[3], features[2])
        x = self.up_res_2(x, features[1])
        x = self.up_res_3(x, features[0])

        x = self.final_up(x)
        x = torch.cat((x, identity), dim=1)
        heatmap = self.final_res_conv(x)
        return heatmap

    def init_swin(
        self,
        in_channels,
        swin_model="swin_base_patch4_window7_224",
        pretrained=True,
        img_size=448,
    ):
        swin = timm.create_model(
            model_name=swin_model,
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
        )
        # オリジナルの patch_embed 重みを抽出（[out_ch, in_ch, k, k]）
        original_weight = swin.patch_embed.proj.weight.data  # (C, 3, k, k)
        new_bias = swin.patch_embed.proj.bias.data
        # 3チャネルをnum_repeat回繰り返してin_channels用に拡張
        num_repeat = in_channels // 3
        new_weight = (
            original_weight.repeat(1, num_repeat, 1, 1) / num_repeat
        )  # (C, in_ch, k, k) 正規化のために/(in_ch / 3)

        new_proj_embed = nn.Conv2d(
            in_channels=in_channels, out_channels=128, kernel_size=(4, 4), stride=(4, 4)
        )
        new_proj_embed.weight.data.copy_(new_weight)
        new_proj_embed.bias.data.copy_(new_bias)

        swin.patch_embed.proj = new_proj_embed
        return swin


if __name__ == "__main__":
    model = SwinBallUNet(in_channels=9, out_channels=3, img_size=448)
    x = torch.randn(1, 9, 448, 448)  # 画像入力
    out = model(x)
    print(out.shape)
