import timm
import torch
import torch.nn as nn


# オリジナルの3チャネルの事前学習済みモデルをロード
pretrained_model = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=True,
    features_only=True,
    img_size=448,
    in_chans=3
)

# オリジナルの patch_embed 重みを抽出（[out_ch, in_ch, k, k]）
original_weight = pretrained_model.patch_embed.proj.weight.data  # (C, 3, k, k)
new_bias = pretrained_model.patch_embed.proj.bias.data
# 3チャネルを3回繰り返して9チャネル用に拡張
new_weight = original_weight.repeat(1, 3, 1, 1) / 3.0  # (C, 9, k, k) 正規化のために/3

new_proj_embed = nn.Conv2d(in_channels=9, out_channels=128, kernel_size=(4, 4), stride=(4, 4))
new_proj_embed.weight.data.copy_(new_weight)
new_proj_embed.bias.data.copy_(new_bias)

pretrained_model.patch_embed.proj = new_proj_embed
print(pretrained_model.patch_embed.proj)