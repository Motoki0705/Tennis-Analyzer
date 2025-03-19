import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from BallTrack.src.models.base import FeatureExtractorBase

class SegFormerFeatureExtractor(FeatureExtractorBase):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        # モデルの最初のパッチ埋め込み層（チャネル数3→9）の変更
        patch_embed = self.model.segformer.encoder.patch_embeddings[0]
        old_conv = patch_embed.proj  # 元は Conv2d(3, 32, kernel_size=(7,7), stride=(4,4), padding=(3,3))
        
        # 新しい Conv2d を作成（in_channels=9）
        new_conv = nn.Conv2d(
            in_channels=9,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        
        # 既存の重みを複製して9チャネル分に対応させる
        with torch.no_grad():
            # old_conv.weight の shape: [32, 3, 7, 7]
            # new_conv.weight の shape: [32, 9, 7, 7]
            # ここでは、元の3チャネル分の重みをそれぞれ複製して配置します。
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:6] = old_conv.weight
            new_conv.weight[:, 6:9] = old_conv.weight
            
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        
        # 置き換え
        patch_embed.proj = new_conv

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits

if __name__ == "__main__":
    model = SegFormerFeatureExtractor()
    img_size = (320, 240)
    dummy_pixel_values = torch.rand(1, 9, *img_size)  # バッチサイズ1、320x240、チャネル数9
    with torch.no_grad():
        feature_map = model(dummy_pixel_values)
    print("Feature Map Shape:", feature_map.shape)
