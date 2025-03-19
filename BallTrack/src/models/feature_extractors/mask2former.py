import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation
from BallTrack.src.models.base import FeatureExtractorBase

class Mask2FormerFeatureExtractor(FeatureExtractorBase):
    def __init__(
        self,
        model_name="facebook/mask2former-swin-base-ade-semantic",
    ):
        """
        Args:
            model_name (str): Hugging Face 上のセマンティックセグメンテーションモデル名
        """
        super().__init__()
        
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 入力画像テンソル ([B, 3, H, W])
        Returns:
            torch.Tensor: クラスごとの特徴マップ ([B, num_classes, H // 4, W // 4])
        """
        outputs = self.model(x)
        
        return outputs.masks_queries_logits


if __name__ == "__main__":
    # 動作確認用テストコード
    model = Mask2FormerFeatureExtractor()
    img_size = (320, 240)
    dummy_pixel_values = torch.rand(1, 3, *img_size)  # バッチサイズ1、320x240 の画像

    with torch.no_grad():
        output = model(dummy_pixel_values)
    print("heatmap:", output.shape)
