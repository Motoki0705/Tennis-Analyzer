import torch
import torch.nn as nn
from transformers import ViTModel

class ViTCourt(nn.Module):
    def __init__(self, num_keypoints=15, pretrained_model="google/vit-base-patch16-224-in21k"):
        super().__init__()
        
        # 事前学習済みViTをロード
        self.encoder = ViTModel.from_pretrained(pretrained_model)
        
        # ViTの出力次元（hidden size）
        hidden_dim = self.encoder.config.hidden_size
        
        # ViTPose風のSimple Decoder
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_dim, num_keypoints, kernel_size=3, padding=1)
        )

    def forward(self, x):
        outputs = self.encoder(pixel_values=x)
        # CLSトークンを除去
        features = outputs.last_hidden_state[:, 1:, :]  # (B, N, C)

        # パッチ数からH, Wを計算
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        
        # 正しい reshape
        features = features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        heatmaps = self.decoder(features)  # (B, num_keypoints, H*4, W*4)
        return heatmaps

if __name__ == '__main__':
    inputs = torch.rand(1, 3, 224, 224)
    model = ViTCourt()
    with torch.no_grad():
        outputs = model(inputs)
    print(outputs.shape)
