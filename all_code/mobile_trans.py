import torch
import torch.nn as nn
import torchvision.models as models

class CNNTemporalKeypointModel(nn.Module):
    def __init__(self, num_keypoints=1, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # CNN encoder (MobileNetV2 backbone)
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.cnn_backbone = mobilenet.features  # [B, 1280, H', W']

        # Adaptive Pool to [B, 1280, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.feat_dim = 1280

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feat_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head: [B, T, num_keypoints * 2]
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_keypoints * 2)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        feats = []
        for t in range(T):
            xt = x[:, :, t, :, :]           # [B, 3, H, W]
            ft = self.cnn_backbone(xt)     # [B, 1280, H', W']
            ft = self.global_pool(ft)      # [B, 1280, 1,1]
            ft = ft.view(B, -1)            # [B, 1280]
            feats.append(ft)
        feats = torch.stack(feats, dim=1)  # [B, T, 1280]

        # Transformer expects [T, B, D]
        feats = feats.permute(1, 0, 2)
        trans_out = self.transformer(feats)   # [T, B, D]
        trans_out = trans_out.permute(1, 0, 2)  # [B, T, D]

        # Predict keypoints
        kp_out = self.mlp_head(trans_out)  # [B, T, num_keypoints*2]
        return kp_out

if __name__ == '__main__':
    inputs = torch.rand(4, 3, 20, 256, 256)
    model = CNNTemporalKeypointModel()
    for i in range(30):
        with torch.no_grad():
            outputs = model(inputs)

        print(outputs.shape)