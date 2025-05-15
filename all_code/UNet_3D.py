import torch
import torch.nn as nn

# --- 3D ConvBlock ---
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)

# --- 3D U-Net ---
class UNet3D(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, t_frames=3, num_keypoints=1):
        super().__init__()
        # エンコーダ
        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))   # (t,h,w) → (t/2,h/2,w/2)
        self.enc2 = ConvBlock3D(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc3 = ConvBlock3D(base_ch*2, base_ch*4)

        # デコーダ
        self.up2   = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2  = ConvBlock3D(base_ch*4, base_ch*2)
        self.up1   = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1  = ConvBlock3D(base_ch*2, base_ch)

        # 出力層：チャネル数 = フレーム数
        self.outc = nn.Conv3d(base_ch, num_keypoints, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T, H, W]
        c1 = self.enc1(x)        # [B, F,   T,   H,   W]
        p1 = self.pool1(c1)      # [B, F,   T/2, H/2, W/2]

        c2 = self.enc2(p1)       # [B, 2F,  T/2, H/2, W/2]
        p2 = self.pool2(c2)      # [B, 2F,  T/4, H/4, W/4]

        c3 = self.enc3(p2)       # [B, 4F,  T/4, H/4, W/4]  ボトルネック

        u2 = self.up2(c3)        # [B, 2F,  T/2, H/2, W/2]
        cat2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(cat2)     # [B, 2F,  T/2, H/2, W/2]

        u1 = self.up1(d2)        # [B, F,   T,   H,   W]
        cat1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(cat1)     # [B, F,   T,   H,   W]

        out = self.outc(d1)      # [B, num_keypoints, T, H, W]
        return out

if __name__ == '__main__':
    N = 3
    inputs = torch.rand(1, 3, N, 224, 224)
    model = UNet3D(in_ch=3, base_ch=32, t_frames=N)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    print(outputs.shape)