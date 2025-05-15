import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """3×3 Conv + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """ConvTranspose2d で 2 倍アップサンプリング + ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv    = ConvBlock(out_ch, out_ch)
    def forward(self, x):
        x = self.up_conv(x)
        return self.conv(x)

class CourtDetectorFPN(nn.Module):
    """
    down × 3, up × 3 の UNet ライクな構造に FPN によるトップダウン融合を組み合わせたモデル
    入力:  [B, 3,  H,  W] → 出力: [B, 1,  H,  W]  （セグメンテーションマスク）
    """
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        # ─── Encoder (down × 3) ───────────────────────
        self.enc1 = ConvBlock(in_channels,    base_ch)       # [B,  64, H,   W]
        self.enc2 = ConvBlock(base_ch,        base_ch * 2)   # [B, 128, H/2, W/2]
        self.enc3 = ConvBlock(base_ch * 2,    base_ch * 4)   # [B, 256, H/4, W/4]
        self.pool = nn.MaxPool2d(2, 2)

        # ─── Bottleneck ────────────────────────────────
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)  # [B, 512, H/8, W/8]

        # ─── FPN lateral 1×1 convs ─────────────────────
        self.lat3 = nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=1)
        self.lat2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=1)
        self.lat1 = nn.Conv2d(base_ch,     base_ch * 2, kernel_size=1)

        # ─── FPN 融合後の 3×3 conv ────────────────────
        self.topconv3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.topconv2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.topconv1 = ConvBlock(base_ch * 2, base_ch)

        # ─── Decoder (up × 3) ─────────────────────────
        self.up3 = UpBlock(base_ch,     base_ch)     # [B,  64, 2H,  2W]
        self.up2 = UpBlock(base_ch,     base_ch // 2) # [B,  32, 4H,  4W]
        self.up1 = UpBlock(base_ch // 2, base_ch // 4) # [B,  16, 8H,  8W]

        # ─── 最終セグメンテーションヘッド ───────────
        self.head = nn.Conv2d(base_ch // 4, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)        # → [B,  64, H,   W]
        p1 = self.pool(c1)       # → [B,  64, H/2, W/2]
        c2 = self.enc2(p1)       # → [B, 128, H/2, W/2]
        p2 = self.pool(c2)       # → [B, 128, H/4, W/4]
        c3 = self.enc3(p2)       # → [B, 256, H/4, W/4]
        p3 = self.pool(c3)       # → [B, 256, H/8, W/8]

        # Bottleneck
        bn = self.bottleneck(p3) # → [B, 512, H/8, W/8]

        # FPN トップダウン融合
        p3_td = F.interpolate(bn, scale_factor=2, mode='nearest')  # [B,512,H/4,W/4]
        p3_td = p3_td + self.lat3(c3)
        p3_td = self.topconv3(p3_td)                              # -> [B,256,H/4,W/4]

        p2_td = F.interpolate(p3_td, scale_factor=2, mode='nearest')  # [B,256,H/2,W/2]
        p2_td = p2_td + self.lat2(c2)
        p2_td = self.topconv2(p2_td)                                # -> [B,128,H/2,W/2]

        p1_td = F.interpolate(p2_td, scale_factor=2, mode='nearest')  # [B,128,  H,  W]
        p1_td = p1_td + self.lat1(c1)
        p1_td = self.topconv1(p1_td)                                # -> [B, 64, H,   W]

        # Decoder アップサンプリング (up × 3)
        d3 = self.up3(p1_td)  # -> [B, 64, 2H,  2W]
        d2 = self.up2(d3)     # -> [B, 32, 4H,  4W]
        d1 = self.up1(d2)     # -> [B, 16, 8H,  8W]

        # 最終出力
        out = self.head(d1)   # -> [B, 1, 8H, 8W]
        # 必要に応じて元の入力サイズに補間
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

if __name__ == '__main__':
    inputs = torch.rand(3, 3, 256, 256)
    model = CourtDetectorFPN()
    with torch.no_grad():
        outputs = model(inputs)
    print(outputs.shape)