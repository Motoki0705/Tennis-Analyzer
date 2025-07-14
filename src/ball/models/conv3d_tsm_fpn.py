"""
ball_det_net.py
────────────────────────────────────────────────────────────
End-to-End  Small-Object (Ball) Detector for Video Clips
    • Conv3D Stem        : 時間情報を早期に捉える Depth-wise 3D 畳み込み
    • TSM (in-place)    : ほぼゼロコストで時系列文脈をバックボーン奥まで伝搬
    • ResNet-lite 2D    : 既存 2D Conv 重み資産を流用しつつ 5D テンソル対応
    • Bi-FPN-T          : マルチスケール特徴を時間バッチ化で効率融合
    • Shared CenterNet Head : 各フレームのヒートマップを一括出力
入出力
    in  :  x  … [B, N, 3,  H,  W]
    out :  y  … [B, N,    H,  W]   (num_classes = 1 の場合)
動作確認
    $ python ball_det_net.py   # CPU / CUDA どちらでも OK
────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# -----------------------------------------------------------
# 1.  Conv3D Stem (Depth-wise Temporal + Spatial Conv)
# -----------------------------------------------------------
class Conv3DStem(nn.Module):
    """
    入力 : [B, N, C_in, H, W]
    出力 : [B, N, C_out, H/2, W/2]  (空間ストライド=2)
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 32):
        super().__init__()
        # 時間方向の Depth-wise 3D Conv (T=3, groups=in_ch)
        self.t_conv = nn.Conv3d(
            in_ch, in_ch,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=in_ch,
            bias=False,
        )
        # 空間 Conv (1 × 3 × 3) でチャネル拡張＋ダウンサンプル
        self.s_conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),           # H,W /2
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn  = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t_conv(x)  # 時間情報保持
        x = self.s_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x            # [B,N,C,H/2,W/2]


# -----------------------------------------------------------
# 2.  Temporal Shift Module (高速 in-place 実装)
# -----------------------------------------------------------
def temporal_shift(x: torch.Tensor, fold_div: int = 8) -> torch.Tensor:
    """
    TSM : チャンネルの 1/8 を左シフト, 1/8 を右シフト, 残りは不変
    x : [B, N, C, H, W]
    return 同形状
    """
    B, N, C, H, W = x.size()
    if N == 1:
        return x  # 単フレームならスキップ

    fold = C // fold_div
    # clone はバックプロパゲーション対応のため必要
    out = x.clone()

    # ← (過去フレームから現在へ)
    out[:, 1:, :fold] = x[:, :-1, :fold]
    # → (未来フレームを現在へ)
    out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
    # 残り 6/8 はそのまま
    return out


# -----------------------------------------------------------
# 3.  Residual 2D Block (TSM 対応)
# -----------------------------------------------------------
class ResidualBlock2D(nn.Module):
    """
    入力 / 出力 : [B, N, C, H, W]
    """
    def __init__(self, c_in: int, c_out: int, stride: int = 1, use_tsm: bool = True):
        super().__init__()
        self.use_tsm = use_tsm

        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)

        self.downsample = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_tsm:
            x = temporal_shift(x)

        B, N, C, H, W = x.size()
        y = x.reshape(B * N, C, H, W)        # 2D Conv 処理

        y = self.act(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        identity = x if self.downsample is None else self.downsample(
            x.reshape(B * N, C, H, W)
        )

        y = self.act(y + identity)
        # 形状復元
        C_out, H_out, W_out = y.shape[1:]
        return y.view(B, N, C_out, H_out, W_out)


# -----------------------------------------------------------
# 4.  Bi-Directional FPN (時間をバッチに畳んで処理)
# -----------------------------------------------------------
class BiFPN(nn.Module):
    """
    入力 :  List[Tensor]  (low_res → high_res), 各 [B*N, C_i, h, w]
    出力 :  同数 List[Tensor] (同形状, 融合後)
    シンプル実装: 各パス毎に learnable weight (α>=0) で加重平均
    """
    def __init__(self, in_channels: List[int], fpn_ch: int = 128):
        super().__init__()
        self.num_levels = len(in_channels)

        # 1×1 lateral conv
        self.lat_convs = nn.ModuleList(
            nn.Conv2d(c, fpn_ch, 1, bias=False) for c in in_channels
        )
        # 3×3 fusion conv
        self.fuse_convs = nn.ModuleList(
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False)
            for _ in range(self.num_levels)
        )
        self.bns = nn.ModuleList(nn.BatchNorm2d(fpn_ch) for _ in range(self.num_levels))
        self.act = nn.SiLU(inplace=True)

        # learnable weights (top-down, bottom-up)
        self.w = nn.Parameter(torch.ones(2, self.num_levels - 1))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        P = [lat(f) for lat, f in zip(self.lat_convs, feats)]

        # ---------- top-down ----------
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.w[0, i])
            P[i] = P[i] + w / (w + 1e-4) * F.interpolate(
                P[i + 1], size=P[i].shape[-2:], mode="nearest"
            )

        # ---------- bottom-up ----------
        for i in range(1, self.num_levels):
            w = F.relu(self.w[1, i - 1])
            P[i] = P[i] + w / (w + 1e-4) * F.max_pool2d(P[i - 1], 2)

        # 3×3 conv + BN + Act
        out = []
        for p, conv, bn in zip(P, self.fuse_convs, self.bns):
            o = self.act(bn(conv(p)))
            out.append(o)
        return out  # List[Tensor]


# -----------------------------------------------------------
# 5.  Main Network
# -----------------------------------------------------------
class BallDetNet(nn.Module):
    """
    stack+all 形式のビデオクリップを一括推論する小物体検出ネット
    """
    def __init__(self, num_frames: int = 5, num_classes: int = 1):
        super().__init__()
        self.num_frames  = num_frames
        self.num_classes = num_classes

        # --- Stem ---
        self.stem = Conv3DStem(3, 32)

        # --- Backbone (ResNet-lite) ---
        self.layer1 = ResidualBlock2D(32,  64, 1)  # H/2
        self.layer2 = ResidualBlock2D(64, 128, 2)  # H/4
        self.layer3 = ResidualBlock2D(128,256, 2)  # H/8
        self.layer4 = ResidualBlock2D(256,256, 2)  # H/16

        # --- Bi-FPN ---
        self.fpn = BiFPN([64, 128, 256, 256], fpn_ch=128)

        # --- Head (CenterNet-like, shared for all frames) ---
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B,N,3,H,W] → heatmaps : [B,N,H,W]  (num_classes==1)
        """
        B, N, C, H, W = x.shape
        # 1. Stem
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stem(x)               # [B,N,32,H/2,W/2]
        x = x.permute(0, 2, 1, 3, 4)

        # 2. Backbone
        f1 = self.layer1(x)            # [B,N,64 ,H/2 ,W/2 ]
        f2 = self.layer2(f1)           # [B,N,128,H/4 ,W/4 ]
        f3 = self.layer3(f2)           # [B,N,256,H/8 ,W/8 ]
        f4 = self.layer4(f3)           # [B,N,256,H/16,W/16]

        # 3. Bi-FPN : 時間をバッチに畳む
        feats = []
        for f in (f1, f2, f3, f4):
            B_, N_, C_, H_, W_ = f.shape
            feats.append(f.view(B * N, C_, H_, W_))  # [B*N, C, h, w]

        p_feats = self.fpn(feats)      # List len=4

        # 4. Detection Head (最高解像度の P0 のみ使用)
        p0 = p_feats[0]                # [B*N,128,H/2,W/2]
        y  = self.head(p0)             # [B*N,num_classes,H/2,W/2]
        y  = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        # 5. reshape back to [B,N,...]
        y  = y.view(B, N, self.num_classes, H, W)
        if self.num_classes == 1:
            y = y.squeeze(2)           # [B,N,H,W]
        return y


# -----------------------------------------------------------
# 6.  オプション : Focal Loss (CenterNet Heatmap 用)
# -----------------------------------------------------------
class SigmoidFocalLoss(nn.Module):
    """
    CenterNet で使われるバイナリフォーカルロスの簡易版
    入力 : pred∈[0,1], target∈[0,1]  (同形状)
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, self.beta)

        pred = torch.clamp(pred.sigmoid(), eps, 1 - eps)

        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        loss = (pos_loss + neg_loss).sum()
        loss = loss / num_pos.clamp(min=1.0)
        return loss


# -----------------------------------------------------------
# 7.  動作テスト
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ネットワーク生成
    model = BallDetNet(num_frames=5, num_classes=1).to(device).eval()

    # ダミー入力 (B=2, N=5)
    x = torch.randn(2, 5, 3, 320, 640, device=device)
    y = model(x)

    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(y.shape)}")  # → (2, 5, 320, 640)

    # パラメータ数
    tot = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {tot/1e6:.2f} M")