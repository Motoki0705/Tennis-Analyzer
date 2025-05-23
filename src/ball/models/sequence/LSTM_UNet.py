from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    単層の ConvLSTMCell。
    入力: (B, C_in, H, W)
    隠れ状態: (B, C_hidden, H, W)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        bias: bool = True,
    ):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 入力ゲート・忘却ゲート・セル候補・出力ゲートを一度に畳み込む
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C_in, H, W]
        # state: (h_cur, c_cur), 各 [B, C_hidden, H, W]
        if state is None or state[0] is None or state[1] is None:
            B, _, H, W = x.size()
            h_cur = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c_cur = torch.zeros_like(h_cur)
        else:
            h_cur, c_cur = state

        # concat 入力と隠れ状態
        combined = torch.cat([x, h_cur], dim=1)  # [B, C_in+C_hidden, H, W]
        conv_out = self.conv(combined)  # [B, 4*C_hidden, H, W]
        cc_i, cc_f, cc_g, cc_o = torch.split(conv_out, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# 2D 用の畳み込みブロック
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2DConvLSTM(nn.Module):
    """
    フレーム毎に2D U-Net のエンコーダで特徴抽出 →
    ConvLSTM で時系列統合 → デコーダでヒートマップ再構築
    入力:  x [B, 3, T, H, W]
    出力:  [B, 1, T, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        t_frames: int = 3,
        num_keypoints: int = 1,
    ):
        super().__init__()
        self.t_frames = t_frames

        # エンコーダ
        self.enc1 = ConvBlock2D(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)  # H,W → H/2,W/2
        self.enc2 = ConvBlock2D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)  # H/2,W/2 → H/4,W/4

        # ボトルネック特徴次元
        feat_ch = base_channels * 2

        # ConvLSTM
        self.convlstm = ConvLSTMCell(
            input_dim=feat_ch, hidden_dim=feat_ch, kernel_size=(3, 3)
        )

        # デコーダ
        self.up2 = nn.ConvTranspose2d(feat_ch, base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(base_channels + feat_ch, base_channels)
        self.up1 = nn.ConvTranspose2d(
            base_channels, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock2D(base_channels * 2, base_channels)

        # 最終出力層
        self.final = nn.Conv2d(base_channels, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W] → 時間軸を先に取り出す
        B, C, T, H, W = x.shape

        # 各フレームを encoder でボトルネック特徴に
        feats = []
        for t in range(T):
            xt = x[:, :, t]  # [B,3,H,W]
            e1 = self.enc1(xt)  # [B, base, H, W]
            p1 = self.pool1(e1)  # [B, base, H/2, W/2]
            e2 = self.enc2(p1)  # [B, base*2, H/2, W/2]
            p2 = self.pool2(e2)  # [B, base*2, H/4, W/4]
            feats.append((e1, e2, p2))

        # ConvLSTM で時系列統合
        h, c = None, None
        lstm_outs = []
        for _, _, p2 in feats:
            h, c = self.convlstm(p2, (h, c))
            lstm_outs.append(h)  # [B, base*2, H/4, W/4]

        # 各タイムステップごとにデコーダを適用しヒートマップを生成
        outs = []
        for t in range(T):
            h_t = lstm_outs[t]  # [B, base*2, H/4, W/4]
            e2, e1 = feats[t][1], feats[t][0]

            u2 = self.up2(h_t)  # [B, base, H/2, W/2]
            cat2 = torch.cat([u2, e2], dim=1)  # skip 結合
            d2 = self.dec2(cat2)  # [B, base, H/2, W/2]

            u1 = self.up1(d2)  # [B, base, H, W]
            cat1 = torch.cat([u1, e1], dim=1)
            d1 = self.dec1(cat1)  # [B, base, H, W]

            out = self.final(d1)  # [B, num_keypoints, H, W]
            outs.append(out.unsqueeze(2))  # 時間軸を挿入 → [B, num_keypoints, 1, H, W]

        # 時間軸で連結 → [B, num_keypoints, T, H, W]
        return torch.cat(outs, dim=2)


if __name__ == "__main__":
    inputs = torch.rand(4, 3, 20, 256, 256)
    model = UNet2DConvLSTM(in_channels=3, base_channels=32, t_frames=3, num_keypoints=1)
    with torch.no_grad():
        outputs = model(inputs)

    print(outputs.shape)
