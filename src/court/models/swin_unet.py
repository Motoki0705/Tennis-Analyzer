import torch
import torch.nn as nn
import timm
from typing import List, Tuple

# =============================================================================
# LiteTrackNetから移植したビルディングブロック
# これらのブロックはSwinUNetのデコーダ部分で使用されます。
# =============================================================================

class SE(nn.Module):
    """
    Squeeze-and-Excitation block
    """

    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        return x * scale


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution + SE + Hardswish + Residual
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        self.use_residual = (in_c == out_c) and (stride == 1)
        padding = kernel_size // 2

        # Depthwise
        self.dw = nn.Conv2d(
            in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_c)

        # Pointwise
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        # Activation
        self.act = nn.Hardswish(inplace=True)

        # SE
        self.se = SE(out_c, se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        # 論文によってはpwの後にSEを置くケースもあるが、ここではBNの後
        x = self.se(x) 
        x = self.act(x) # 2回目のActivation
        if self.use_residual:
            x = x + residual
        return x


def make_dsconv_block(in_c, out_c, stride=1, se_ratio=0.25, repeat=3):
    layers = []
    # 最初のレイヤーでチャンネル数とストライドを調整
    layers.append(DSConv(in_c, out_c, stride=stride, se_ratio=se_ratio))
    # 残りのレイヤーはチャンネル数と解像度を維持
    for _ in range(1, repeat):
        layers.append(DSConv(out_c, out_c, stride=1, se_ratio=se_ratio))
    return nn.Sequential(*layers)


class PixelShuffleBlock(nn.Module):
    """
    Sub-pixel upsampling block:
    - 1×1 Conv でチャンネルを (out_c * r^2) に引き上げ
    - PixelShuffle(r) で空間解像度を r 倍
    - BN + Hardswish
    """

    def __init__(self, in_c: int, out_c: int, upscale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * (upscale_factor**2), 1, bias=False)
        self.ps = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ImprovedCNNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # グローバルな文脈を捉えるためのアテンション機構
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # キーポイント間の関係を学習
        self.keypoint_interaction = nn.Conv2d(
            in_channels, in_channels, 1, groups=1
        )
        
        # 最終的な出力
        self.final_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        # チャンネルアテンション
        att = self.channel_attention(x)
        x = x * att
        
        # キーポイント間の相互作用
        x = self.keypoint_interaction(x)
        
        # 最終出力
        return self.final_conv(x)
    
class TransformerHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, num_layers=2):
        super().__init__()
        self.flatten = nn.Flatten(2)  # (B, C, H*W)
        self.pos_embed = nn.Parameter(torch.randn(1, in_channels, 1))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # キーポイント専用のクエリ
        self.keypoint_queries = nn.Parameter(torch.randn(out_channels, in_channels))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(in_channels, num_heads, batch_first=True),
            num_layers
        )
        
        self.output_proj = nn.Linear(in_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 特徴マップをシーケンスに変換
        x_flat = self.flatten(x).transpose(1, 2)  # (B, H*W, C)
        
        # Transformer処理
        x_encoded = self.transformer(x_flat)
        
        # キーポイント専用のクエリを使用
        queries = self.keypoint_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, x_encoded)
        
        # 各キーポイントの位置を予測
        heatmaps = self.output_proj(decoded).squeeze(-1)  # (B, out_channels)
        
        # ヒートマップとして再構成
        return heatmaps.view(B, -1, H, W)
    

# =============================================================================
# Swin-Unet モデル定義
# =============================================================================

class SwinUNet(nn.Module):
    """
    Swin Transformerをエンコーダとして使用するU-Netモデル。
    """
    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 15,
        decoder_channels: Tuple[int, int, int, int] = (256, 128, 64, 32),
        decoder_repeats: Tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        
        # 1. Encoderのロード (timmを利用)
        self.encoder = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
            # Swin-T V2のようにout_indicesを指定できるモデルもある
            # out_indices=(0, 1, 2, 3) 
        )
        
        # エンコーダの各出力ステージのチャンネル数を取得
        encoder_channels = self.encoder.feature_info.channels()
        # Swin-Tの場合、[96, 192, 384, 768] の4要素リスト
        
        # 2. Decoderの設計
        # 最も深い層 (s3) からのアップサンプリング
        # (B, 768, H/32, W/32) -> (B, 256, H/16, W/16)
        self.up4 = PixelShuffleBlock(encoder_channels[3], decoder_channels[0])
        
        # Decoder Stage 3 (s2と結合)
        # in: (B, 384+256, H/16, W/16), out: (B, 256, H/16, W/16)
        self.dec3_conv = make_dsconv_block(encoder_channels[2] + decoder_channels[0], decoder_channels[0], repeat=decoder_repeats[0])
        # (B, 256, H/16, W/16) -> (B, 128, H/8, W/8)
        self.dec3_up = PixelShuffleBlock(decoder_channels[0], decoder_channels[1])

        # Decoder Stage 2 (s1と結合)
        # in: (B, 192+128, H/8, W/8), out: (B, 128, H/8, W/8)
        self.dec2_conv = make_dsconv_block(encoder_channels[1] + decoder_channels[1], decoder_channels[1], repeat=decoder_repeats[1])
        # (B, 128, H/8, W/8) -> (B, 64, H/4, W/4)
        self.dec2_up = PixelShuffleBlock(decoder_channels[1], decoder_channels[2])

        # Decoder Stage 1 (s0と結合)
        # in: (B, 96+64, H/4, W/4), out: (B, 64, H/4, W/4)
        self.dec1_conv = make_dsconv_block(encoder_channels[0] + decoder_channels[2], decoder_channels[2], repeat=decoder_repeats[2])
        # (B, 64, H/4, W/4) -> (B, 32, H/2, W/2)
        self.dec1_up = PixelShuffleBlock(decoder_channels[2], decoder_channels[3])
        
        # Final upsampling to original resolution
        # (B, 32, H/2, W/2) -> (B, 32, H, W)
        self.final_up = PixelShuffleBlock(decoder_channels[3], decoder_channels[3])
        
        # 3. Headの設計
        # in: (B, 32, H, W), out: (B, out_channels, H, W)
        self.head = ImprovedCNNHead(decoder_channels[3], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder ---
        # エンコーダはNHWC形式で4つの特徴マップをリストで返す
        # s0: (B, H/4, W/4, 96), s1: (B, H/8, W/8, 192)
        # s2: (B, H/16, W/16, 384), s3: (B, H/32, W/32, 768)
        s0_nhwc, s1_nhwc, s2_nhwc, s3_nhwc = self.encoder(x)

        # --- Convert to NCHW format ---
        # .permute(0, 3, 1, 2) を使って次元を入れ替える (B, H, W, C) -> (B, C, H, W)
        s0 = s0_nhwc.permute(0, 3, 1, 2).contiguous()
        s1 = s1_nhwc.permute(0, 3, 1, 2).contiguous()
        s2 = s2_nhwc.permute(0, 3, 1, 2).contiguous()
        s3 = s3_nhwc.permute(0, 3, 1, 2).contiguous()
        
        # --- Decoder with Skip Connections ---
        d4 = self.up4(s3)
        
        d3_in = torch.cat([d4, s2], 1)
        d3_conv = self.dec3_conv(d3_in)
        d3_up = self.dec3_up(d3_conv)
        
        d2_in = torch.cat([d3_up, s1], 1)
        d2_conv = self.dec2_conv(d2_in)
        d2_up = self.dec2_up(d2_conv)
        
        d1_in = torch.cat([d2_up, s0], 1)
        d1_conv = self.dec1_conv(d1_in)
        d1_up = self.dec1_up(d1_conv)

        d0 = self.final_up(d1_up)
        
        # --- Head ---
        out = self.head(d0)
        
        return out
    
if __name__ == "__main__":
    # --- 1. モデルのインスタンス化 ---
    # デフォルト設定でモデルを作成
    print("Initializing SwinUNet model...")
    # Swin-Tは224x224で学習されているため、テストもそれに合わせる
    model = SwinUNet(
        in_channels=3, 
        out_channels=15,
        decoder_channels=(256, 128, 64, 32), # デコーダのチャンネル数を調整
    )
    print("Model initialized successfully.")

    # モデルのパラメータ数を計算して表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    
    # --- 2. ダミー入力データの作成 ---
    # バッチサイズ=2, 入力チャンネル=3, 解像度=224x224 のランダムなテンソルを生成
    batch_size = 2
    in_channels = 3
    height, width = 224, 224
    dummy_input = torch.randn(batch_size, in_channels, height, width)
    print(f"\nCreating dummy input with shape: {dummy_input.shape}")

    # --- 3. 順伝播の実行 ---
    # モデルを評価モードに設定
    model.eval()
    print("Performing a forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print("Forward pass completed.")

    # --- 4. 出力の確認 ---
    # 期待される出力形状: (バッチサイズ, 出力チャンネル数, 高さ, 幅)
    # デフォルトでは (2, 15, 224, 224)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # 出力形状が期待通りか簡単なアサーションでチェック
    expected_out_channels = 15
    assert output.shape == (batch_size, expected_out_channels, height, width), \
        f"Output shape mismatch! Expected {(batch_size, expected_out_channels, height, width)}, but got {output.shape}"
    
    print("\nTest finished successfully! The output shape is as expected.")