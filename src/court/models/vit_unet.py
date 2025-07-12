import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, List

# =============================================================================
# ViT-Unet モデル定義 (U-Net形式 with スキップコネクション)
# Vision Transformerをエンコーダとして使用し、
# スキップコネクションを持つU-Netスタイルのデコーダを組み合わせる。
# =============================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ViTUNet(nn.Module):
    """
    Vision Transformer (ViT) をエンコーダとして使用し、
    スキップコネクションを持つU-Netアーキテクチャ。
    """
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 15,
        img_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.out_channels = out_channels
        self.img_size = img_size

        # 1. Encoderのロード (timmを利用)
        # 【変更点】features_only=True で中間層の特徴マップをリストとして出力させる。
        # out_indicesでどのブロックの出力を取得するか指定する。(ViT-Baseは12ブロック)
        # これにより、U-Netのスキップコネクションに利用する特徴量を取得できる。
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=[2, 5, 8, 11], # 4つのステージから特徴を取得
        )

        # エンコーダの出力チャンネル数を取得
        encoder_channels = self.encoder.feature_info.channels()
        # -> [768, 768, 768, 768] for vit_base

        # 2. Decoderの設計 (U-Netスタイル)
        # スキップコネクションを受け取るため、各ステージを個別に定義する。
        decoder_channels = (256, 128, 64, 32)
        
        # ボトルネック部分: エンコーダの最も深い特徴マップを処理
        self.bottleneck = DoubleConv(encoder_channels[-1], decoder_channels[0])

        # デコーダの各ステージを定義
        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_channels)):
            # アップサンプリング層
            is_final_stage = i == len(decoder_channels) - 1
            up_out_channels = decoder_channels[i] if not is_final_stage else decoder_channels[-1]
            up_in_channels = decoder_channels[i-1] if i > 0 else decoder_channels[0]
            
            upsample = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)

            # 畳み込み層 (スキップコネクションと結合するためチャンネル数を調整)
            # スキップ接続はエンコーダの浅い層から来る (逆順にアクセス)
            skip_channels = encoder_channels[len(decoder_channels) - 2 - i] if not is_final_stage else 0
            conv_in_channels = up_out_channels + skip_channels
            conv = DoubleConv(conv_in_channels, up_out_channels)
            
            self.decoder_stages.append(nn.ModuleDict({
                'upsample': upsample,
                'conv': conv
            }))
        
        # 3. Headの設計
        # 最終的なヒートマップを生成する1x1畳み込み
        self.head = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=self.out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder ---
        # 【変更点】エンコーダは特徴マップのリストを返す
        # (B, C, H, W) -> List[(B, C_i, H_i, W_i)]
        # ViTの場合、解像度は全て同じ (H/16, W/16)
        encoder_features = self.encoder(x)
        
        # 深い特徴量から浅い特徴量への順に並び替える (U-Netのデコード処理のため)
        # [feat_at_layer_2, feat_at_layer_5, feat_at_layer_8, feat_at_layer_11]
        # -> [feat_at_layer_11, feat_at_layer_8, feat_at_layer_5, feat_at_layer_2]
        skips = encoder_features[::-1]
        
        # --- Decoder (with Skip Connections) ---
        
        # ボトルネック (最も深い特徴量) から開始
        d = self.bottleneck(skips[0])

        # デコーダステージをループ処理
        for i, stage in enumerate(self.decoder_stages):
            # アップサンプリング
            d = stage['upsample'](d)
            
            # 【U-Netの核心】スキップコネクションの結合
            # i=0 -> skip[1] (layer 8), i=1 -> skip[2] (layer 5), i=2 -> skip[3] (layer 2)
            # 最終ステージ (i=3)ではスキップ接続は行わない
            if i < len(skips) - 1:
                skip_connection = skips[i + 1]
                
                # 【重要】ViTの中間特徴は低解像度なので、デコーダの解像度に合わせる
                if d.shape != skip_connection.shape:
                    skip_connection = F.interpolate(
                        skip_connection, 
                        size=d.shape[2:],  # (H, W)
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # チャンネル次元で結合
                d = torch.cat((d, skip_connection), dim=1)

            # 畳み込みブロック
            d = stage['conv'](d)

        # --- Head ---
        out = self.head(d)

        return out

if __name__ == "__main__":
    # --- 1. モデルのインスタンス化 ---
    print("Initializing ViT-UNet model...")
    img_size = 224
    model = ViTUNet(
        model_name='vit_base_patch16_224',
        pretrained=False, # 事前学習済みモデルのダウンロードを避けるためFalseに設定
        in_channels=3, 
        out_channels=15,
        img_size=(img_size, img_size)
    )
    print("Model initialized successfully.")

    # モデルのパラメータ数を計算して表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    
    # --- 2. ダミー入力データの作成 ---
    batch_size = 2
    in_channels = 3
    dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"\nCreating dummy input with shape: {dummy_input.shape}")

    # --- 3. 順伝播の実行 ---
    model.eval()
    print("Performing a forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print("Forward pass completed.")

    # --- 4. 出力の確認 ---
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    expected_out_channels = 15
    assert output.shape == (batch_size, expected_out_channels, img_size, img_size), \
        f"Output shape mismatch! Expected {(batch_size, expected_out_channels, img_size, img_size)}, but got {output.shape}"
    
    print("\nTest finished successfully! The output shape is as expected.")