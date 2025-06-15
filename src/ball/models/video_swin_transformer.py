import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# --- ユーザー提供の既存ブロック ---
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
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, se_ratio: float = 0.25):
        super().__init__()
        self.use_residual = (in_c == out_c) and (stride == 1)
        padding = kernel_size // 2
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.Hardswish(inplace=True)
        self.se = SE(out_c, se_ratio) if se_ratio else nn.Identity()
    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.se(x)
        if self.use_residual:
            x = x + residual
        return x

def make_dsconv_block(in_c, out_c, stride=1, se_ratio=0.25, repeat=3):
    layers = []
    for i in range(repeat):
        s = stride if i == 0 else 1
        inp = in_c if i == 0 else out_c
        layers.append(DSConv(inp, out_c, stride=s, se_ratio=se_ratio))
    return nn.Sequential(*layers)

class PixelShuffleBlock(nn.Module):
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

def window_partition(x, window_size):
    """
    (B, H, W, C) -> (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    (num_windows*B, window_size, window_size, C) -> (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """ W-MSA / SW-MSA のためのウィンドウベース自己アテンション """
    def __init__(self, dim, window_size, n_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -0.5

        # 相対位置バイアスのためのテーブルを定義
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads))

        # 相対位置インデックスを取得
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初期化の改良 - より小さな標準偏差を使用
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        # Linearレイヤーの初期化を追加
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        nn.init.trunc_normal_(self.proj.weight, std=.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
            
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 相対位置バイアスをアテンションスコアに加算
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # アテンションの安定化
        attn = torch.clamp(attn, min=1e-8)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """ W-MSAとSW-MSAを1セットにしたSwin Transformerブロック """
    def __init__(self, dim, input_resolution, n_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # epsを明示的に設定
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), n_heads=n_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() # DropPathは省略
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLPの初期化も改良
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), 
            nn.GELU(), 
            nn.Dropout(drop),  # Dropoutを追加
            nn.Linear(mlp_hidden_dim, dim), 
            nn.Dropout(drop)
        )

        # MLP用の重みも初期化
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # パディング処理
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H_pad, W_pad, _ = x.shape

        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window Partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # パディング解除処理
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SpatioTemporalSwinBlock(nn.Module):
    """
    Swin Transformerによる空間アテンションと標準的な時間アテンションを組み合わせたブロック。
    """
    def __init__(self, dim, input_resolution, n_heads, window_size=7, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()

        # 空間アテンション: Swin Transformer Blockを2つ重ねる
        self.spatial_block_1 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            n_heads=n_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop
        )

        self.spatial_block_2 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            n_heads=n_heads,
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop
        )
        
        # 時間アテンション
        self.norm_temporal = nn.LayerNorm(dim, eps=1e-6)
        self.attn_temporal = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=attn_drop,
            batch_first=True
        )
        
        # 最終FFN
        self.norm_final = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn_final = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # FFN重みの初期化
        for layer in self.ffn_final:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        x: (B, N, T, D) -> Batch, NumFrames, NumTokens, Dim
        """
        B, N, T, D = x.shape
        
        # 空間アテンション
        residual = x
        x_spatial = x.view(B * N, T, D)
        x_spatial = self.spatial_block_1(x_spatial)
        x_spatial = self.spatial_block_2(x_spatial)
        x_spatial = x_spatial.view(B, N, T, D)
        x = residual + x_spatial
        
        # 時間アテンション
        residual = x
        x_temporal = x.permute(0, 2, 1, 3).contiguous().view(B * T, N, D)
        
        # Temporal self-attention
        x_norm = self.norm_temporal(x_temporal)
        x_temporal, _ = self.attn_temporal(x_norm, x_norm, x_norm)

        x_temporal = x_temporal.view(B, T, N, D).permute(0, 2, 1, 3).contiguous()
        x = residual + x_temporal
        
        # Feed Forward
        x = x + self.ffn_final(self.norm_final(x))
        
        return x

class VideoSwinTransformer(nn.Module):
    """
    Swin Transformerベースの空間アテンションを持つ、最終的なビデオモデル。
    """
    def __init__(self,
                 img_size=(224, 384),
                 in_channels: int = 3,
                 out_channels: int = 1,
                 n_frames: int = 10,
                 window_size: int = 7,
                 feature_dim: int = 256,
                 transformer_blocks: int = 2,
                 transformer_heads: int = 8):
        super().__init__()
        
        # CNN Encoder (5段階に深化)
        self.enc1 = make_dsconv_block(in_channels, 16, stride=2)
        self.enc2 = make_dsconv_block(16, 32, stride=2)
        self.enc3 = make_dsconv_block(32, 64, stride=2)
        self.enc4 = make_dsconv_block(64, 128, stride=2)
        self.enc5 = make_dsconv_block(128, 256, stride=2)

        # Feature Embedding & Positional Encoding
        self.feat_proj = nn.Conv2d(256, feature_dim, kernel_size=1)
        
        # 時間方向の位置エンコーディングを改良
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, n_frames, 1, feature_dim))
        nn.init.trunc_normal_(self.pos_embed_temporal, std=.02)

        # Spatio-Temporal Swin Transformer
        final_resolution = (img_size[0] // 32, img_size[1] // 32)

        self.transformer_blocks = nn.Sequential(
            *[SpatioTemporalSwinBlock(
                dim=feature_dim,
                input_resolution=final_resolution,
                n_heads=transformer_heads,
                window_size=window_size,
                drop=0.1,  # Dropoutを追加
                attn_drop=0.1
              ) for _ in range(transformer_blocks)]
        )

        # CNN Decoder
        self.dec4_up = PixelShuffleBlock(feature_dim, 128, upscale_factor=2)
        self.dec4_conv = make_dsconv_block(128 + 128, 128, stride=1)
        self.dec3_up = PixelShuffleBlock(128, 64, upscale_factor=2)
        self.dec3_conv = make_dsconv_block(64 + 64, 64, stride=1)
        self.dec2_up = PixelShuffleBlock(64, 32, upscale_factor=2)
        self.dec2_conv = make_dsconv_block(32 + 32, 32, stride=1)
        self.dec1_up = PixelShuffleBlock(32, 16, upscale_factor=2)
        self.dec1_conv = make_dsconv_block(16 + 16, 16, stride=1)
        self.dec0_up = PixelShuffleBlock(16, 16, upscale_factor=2)
        self.dec0_conv = make_dsconv_block(16 + in_channels, 16, stride=1)
        
        # 出力レイヤーの改良
        self.head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, kernel_size=1)
        )
        
        # 重みの初期化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, debug=False):
        # x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        
        # 元の入力画像をデコーダの最終段でのskip connection用に保持
        x_orig = x.view(B*N, C, H, W)
        if debug: print(f"Input: {x_orig.shape}")
        
        # CNN Encoder
        s1 = self.enc1(x_orig)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)
        if debug: print(f"CNN s5 out: {s5.shape}")
        
        # Feature Projection & Embedding
        x_feat = self.feat_proj(s5)
        
        D, h_feat, w_feat = x_feat.shape[1], x_feat.shape[2], x_feat.shape[3]
        
        # Transformerの入力形状に変換
        x_feat = x_feat.flatten(2).permute(0, 2, 1)
        x_feat = x_feat.view(B, N, -1, D)
        T = x_feat.shape[2]
        if debug: print(f"Reshaped for Transformer: {x_feat.shape}")

        # 時間の位置エンベディングを加算
        x_feat = x_feat + self.pos_embed_temporal[:, :N, :, :]

        # 勾配クリッピング付きのTransformer
        x_transformed = self.transformer_blocks(x_feat)
        
        # 勾配の安定化
        if self.training:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
        if debug: print(f"Transformer out: {x_transformed.shape}")

        # CNN Decoder
        t_out = x_transformed.permute(0, 1, 3, 2).contiguous().view(B * N, D, h_feat, w_feat)
        if debug: print(f"Reshaped for Decoder: {t_out.shape}")
        
        # Decoder
        d4 = self.dec4_up(t_out); d4 = torch.cat([d4, s4], dim=1); d4 = self.dec4_conv(d4)
        d3 = self.dec3_up(d4);    d3 = torch.cat([d3, s3], dim=1); d3 = self.dec3_conv(d3)
        d2 = self.dec2_up(d3);    d2 = torch.cat([d2, s2], dim=1); d2 = self.dec2_conv(d2)
        d1 = self.dec1_up(d2);    d1 = torch.cat([d1, s1], dim=1); d1 = self.dec1_conv(d1)
        d0 = self.dec0_up(d1);    d0 = torch.cat([d0, x_orig], dim=1); d0 = self.dec0_conv(d0)
        
        if debug: print(f"Final Decoder out: {d0.shape}")

        # Head
        out = self.head(d0)
        
        # 最終的な出力形状を (B, N, C_out, H, W) に戻す
        out = out.view(B, N, 1, H, W)
        if debug: print(f"Final output: {out.shape}")
        
        return out

if __name__ == '__main__':
    # テスト実行例
    batch_size = 2
    n_frames = 5
    in_channels = 3
    h, w = 320, 640

    model = VideoSwinTransformer(
        img_size=(h, w), 
        window_size=5,
        feature_dim=128,  # より小さな特徴次元
        transformer_blocks=1,  # ブロック数を減らす
        transformer_heads=4   # ヘッド数を減らす
    ).to("cuda")

    x = torch.randn(batch_size, n_frames, in_channels, h, w).to("cuda")
    
    print("<<<<< Running with debug=True >>>>>")
    with torch.no_grad():
        y = model(x, debug=True)

    print("\n<<<<< Final output shape check >>>>>")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")