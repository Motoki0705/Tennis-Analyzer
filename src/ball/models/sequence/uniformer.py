from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(
        inp,
        oup,
        (3, kernel_size, kernel_size),
        (2, stride, stride),
        (1, 0, 0),
        groups=groups,
    )


def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(
        inp,
        oup,
        (1, kernel_size, kernel_size),
        (1, stride, stride),
        (0, 0, 0),
        groups=groups,
    )


def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(
        inp,
        oup,
        (3, kernel_size, kernel_size),
        (1, stride, stride),
        (1, 0, 0),
        groups=groups,
    )


def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)


def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


def bn_3d(dim):
    return nn.BatchNorm3d(dim)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class SplitSABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(
            in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0]
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(
                in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0]
            )
        else:
            self.proj = conv_1xnxn(
                in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0]
            )

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class Uniformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        depth=[5, 8, 20, 7],
        num_classes=400,
        img_size=224,
        in_chans=3,
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.3,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        split=False,
        std=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0]
        )
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=2,
            in_chans=embed_dim[0],
            embed_dim=embed_dim[1],
            std=std,
        )
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8,
            patch_size=2,
            in_chans=embed_dim[1],
            embed_dim=embed_dim[2],
            std=std,
        )
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16,
            patch_size=2,
            in_chans=embed_dim[2],
            embed_dim=embed_dim[3],
            std=std,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))
        ]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth[0])
            ]
        )
        self.blocks2 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + depth[0]],
                    norm_layer=norm_layer,
                )
                for i in range(depth[1])
            ]
        )
        if split:
            self.blocks3 = nn.ModuleList(
                [
                    SplitSABlock(
                        dim=embed_dim[2],
                        num_heads=num_heads[2],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + depth[0] + depth[1]],
                        norm_layer=norm_layer,
                    )
                    for i in range(depth[2])
                ]
            )
            self.blocks4 = nn.ModuleList(
                [
                    SplitSABlock(
                        dim=embed_dim[3],
                        num_heads=num_heads[3],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                        norm_layer=norm_layer,
                    )
                    for i in range(depth[3])
                ]
            )
        else:
            self.blocks3 = nn.ModuleList(
                [
                    SABlock(
                        dim=embed_dim[2],
                        num_heads=num_heads[2],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + depth[0] + depth[1]],
                        norm_layer=norm_layer,
                    )
                    for i in range(depth[2])
                ]
            )
            self.blocks4 = nn.ModuleList(
                [
                    SABlock(
                        dim=embed_dim[3],
                        num_heads=num_heads[3],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                        norm_layer=norm_layer,
                    )
                    for i in range(depth[3])
                ]
            )
        self.norm = bn_3d(embed_dim[-1])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (
            nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if "t_attn.qkv.weight" in name:
                nn.init.constant_(p, 0)
            if "t_attn.qkv.bias" in name:
                nn.init.constant_(p, 0)
            if "t_attn.proj.weight" in name:
                nn.init.constant_(p, 1)
            if "t_attn.proj.bias" in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        y = []
        for blk in self.blocks1:
            x = blk(x)
            y.append(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
            y.append(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
            y.append(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
            y.append(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x, y

    def forward(self, x):
        x, y = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x, y


def uniformer_small():
    return Uniformer(
        depth=[3, 4, 8, 3], embed_dim=[64, 128, 320, 512], head_dim=64, drop_rate=0.1
    )


def uniformer_base():
    return Uniformer(
        depth=[5, 8, 20, 7], embed_dim=[64, 128, 320, 512], head_dim=64, drop_rate=0.3
    )


# ------------------------------------------------------------
# ★ 1) UpSampler 系のユーティリティ
# ------------------------------------------------------------
class UpEmbed(nn.Module):
    """
    ConvTranspose3d で (H,W) を×2 するモジュール
    T を変えないよう kernel_size=(1,2,2), stride=(1,2,2)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.norm = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        return self.norm(self.up(x))


class UpTimeFinal(nn.Module):
    """
    時間軸（T）だけ ×2 に戻す最終アップサンプラ
    """

    def __init__(self, ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(ch, ch, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.norm = nn.BatchNorm3d(ch)

    def forward(self, x):
        return self.norm(self.up(x))


class DecoderBlock(nn.Module):
    """Uniformer の Attention＋MLP をそのまま使う軽量デコーダ"""

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.block = SABlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop=0.0,
            attn_drop=0.0,
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------
# ★ 2) Uniformer UNet ラッパ
# ------------------------------------------------------------
class UniformerUNet(nn.Module):
    def __init__(
        self,
        uniformer_cfg={
            "depth": [3, 4, 8, 3],
            "embed_dim": [64, 128, 320, 512],
            "head_dim": 64,
            "mlp_ratio": 4,
            "drop_rate": 0.5,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "num_classes": 400,
            "img_size": 224,
            "in_chans": 3,
            "split": False,
        },
        weight_path="checkpoints/ball/uniformer/uniformer_small_k400_8x8.pth",
        up_t: bool = True,
    ):
        """
        uniformer_cfg: dict of Uniformer __init__ params
        weight_path:   事前学習済み .pth のパス
        up_t:          True=最初にT/2 ダウンサンプリング、False=時間はダウンサンプルしない
        """
        super().__init__()
        # ---- ① backbone ----
        self.backbone = Uniformer(**uniformer_cfg)
        sd = torch.load(weight_path, map_location="cpu")
        self.backbone.load_state_dict(sd, strict=False)

        # ---- ② PatchEmbed1 の temporal stride 調整 ----
        if not up_t:
            # patch_size=img_size//(H→H/4)→4 なので
            d = uniformer_cfg["embed_dim"][0]
            self.backbone.patch_embed1.proj = conv_1xnxn(
                uniformer_cfg["in_chans"],
                d,
                kernel_size=uniformer_cfg["img_size"] // 4,
                stride=uniformer_cfg["img_size"] // 4,
            )

        # ---- ③ UNet decoder ----
        e0, e1, e2, e3 = uniformer_cfg["embed_dim"]
        self.up4 = UpEmbed(e3, e2)
        self.dec3 = DecoderBlock(e2, e2 // uniformer_cfg["head_dim"])
        self.up3 = UpEmbed(e2, e1)
        self.dec2 = DecoderBlock(e1, e1 // uniformer_cfg["head_dim"])
        self.up2 = UpEmbed(e1, e0)
        self.dec1 = DecoderBlock(e0, e0 // uniformer_cfg["head_dim"])
        self.up_time = UpTimeFinal(e0)
        self.out_conv = nn.Conv3d(e0, 1, kernel_size=1)

    def forward(self, x):
        # backbone から stage ごとに特徴を取得
        _, feats = self.backbone(x)
        # feats[2], feats[5], feats[8], feats[-1] が H/4,H/8,H/16,H/32
        x1, x2, x3, x4 = feats[2], feats[5], feats[8], feats[-1]

        d3 = self.dec3(self.up4(x4) + x3)
        d2 = self.dec2(self.up3(d3) + x2)
        d1 = self.dec1(self.up2(d2) + x1)

        d1 = self.up_time(d1)
        return self.out_conv(d1)


if __name__ == "__main__":
    cfg = dict(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        mlp_ratio=4,
        drop_rate=0.5,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_classes=400,
        img_size=224,
        in_chans=3,
        split=False,
    )

    model = UniformerUNet(
        uniformer_cfg=cfg,
        weight_path=r"checkpoints\ball\uniformer\uniformer_small_k400_8x8.pth",
        up_t=True,  # PatchEmbed1 が T/2 する場合
    ).eval()

    # ダミーデータ
    dummy_input = torch.randn(1, 3, 4, 320, 640)  # [B=1, C=3, T=4, H=128, W=128]

    # 推論
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        print("Logits shape:", logits.shape)
