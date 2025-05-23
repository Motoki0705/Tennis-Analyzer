import torch.nn as nn
from transformers import TimesformerModel


class TimeSformerBall(nn.Module):
    def __init__(
        self,
        num_keypoints=1,
        pretrained_model="facebook/timesformer-base-finetuned-k400",
    ):
        super().__init__()

        # 事前学習済み TimeSformer をロード（動画分類モデル）
        self.encoder = TimesformerModel.from_pretrained(pretrained_model)

        # 出力次元（ViTと同様に hidden_size）
        hidden_dim = self.encoder.config.hidden_size

        # Decoder: simple upsampling + conv to heatmap
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(hidden_dim, num_keypoints, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) → (B, 3, 3, 224, 224) 例: 3フレームの動画（RGB）
        Returns:
            heatmap: (B, num_keypoints, H_out, W_out)
        """
        B, T, C, H, W = x.shape

        # TimeSformerの入力形式に整形
        inputs = {"pixel_values": x}  # transformersはこのkey名で受け取る

        outputs = self.encoder(**inputs)

        # 出力: (B, num_patches+1, hidden_dim)
        hidden = outputs.last_hidden_state[:, 1:, :]  # remove CLS

        # reshape: (B, hidden_dim, H_patches, W_patches)
        # 1. hidden: (B, T*N, C)
        B, N, C = hidden.shape
        T = self.encoder.config.num_frames
        N_p = N // T  # パッチ数 per フレーム
        H_feat = W_feat = int(N_p**0.5)

        # 2. reshape to (B, T, H_feat, W_feat, C)
        hidden = hidden.view(B, T, H_feat, W_feat, C)

        # 3. 時間次元 T を平均
        hidden = hidden.mean(dim=1)  # → (B, H_feat, W_feat, C)

        # 4. (B, C, H_feat, W_feat)
        feat_map = hidden.permute(0, 3, 1, 2).contiguous()

        heatmaps = self.decoder(feat_map)  # (B, num_keypoints, H_out, W_out)
        return heatmaps


def check_param_groups(model):
    # グループの準備
    backbone_params = list(model.encoder.embeddings.parameters()) + list(
        model.encoder.encoder.layer.parameters()
    )
    decoder_params = list(model.decoder.parameters())

    # IDで高速比較のためにsetに変換
    backbone_ids = set(id(p) for p in backbone_params)
    decoder_ids = set(id(p) for p in decoder_params)

    print("===== Parameter Group Check =====")
    for name, param in model.named_parameters():
        pid = id(param)
        if pid in backbone_ids and pid in decoder_ids:
            print(f"[❌ 重複] {name}")
        elif pid in backbone_ids:
            print(f"[Backbone] {name}")
        elif pid in decoder_ids:
            print(f"[Decoder ] {name}")
        else:
            print(f"[⚠️ 未分類] {name}")

    print("Total backbone params:", len(backbone_ids))
    print("Total decoder  params:", len(decoder_ids))
    print("Total model    params:", len(list(model.parameters())))
    print("Sum of groups  params:", len(backbone_ids.union(decoder_ids)))


if __name__ == "__main__":
    model = TimeSformerBall()
    check_param_groups(model)
