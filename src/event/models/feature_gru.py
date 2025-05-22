import torch
import torch.nn as nn


class FrameFeatureEncoder(nn.Module):
    def __init__(
        self,
        court_dim=45,
        ball_dim=3,
        pose_dim=204,
        bbox_dim=20,
        embed_dim=64,  # 各要素の中間特徴次元
        out_dim=128,
    ):  # 最終的なフレーム特徴次元
        super().__init__()

        self.court_fc = nn.Sequential(nn.Linear(court_dim, embed_dim), nn.ReLU())
        self.ball_fc = nn.Sequential(nn.Linear(ball_dim, embed_dim), nn.ReLU())
        self.pose_fc = nn.Sequential(nn.Linear(pose_dim, embed_dim), nn.ReLU())
        self.bbox_fc = nn.Sequential(nn.Linear(bbox_dim, embed_dim), nn.ReLU())

        # 4つの埋め込みを結合して最終特徴に
        self.fuse_fc = nn.Sequential(nn.Linear(embed_dim * 4, out_dim), nn.ReLU())

    def forward(self, court_feat, ball_feat, pose_feat, bbox_feat):
        """
        Inputs:
            court_feat: (B, T, 45)
            ball_feat:  (B, T, 3)
            pose_feat:  (B, T, 204)
            bbox_feat:  (B, T, 20)
        Output:
            frame_feat: (B, T, out_dim)
        """
        B, T, _ = court_feat.shape

        court_emb = self.court_fc(court_feat)
        ball_emb = self.ball_fc(ball_feat)
        pose_emb = self.pose_fc(pose_feat)
        bbox_emb = self.bbox_fc(bbox_feat)

        concat_feat = torch.cat([court_emb, ball_emb, pose_emb, bbox_emb], dim=-1)
        fused = self.fuse_fc(concat_feat)

        return fused  # shape: (B, T, out_dim)


class EventPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,  # FrameFeatureEncoder の出力次元
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        num_event_classes: int = 4,  # イベント数（背景含む）
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.out_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Linear(self.out_dim, num_event_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) - フレーム特徴列
        Returns:
            logits: (B, T, num_event_classes)
        """
        gru_out, _ = self.gru(x)  # (B, T, H or 2H)
        logits = self.fc(gru_out)
        return logits


class EventStatusModel(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        frame_out_dim=128,
        gru_hidden_dim=128,
        gru_layers=1,
        bidirectional=False,
        num_event_classes=5,  # 背景 + Nイベント
    ):
        super().__init__()

        self.encoder = FrameFeatureEncoder(
            court_dim=45,
            ball_dim=3,
            pose_dim=204,
            bbox_dim=20,
            embed_dim=embed_dim,
            out_dim=frame_out_dim,
        )

        self.predictor = EventPredictor(
            input_dim=frame_out_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_layers,
            bidirectional=bidirectional,
            num_event_classes=num_event_classes,
        )

    def forward(self, court, ball, pose, bbox):
        """
        Inputs:
            court: [B, T, 45]
            ball:  [B, T, 3]
            pose:  [B, T, 204]
            bbox:  [B, T, 20]
        Returns:
            logits: [B, T, num_event_classes]
        """
        frame_feat = self.encoder(court, ball, pose, bbox)  # → [B, T, D]
        logits = self.predictor(frame_feat)  # → [B, T, C]
        return logits
