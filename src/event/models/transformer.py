import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Positional Encoding（学習可能埋め込み）
# ---------------------------------------------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pe(pos)        # (B, T, d)

# ---------------------------------------------------------
# Event Transformer
# ---------------------------------------------------------
class EventTransformer(nn.Module):
    """
    入力:
        ball_tensor         : (B, T, 3)
        player_bbox_tensor  : (B, T, P, 5)
        player_pose_tensor  : (B, T, P, pose_dim)
        court_tensor        : (B, T, 31)
    出力:
        logits              : (B, T)          # BCEWithLogitsLoss 用
    """
    def __init__(
        self,
        pose_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # --- 1. Frame-wise embedding --------------------------------------
        self.ball_embed   = nn.Linear(3, d_model)
        self.court_embed  = nn.Linear(31, d_model)
        self.player_embed = nn.Linear(5 + pose_dim, d_model)   # per player
        self.frame_proj   = nn.Linear(3 * d_model, d_model)    # concat → d

        # --- 2. Positional encoding --------------------------------------
        self.pos_enc = LearnablePositionalEncoding(max_seq_len, d_model)

        # --- 3. Transformer encoder --------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,        # (B, T, d)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # --- 4. Classifier head ------------------------------------------
        self.classifier = nn.Linear(d_model, 1)

    # ---------------------------------------------------------------------
    def forward(
        self,
        ball: torch.Tensor,          # (B, T, 3)
        player_bbox: torch.Tensor,   # (B, T, P, 5)
        player_pose: torch.Tensor,   # (B, T, P, pose_dim)
        court: torch.Tensor,         # (B, T, 31)
    ) -> torch.Tensor:
        B, T, P, _ = player_bbox.shape

        # ------ frame-wise player aggregation ----------------------------
        players   = torch.cat([player_bbox, player_pose], dim=-1)  # (B,T,P,5+pose_dim)

        # mask: True for “valid” rows (≠0) after Dataset padding
        valid_mask = (players.abs().sum(dim=-1) > 0)               # (B, T, P)

        # embed each player
        players_emb = self.player_embed(players)                   # (B, T, P, d)
        players_emb = players_emb * valid_mask.unsqueeze(-1)       # zero-out padded

        # masked mean pooling over players
        denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)  # avoid 0-div
        player_repr = players_emb.sum(dim=-2) / denom              # (B, T, d)

        # ------ ball & court embedding -----------------------------------
        ball_repr  = self.ball_embed(ball)     # (B, T, d)
        court_repr = self.court_embed(court)   # (B, T, d)

        # ------ concat & project to frame token --------------------------
        frame_token = torch.cat([ball_repr, court_repr, player_repr], dim=-1)  # (B,T,3d)
        frame_token = self.frame_proj(frame_token)                             # (B,T,d)

        # ------ positional encoding & Transformer ------------------------
        x = self.pos_enc(frame_token)               # (B,T,d)
        x = self.encoder(x)                         # (B,T,d)

        # ------ per-time-step logits -------------------------------------
        logits = self.classifier(x).squeeze(-1)     # (B,T)

        return logits             # ※ sigmoid は Loss 側で
