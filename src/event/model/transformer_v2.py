# event_transformer_v2.py

from typing import Dict, List, Tuple, Optional, Any, Callable

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
        return x + self.pe(pos)      # (B, T, d)


class EventTransformerV2(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, dropout=0.1,
                 max_seq_len=512, pose_dim=51):
        super().__init__()
        # 1) 各モダリティの埋め込み
        self.ball_embed   = nn.Linear(3, d_model)
        self.court_embed  = nn.Linear(45, d_model)
        self.player_embed = nn.Linear(5 + pose_dim, d_model)

        # 2) プレイヤー情報用の Attention Pooling
        self.player_attn_q = nn.Parameter(torch.zeros(1, 1, d_model))  # learnable query
        self.player_attn   = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # 3) モダリティごとに小さな Transformer
        enc_layer_ball   = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, dropout, batch_first=True, norm_first=True)
        enc_layer_court  = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, dropout, batch_first=True, norm_first=True)
        enc_layer_player = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, dropout, batch_first=True, norm_first=True)
        self.encoder_ball   = nn.TransformerEncoder(enc_layer_ball,   num_layers)
        self.encoder_court  = nn.TransformerEncoder(enc_layer_court,  num_layers)
        self.encoder_player = nn.TransformerEncoder(enc_layer_player, num_layers)

        # 4) Positional Encoding（ここは共通にしてもOK）
        self.pos_enc = LearnablePositionalEncoding(max_seq_len, d_model)

        # 5) Cross‐Attention で融合
        self.cross_query = nn.Parameter(torch.zeros(1, 1, d_model))   # learnable fusion query
        self.cross_attn  = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # 6) 分類ヘッド
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, ball, player_bbox, player_pose, court):
        B, T, P, _ = player_bbox.shape

        # --- (1) プレイヤー特徴の Attention Pooling ___
        # players: (B, T, P, d_model)
        players = torch.cat([player_bbox, player_pose], dim=-1)
        players_emb = self.player_embed(players)  # (B, T, P, d)
        valid_mask = (players_emb.abs().sum(-1) > 0)  # (B, T, P)

        # flatten (B*T, P, d)
        flat_players = players_emb.view(B*T, P, -1)
        flat_mask = valid_mask.view(B*T, P)

        # attention pooling: query は (1,1,d) → expand to (B*T,1,d)
        q = self.player_attn_q.expand(B*T, -1, -1)      # (B*T,1,d)
        k = flat_players                               # (B*T,P,d)
        v = flat_players                               # (B*T,P,d)
        attn_out, _ = self.player_attn(q, k, v, key_padding_mask=~flat_mask)  # out: (B*T,1,d)
        player_repr = attn_out.view(B, T, -1)          # (B,T,d)

        # --- (2) Ball & Court 埋め込み + 各 Encoder ___
        ball_repr  = self.ball_embed(ball)    # (B,T,d)
        court_repr = self.court_embed(court)  # (B,T,d)

        # pos encoding
        ball_repr   = self.pos_enc(ball_repr)
        court_repr  = self.pos_enc(court_repr)
        player_repr = self.pos_enc(player_repr)

        # 各モダリティに Transformer
        x_ball   = self.encoder_ball(ball_repr)    # (B,T,d)
        x_court  = self.encoder_court(court_repr)  # (B,T,d)
        x_player = self.encoder_player(player_repr) # (B,T,d)

        # --- (3) Cross‐Attention で融合 ___
        # stack to keys/values: (B,T,3,d) → flatten (B*T, 3, d)
        stacked = torch.stack([x_ball, x_court, x_player], dim=2)  # (B,T,3,d)
        flat_stacked = stacked.view(B*T, 3, -1)                    # (B*T,3,d)

        # query: (1,1,d) → expand (B*T,1,d)
        q2 = self.cross_query.expand(B*T, -1, -1)                   # (B*T,1,d)
        k2 = flat_stacked                                          # (B*T,3,d)
        v2 = flat_stacked                                          # (B*T,3,d)

        fused, _ = self.cross_attn(q2, k2, v2)                      # (B*T,1,d)
        fused = fused.view(B, T, -1)                               # (B,T,d)

        # --- (4) 最後の分類ヘッド ___
        logits = self.classifier(fused)  # (B,T,2)
        return logits