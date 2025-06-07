from typing import Dict, List, Tuple, Optional, Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall


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
        q = self.player_attn_q.expand(B*T, -1, -1)        # (B*T,1,d)
        k = flat_players                                  # (B*T,P,d)
        v = flat_players                                  # (B*T,P,d)
        attn_out, _ = self.player_attn(q, k, v, key_padding_mask=~flat_mask)  # out: (B*T,1,d)
        player_repr = attn_out.view(B, T, -1)             # (B,T,d)

        # --- (2) Ball & Court 埋め込み + 各 Encoder ___
        ball_repr  = self.ball_embed(ball)    # (B,T,d)
        court_repr = self.court_embed(court)  # (B,T,d)

        # pos encoding
        ball_repr   = self.pos_enc(ball_repr)
        court_repr  = self.pos_enc(court_repr)
        player_repr = self.pos_enc(player_repr)

        # 各モダリティに Transformer
        x_ball   = self.encoder_ball(ball_repr)     # (B,T,d)
        x_court  = self.encoder_court(court_repr)   # (B,T,d)
        x_player = self.encoder_player(player_repr) # (B,T,d)

        # --- (3) Cross‐Attention で融合 ___
        # stack to keys/values: (B,T,3,d) → flatten (B*T, 3, d)
        stacked = torch.stack([x_ball, x_court, x_player], dim=2)  # (B,T,3,d)
        flat_stacked = stacked.view(B*T, 3, -1)                   # (B*T,3,d)

        # query: (1,1,d) → expand (B*T,1,d)
        q2 = self.cross_query.expand(B*T, -1, -1)                  # (B*T,1,d)
        k2 = flat_stacked                                         # (B*T,3,d)
        v2 = flat_stacked                                         # (B*T,3,d)

        fused, _ = self.cross_attn(q2, k2, v2)                     # (B*T,1,d)
        fused = fused.view(B, T, -1)                               # (B,T,d)

        # --- (4) 最後の分類ヘッド ___
        logits = self.classifier(fused)  # (B,T,2)
        return logits


class EventDetectionLitModule(pl.LightningModule):
    """
    イベント検出用のLightningModule。
    時系列モデルを用いてボールのイベントステータス（ヒットとバウンド）を予測します。
    マルチラベル分類として[hit, bounce]の2チャンネルで出力します。
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pose_dim: int = 51,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        no_hit_weight: float = 0.01,  # no_hit(0,0)の重み
        hit_weight: float = 1.0,     # hit(1,0)の重み
        bounce_weight: float = 1.0,  # bounce(0,1)の重み
        clarity_weight: float = 0.02,  # 明確な予測を促進する重み
    ):
        super().__init__()
        
        # モデルを直接LitModule内で初期化
        self.model = EventTransformerV2(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pose_dim=pose_dim
        )
        
        # ハイパーパラメータを保存
        self.save_hyperparameters()
        
        # 損失関数の重み
        self.no_hit_weight = no_hit_weight
        self.hit_weight = hit_weight
        self.bounce_weight = bounce_weight
        self.clarity_weight = clarity_weight
        
        # 評価指標（マルチラベル）
        self.train_f1 = MultilabelF1Score(num_labels=2)
        self.val_f1 = MultilabelF1Score(num_labels=2)
        self.val_precision = MultilabelPrecision(num_labels=2)
        self.val_recall = MultilabelRecall(num_labels=2)
        self.val_auroc = AUROC(task="multilabel", num_labels=2)

    def forward(self, ball_features, player_bbox_features, player_pose_features, court_features):
        return self.model(ball_features, player_bbox_features, player_pose_features, court_features)

    def custom_event_loss(self, predictions, targets):
        """
        カスタムイベント検出損失関数。以下の特性を持ちます：
        1. no_hit(0,0)に対する重みを下げる
        2. hit(1,0)とbounce(0,1)への明確な分類を促進
        3. 曖昧な予測(1,1)を抑制
        
        Args:
            predictions: モデルの予測 [batch_size, seq_len, 2]
            targets: 教師データ [batch_size, seq_len, 2]
            
        Returns:
            torch.Tensor: 計算された損失値
        """
        batch_size, seq_len, _ = predictions.shape
        predictions = predictions.view(-1, 2)  # [batch_size*seq_len, 2]
        targets = targets.view(-1, 2)  # [batch_size*seq_len, 2]
        
        # 1. 基本的なBCELoss (バイナリクロスエントロピー)
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # 2. サンプルごとの重み付け
        weights = torch.ones_like(targets)
        
        # no_hit (0,0)のサンプルに低い重みを設定
        no_hit_mask = (targets[:, 0] == 0) & (targets[:, 1] == 0)
        weights[no_hit_mask, :] = self.no_hit_weight
        
        # hit (1,0)のサンプルに高い重みを設定
        hit_mask = (targets[:, 0] == 1) & (targets[:, 1] == 0)
        weights[hit_mask, :] = self.hit_weight
        
        # bounce (0,1)のサンプルに高い重みを設定
        bounce_mask = (targets[:, 0] == 0) & (targets[:, 1] == 1)
        weights[bounce_mask, :] = self.bounce_weight
        
        # 重み付きBCELoss
        weighted_bce_loss = (bce_loss * weights).mean()
        
        # 3. 明確な予測を促進する項（ヒットとバウンスの排他性を強調）
        # シグモイド関数を適用して確率に変換
        probs = torch.sigmoid(predictions)
        
        # 確率の積が小さくなるように損失を追加（排他的な関係を促進）
        clarity_loss = (probs[:, 0] * probs[:, 1]).mean()
        
        # 4. 最終的な損失を計算
        total_loss = weighted_bce_loss + self.clarity_weight * clarity_loss
        
        return total_loss

    def _step(self, batch, batch_idx, stage):
        ball_features, player_bbox_features, player_pose_features, court_features, targets, _ = batch
        
        # 予測
        logits = self(ball_features, player_bbox_features, player_pose_features, court_features)  # [batch_size, seq_len, 2]
        
        # ロス計算（カスタム損失関数）
        loss = self.custom_event_loss(logits, targets)
        
        # 予測結果（シグモイド関数で確率に変換し、0.5以上を陽性と判定）
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # 指標計算
        if stage == "train":
            self.train_f1(preds.view(-1, 2), targets.view(-1, 2).long())
            self.log(f"{stage}_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.val_f1(preds.view(-1, 2), targets.view(-1, 2).long())
            self.val_precision(preds.view(-1, 2), targets.view(-1, 2).long())
            self.val_recall(preds.view(-1, 2), targets.view(-1, 2).long())
            self.val_auroc(probs.view(-1, 2), targets.view(-1, 2).long())
            
            self.log(f"{stage}_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_precision", self.val_precision, on_step=False, on_epoch=True)
            self.log(f"{stage}_recall", self.val_recall, on_step=False, on_epoch=True)
            self.log(f"{stage}_auroc", self.val_auroc, on_step=False, on_epoch=True)
        
        # ロス記録
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        # ウォームアップ → コサインアニーリングスケジューラ
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: min(1.0, float(e + 1) / float(self.hparams.warmup_epochs)),
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 1e-2,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs],
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        } 