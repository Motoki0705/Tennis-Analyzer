from typing import Dict, List, Tuple, Optional, Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall


class EventDetectionLitModule(pl.LightningModule):
    """
    イベント検出用のLightningModule。
    時系列モデルを用いてボールのイベントステータス（ヒットとバウンド）を予測します。
    マルチラベル分類として[hit, bounce]の2チャンネルで出力します。
    """

    def __init__(
        self,
        model: nn.Module,
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
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        
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
            self.train_f1(preds.view(-1, 2), targets.view(-1, 2))
            self.log(f"{stage}_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.val_f1(preds.view(-1, 2), targets.view(-1, 2))
            self.val_precision(preds.view(-1, 2), targets.view(-1, 2))
            self.val_recall(preds.view(-1, 2), targets.view(-1, 2))
            self.val_auroc(probs.view(-1, 2), targets.view(-1, 2))
            
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