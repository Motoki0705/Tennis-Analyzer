from typing import Dict, List, Tuple, Optional, Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics import Accuracy, F1Score, Precision, Recall


class EventDetectionLitModule(pl.LightningModule):
    """
    イベント検出用のLightningModule。
    時系列モデルを用いてボールのイベントステータスを予測します。
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=["model"])
        
        # クラス重みがある場合は重み付き損失関数を使用
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None
        
        # 評価指標
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, ball_features, player_bbox_features, player_pose_features, court_features):
        return self.model(ball_features, player_bbox_features, player_pose_features, court_features)

    def _step(self, batch, batch_idx, stage):
        ball_features, player_bbox_features, player_pose_features, court_features, targets, _ = batch
        
        # 予測
        logits = self(ball_features, player_bbox_features, player_pose_features, court_features)  # [batch_size, seq_len, num_classes]
        
        # ロス計算
        if stage == "train" and self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes), 
                targets.view(-1), 
                weight=self.class_weights
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes), 
                targets.view(-1)
            )
        
        # 予測結果
        preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        
        # 指標計算
        if stage == "train":
            self.train_acc(preds.view(-1), targets.view(-1))
            self.log(f"{stage}_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.val_acc(preds.view(-1), targets.view(-1))
            self.val_f1(preds.view(-1), targets.view(-1))
            self.val_precision(preds.view(-1), targets.view(-1))
            self.val_recall(preds.view(-1), targets.view(-1))
            
            self.log(f"{stage}_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_precision", self.val_precision, on_step=False, on_epoch=True)
            self.log(f"{stage}_recall", self.val_recall, on_step=False, on_epoch=True)
        
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