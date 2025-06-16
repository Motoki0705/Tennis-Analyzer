"""
LiteTrackNet用LightningModule with Focal Loss
"""
from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics.classification import BinaryJaccardIndex

from src.court.models.lite_tracknet import LiteTrackNet


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCELossを計算
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # probabilities を計算
        p_t = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        
        # Focal weight を計算
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Focal loss を計算
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LitLiteTracknetFocal(pl.LightningModule):
    """
    LiteTrackNet用LightningModule with Focal Loss
    
    入力: Frames [B, C, H, W]
    出力: Heatmaps [B, out_channels, H, W]
    """

    def __init__(
        self,
        # モデル構成パラメータ
        in_channels: int = 3,
        out_channels: int = 15,
        
        # 学習パラメータ
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
        
        # Focal Loss パラメータ
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        
        # モデルの初期化
        self.model = LiteTrackNet(
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        # 学習パラメータの保存
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 損失関数と評価指標
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

        # ハイパーパラメータの保存
        self.save_hyperparameters()

    def forward(self, x):
        """順伝播処理"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """学習ステップ"""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """検証ステップ"""
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """テストステップ"""
        return self._common_step(batch, "test")

    def _common_step(self, batch, stage: str):
        """共通処理ステップ"""
        frames, heatmaps = batch
        logits = self(frames)
        
        # Focal Lossで損失を計算
        loss = self.criterion(logits, heatmaps)

        # preds: sigmoidして0-1スケールに変換
        preds = torch.sigmoid(logits)

        # カスタムIoUの計算（チャンネル軸の処理を考慮）
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        # メトリクスのロギング
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_iou",
            masked_iou,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        return loss

    @staticmethod
    def compute_masked_iou(
        preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """マスクされたIoUを計算"""
        gt_mask = targets > threshold
        pred_mask = preds > threshold

        # マルチチャンネルの場合はチャンネル方向で平均を取る
        intersection = (gt_mask & pred_mask).float().sum(dim=(2, 3))  # [B, C]
        gt_area = gt_mask.float().sum(dim=(2, 3)) + 1e-8  # [B, C]

        iou = intersection / gt_area  # [B, C]
        return iou.mean()  # 全体の平均

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Warm-up スケジューラ
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: float(epoch + 1) / float(self.warmup_epochs),
        )

        # CosineAnnealing スケジューラ
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.lr * 1e-2,
        )

        # 順次適用するスケジューラ（Warm-up → Cosine）
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }