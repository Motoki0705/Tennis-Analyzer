import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR
from torchmetrics.classification import BinaryJaccardIndex
from typing import Callable


# ────────────────────────────────────────────────────────
# 2. LightningModule
# ────────────────────────────────────────────────────────
class CatFramesLitModule(pl.LightningModule):
    """
    - 入力: Frames [B, C, H, W]
    - 出力: Heatmaps [B, 1, H_out, W_out]
    """
    def __init__(
        self,
        model: Callable,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
        bce_weight: float = 0.7,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])   # モデル巨大パラメータは除外

        # 損失関数 & 指標
        self.criterion = nn.MSELoss()
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

    # ───────────── forward ─────────────
    def forward(self, x):
        return self.model(x)

    # ───────────── 共通ステップ ─────────────
    def _step(self, batch, stage: str):
        frames, heatmaps, _ = batch
        logits = self(frames)
        loss = self.criterion(logits, heatmaps)

        # preds: sigmoidして0-1スケールに変換
        preds = torch.sigmoid(logits)

        # カスタムIoUの計算
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True,
                prog_bar=(stage == "val"), logger=True, sync_dist=True)
        self.log(f"{stage}_IoU", masked_iou, on_step=False, on_epoch=True,
                prog_bar=(stage == "val"), logger=True, sync_dist=True)
        return loss

    @staticmethod
    def compute_masked_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        gt_mask = (targets > threshold)
        pred_mask = (preds > threshold)

        intersection = (gt_mask & pred_mask).float().sum(dim=(1, 2, 3))  # [B]
        gt_area = gt_mask.float().sum(dim=(1, 2, 3)) + 1e-8

        iou = intersection / gt_area
        return iou.mean()  # バッチ平均


    # Lightning hooks ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._step(batch, stage="test")

    # ───────────── 最適化器 & Scheduler ─────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )

        # Warm-up スケジューラ
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: float(epoch + 1) / float(self.hparams.warmup_epochs)
        )

        # CosineAnnealing スケジューラ
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 1e-2
        )

        # 順次適用するスケジューラ（Warm-up → Cosine）
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
