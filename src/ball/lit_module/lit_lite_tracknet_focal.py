"""
LiteTrackNet用LightningModule (Focal Loss版)
"""
import os
import random
from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

from src.ball.models.lite_tracknet import LiteTrackNet


class LitLiteTracknetFocalLoss(pl.LightningModule):
    """
    LiteTrackNet用LightningModule (Focal Loss版)
    
    入力: Frames [B, C, H, W]
    出力: Heatmaps [B, H, W]
    """

    def __init__(
        self,
        # モデル構成パラメータ
        in_channels: int = 9,
        out_channels: int = 1,
        
        # 学習パラメータ
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,

        # ===== 変更点: Focal Loss用パラメータ =====
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,

        # 可視化パラメータ
        num_log_images: int = 4,
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.model = LiteTrackNet(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels
        )
        
        # 検証ステップで可視化のために最初のバッチの出力を保存する変数
        self._val_batch_outputs_for_log = None

    def forward(self, x):
        """順伝播処理"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """学習ステップ"""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """検証ステップ"""
        # 最初のバッチの入出力を可視化用に保存
        if batch_idx == 0:
            frames, heatmaps, _, _ = batch
            logits = self(frames)
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            preds = torch.sigmoid(logits)
            
            self._val_batch_outputs_for_log = {
                "frames": frames.cpu(),
                "preds": preds.cpu(),
                "targets": heatmaps.cpu()
            }
            
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """テストステップ"""
        return self._common_step(batch, "test")

    def _common_step(self, batch, stage: str):
        """共通処理ステップ"""
        frames, heatmaps, _, _ = batch
        logits = self(frames)
        
        if heatmaps.dim() == 3:
            heatmaps = heatmaps.unsqueeze(1)
        
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
            
        # ===== 変更点: 損失関数をFocal Lossに変更 =====
        loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=heatmaps,
            alpha=self.hparams.focal_alpha,
            gamma=self.hparams.focal_gamma,
            reduction="mean",
        )
        
        preds = torch.sigmoid(logits)
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_iou", masked_iou, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        """検証エポックの終わりに画像セットを保存"""
        if self._val_batch_outputs_for_log is None or not self.logger or not hasattr(self.logger, 'log_dir'):
            return

        log_dir = self.logger.log_dir
        save_dir = os.path.join(log_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        
        frames = self._val_batch_outputs_for_log["frames"]
        preds = self._val_batch_outputs_for_log["preds"]
        targets = self._val_batch_outputs_for_log["targets"]
        batch_size = frames.size(0)
        
        num_images = min(self.hparams.num_log_images, batch_size)
        selected_indices = random.sample(range(batch_size), num_images)

        for idx, batch_idx in enumerate(selected_indices):
            frame_img = frames[batch_idx, :3]
            pred_heatmap = preds[batch_idx]
            target_heatmap = targets[batch_idx]

            combined_image = torch.cat([
                frame_img,
                pred_heatmap.repeat(3, 1, 1),
                target_heatmap.repeat(3, 1, 1)
            ], dim=2)

            filename = os.path.join(save_dir, f"epoch_{self.current_epoch:03d}_sample_{idx:02d}.png")
            torchvision.utils.save_image(combined_image, filename)

        self._val_batch_outputs_for_log = None

    @staticmethod
    def compute_masked_iou(
        preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """マスクされたIoUを計算"""
        gt_mask = targets > threshold
        pred_mask = preds > threshold
        intersection = (gt_mask & pred_mask).float().sum(dim=(1, 2, 3))
        union = (gt_mask | pred_mask).float().sum(dim=(1, 2, 3))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (epoch + 1) / self.hparams.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs - self.hparams.warmup_epochs, eta_min=self.hparams.lr * 1e-2
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.hparams.warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}