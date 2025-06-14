"""
VideoSwinTransformer用LightningModule (Focal Loss版)
"""
from typing import Dict, Any
import os
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

from src.ball.models.seq_lite_transformer import VideoSwinTransformer


class LitSeqLiteTransformerFocalLoss(pl.LightningModule):
    """
    VideoSwinTransformer用LightningModule (Focal Loss版)
    
    入力: Frame Sequences [B, N, C, H, W]
    出力: Heatmap Sequences [B, N, C_out, H, W]
    """

    def __init__(
        self,
        # モデル構成パラメータ
        img_size: tuple = (320, 640),
        in_channels: int = 3,
        out_channels: int = 1,
        n_frames: int = 10,
        window_size: int = 7,
        feature_dim: int = 256,
        transformer_blocks: int = 2,
        transformer_heads: int = 8,
        
        # 学習パラメータ
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
        # ===== 変更: Focal Loss用パラメータを追加 =====
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,

        # 可視化パラメータ
        num_log_images: int = 4,
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.model = VideoSwinTransformer(
            img_size=self.hparams.img_size,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            n_frames=self.hparams.n_frames,
            window_size=self.hparams.window_size,
            feature_dim=self.hparams.feature_dim,
            transformer_blocks=self.hparams.transformer_blocks,
            transformer_heads=self.hparams.transformer_heads,
        )
        
        self._val_batch_outputs_for_log = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            frames, heatmaps, _, _ = batch
            logits = self(frames)
            preds = torch.sigmoid(logits)
            
            self._val_batch_outputs_for_log = {
                "frames": frames.cpu(),
                "preds": preds.cpu(),
                "targets": heatmaps.cpu()
            }
            
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def _common_step(self, batch, stage: str):
        frames, heatmaps, _, _ = batch
        logits = self(frames)  # [B, N, C_out, H, W]
        
        if heatmaps.dim() == 4:
            heatmaps = heatmaps.unsqueeze(2)
        if logits.dim() == 4:
            logits = logits.unsqueeze(2)
        
        # ===== 変更: Focal Lossの計算 =====
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
        if self._val_batch_outputs_for_log is None or not self.logger or not hasattr(self.logger, 'log_dir'):
            return

        log_dir = self.logger.log_dir
        save_dir = os.path.join(log_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        
        frames = self._val_batch_outputs_for_log["frames"]
        preds = self._val_batch_outputs_for_log["preds"]
        targets = self._val_batch_outputs_for_log["targets"]
        batch_size = frames.shape[0]
        
        num_images = min(self.hparams.num_log_images, batch_size)
        selected_indices = random.sample(range(batch_size), num_images)

        for idx, batch_idx in enumerate(selected_indices):
            # 表示用にシーケンスの中間のフレームを選択
            frame_idx = frames.shape[1] // 2
            
            frame_img = frames[batch_idx, frame_idx]
            pred_heatmap = preds[batch_idx, frame_idx]
            target_heatmap = targets[batch_idx, frame_idx]
            
            combined_image = torch.cat([
                frame_img,
                pred_heatmap.repeat(3, 1, 1),
                target_heatmap.repeat(3, 1, 1)
            ], dim=2)

            filename = os.path.join(save_dir, f"epoch_{self.current_epoch:03d}_sample_{idx:02d}.png")
            torchvision.utils.save_image(combined_image, filename)

        self._val_batch_outputs_for_log = None

    @staticmethod
    def compute_masked_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """マスクされたIoU (Intersection over Union) を計算"""
        gt_mask = targets > threshold
        pred_mask = preds > threshold
        
        intersection = (gt_mask & pred_mask).float().sum(dim=(2, 3, 4))
        union = (gt_mask | pred_mask).float().sum(dim=(2, 3, 4))
        
        # ゼロ除算を避けるための微小値を追加
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
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