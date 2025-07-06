"""
VideoSwinTransformer用LightningModule
"""
from typing import Dict, Any
import os
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics.classification import BinaryJaccardIndex
import torchvision

from src.ball.models.video_swin_transformer import VideoSwinTransformer



class LitVideoSwinTransformer(pl.LightningModule):
    """
    VideoSwinTransformer用LightningModule
    
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
        # ===== 変更点: bce_weightを追加 =====
        bce_weight: float = 0.5,

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
        
        # ===== 変更点: 損失関数をMSEとBCEに分離 =====
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

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
        
        # ===== 変更点: MSE + BCEの加重和損失を計算 =====
        preds = torch.sigmoid(logits)
        loss_mse = self.mse_loss(preds, heatmaps)
        loss_bce = self.bce_loss(logits, heatmaps)
        loss = (1 - self.hparams.bce_weight) * loss_mse + self.hparams.bce_weight * loss_bce
        
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_iou", masked_iou, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        if self._val_batch_outputs_for_log is None:
            return

        if self.logger and hasattr(self.logger, 'log_dir'):
            log_dir = self.logger.log_dir
            save_dir = os.path.join(log_dir, "validation_images")
            os.makedirs(save_dir, exist_ok=True)
            
            frames = self._val_batch_outputs_for_log["frames"]
            preds = self._val_batch_outputs_for_log["preds"]
            targets = self._val_batch_outputs_for_log["targets"]

            batch_size = frames.shape[0]
            
            available_indices = list(range(batch_size))
            num_images = min(self.hparams.num_log_images, batch_size)
            selected_indices = random.sample(available_indices, num_images)

            for idx, batch_idx in enumerate(selected_indices):
                # 表示用にシーケンスの最初のフレームを選択
                frame_idx = 0
                
                frame_img = frames[batch_idx, frame_idx, :3]
                pred_heatmap = preds[batch_idx, frame_idx]
                target_heatmap = targets[batch_idx, frame_idx]
                
                if target_heatmap.dim() == 2:
                    target_heatmap = target_heatmap.unsqueeze(0)
                
                # 画像を結合
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
        gt_mask = targets > threshold
        pred_mask = preds > threshold
        intersection = (gt_mask & pred_mask).float().sum(dim=(2, 3, 4))
        gt_area = gt_mask.float().sum(dim=(2, 3, 4)) + 1e-8
        iou = intersection / gt_area
        return iou.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: float(epoch + 1) / float(self.hparams.warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs - self.hparams.warmup_epochs, eta_min=self.hparams.lr * 1e-2
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.hparams.warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}} 