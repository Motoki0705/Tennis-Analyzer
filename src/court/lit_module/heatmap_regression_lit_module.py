from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics.classification import BinaryJaccardIndex


class HeatmapRegressionLitModule(pl.LightningModule):
    """
    PyTorch Lightning module for court keypoint heatmap regression.
    
    This module handles training, validation, and testing of court keypoint detection models
    that output heatmaps for each keypoint.
    
    Attributes:
        model: The neural network model for court keypoint detection
        criterion: Loss function (MSE by default)
        train_iou: IoU metric for training
        val_iou: IoU metric for validation
    """

    def __init__(
        self,
        model: Callable,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
    ):
        """
        Initialize the HeatmapRegressionLitModule.
        
        Args:
            model: Neural network model for court keypoint detection
            lr: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs for learning rate scheduler
            max_epochs: Maximum number of training epochs
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # モデル巨大パラメータは除外

        # 損失関数 & 指標
        self.criterion = nn.MSELoss()
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)

    def _step(self, batch, stage: str):
        """
        Common step for training, validation, and testing.
        
        Args:
            batch: Input batch containing images and heatmaps
            stage: Current stage ('train', 'val', or 'test')
            
        Returns:
            Calculated loss
        """
        frames, heatmaps = batch  # frames: [B, C, H, W], heatmaps: [B, num_keypoints, H_out, W_out]
        logits = self(frames)
        loss = self.criterion(logits, heatmaps)

        # preds: sigmoidして0-1スケールに変換
        preds = torch.sigmoid(logits)

        # カスタムIoUの計算
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_IoU",
            masked_iou,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            sync_dist=True,
        )
        return loss

    @staticmethod
    def compute_masked_iou(
        preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
    ):
        """
        Compute IoU between predicted and target heatmaps.
        
        Args:
            preds: Predicted heatmaps
            targets: Target heatmaps
            threshold: Threshold for binarizing heatmaps
            
        Returns:
            Mean IoU across batch
        """
        gt_mask = targets > threshold
        pred_mask = preds > threshold

        intersection = (gt_mask & pred_mask).float().sum(dim=(1, 2, 3))  # [B]
        gt_area = gt_mask.float().sum(dim=(1, 2, 3)) + 1e-8

        iou = intersection / gt_area
        return iou.mean()  # バッチ平均

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        self._step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
        """
        self._step(batch, stage="test")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and lr_scheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        # Warm-up スケジューラ
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: float(epoch + 1)
            / float(self.hparams.warmup_epochs),
        )

        # CosineAnnealing スケジューラ
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 1e-2,
        )

        # 順次適用するスケジューラ（Warm-up → Cosine）
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
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