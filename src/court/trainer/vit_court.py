from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn


class CourtLModule(pl.LightningModule):
    def __init__(
        self,
        model: Callable,
        focal_weight: float = 1.0,
        kl_weight: float = 1.0,
        gamma: float = 2.0,
        temperature: float = 1.0,
        lr: int = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.focal_weight = focal_weight
        self.kl_weight = kl_weight
        self.gamma = gamma
        self.temperature = temperature

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx, stage="train"):
        frames, heatmaps = batch  # frames: [B, C, H, W], heatmaps: [B, 1, H_out, W_out]
        preds = self(frames)
        loss = self.bce(preds, heatmaps)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            logger=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
