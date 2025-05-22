from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn


class CourtLModule(pl.LightningModule):
    def __init__(
        self,
        model: Callable,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optim_t_max: int = 10,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.mse = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx, stage: str):
        frames, heatmaps = batch  # frames: [B, C, H, W], heatmaps: [B, 1, H_out, W_out]
        preds = self(frames)
        loss = self.mse(preds, heatmaps)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage != "train"),
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.optim_t_max, eta_min=self.hparams.min_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
