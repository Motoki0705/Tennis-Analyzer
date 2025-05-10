import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class KeypointModule(pl.LightningModule):
    def __init__(
        self,
        model: Callable,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        use_visibility_weighting: bool = False
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_visibility_weighting = use_visibility_weighting

        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 手動で加重平均

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, preds, targets, visibility):
        if self.use_visibility_weighting:
            weights = torch.where(visibility == 1, 2.0, 1.0).to(preds.device)  # 強調
            weights = weights.unsqueeze(-1).unsqueeze(-1)
        else:
            weights = torch.ones_like(preds)

        loss = self.bce(preds, targets) * weights
        return loss.mean()

    def common_step(self, batch, batch_idx, stage="train"):
        frames, heatmaps, visibility = batch
        preds = self(frames)
        loss = self.compute_loss(preds, heatmaps, visibility)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=(stage == "train"), logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
