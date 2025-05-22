from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn


class KeypointModule(pl.LightningModule):
    def __init__(
        self,
        model: Callable,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_backbone_epochs: int = 0,
        use_visibility_weighting: bool = False,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_visibility_weighting = use_visibility_weighting
        self.freeze_backbone_epochs = freeze_backbone_epochs

        self.bce = nn.BCEWithLogitsLoss()  # 手動で加重平均

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print(
                f"[Info] BackBone is frozen (until epoch {self.freeze_backbone_epochs})"
            )

    def unfreeze_backbone(self):
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            print(f"[Info] BackBone is unfrozen (from epoch {self.current_epoch})")

    def on_train_epoch_start(self):
        if self.freeze_backbone_epochs > 0 and self.current_epoch == 0:
            self.freeze_backbone()
        if (
            self.freeze_backbone_epochs > 0
            and self.current_epoch == self.freeze_backbone_epochs
        ):
            self.unfreeze_backbone()

    def common_step(self, batch, batch_idx, stage="train"):
        frames, heatmaps, visibility = batch
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
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
