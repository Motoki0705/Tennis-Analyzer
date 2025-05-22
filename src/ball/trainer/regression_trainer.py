from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR


class CoordRegressionLitModule(pl.LightningModule):
    """
    - 入力: Frames [B, C, H, W]
    - 出力: NormCoords [B, 2]  (x, y) ∈ [0,1]
    """

    def __init__(
        self,
        model: Callable,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        # L1Loss を reduction='none' にして可視フラグでマスクをかける
        self.criterion = nn.L1Loss(reduction="none")

    def forward(self, x):
        # モデルは (B, C, H, W) → (B, 2) を返すものとする
        return self.model(x)

    def _step(self, batch, stage: str):
        frames, coords, visibility = batch
        # coords: [B, 2]  or if "all" then [B, T, 2]  ※今回は last フレームのみ想定
        pred = self(frames)  # [B, 2]

        # L1 損失を計算し、visibility でマスク
        loss_per_dim = self.criterion(pred, coords)  # [B, 2]
        loss = loss_per_dim.mean()  # バッチ内全要素平均

        # 平均L1距離を補助指標として計算
        l1_dist = torch.norm(pred - coords, dim=1).mean()  # [B]

        # ロギング
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_L1Dist",
            l1_dist,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # warm-up → cosine annealing
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: float(e + 1) / float(self.hparams.warmup_epochs),
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 1e-2,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
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
