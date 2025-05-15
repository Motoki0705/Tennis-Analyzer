import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class KeypointRegressionModule(pl.LightningModule):
    """
    座標回帰用 LightningModule:
    入力 x: [B, 3, N, H, W]
    出力 preds: [B, N, 2] (各フレームの x,y)
    """
    def __init__(
        self,
        model: Callable,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        use_visibility_weighting: bool = False,
        hidden_weight: float = 2.0,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_visibility_weighting = use_visibility_weighting
        self.hidden_weight = hidden_weight

        # フレームごとに (x,y) を回帰するので MSELoss を使う。reduction='none' で重み付け可能に。
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, N, H, W]
        # モデルは [B, N, 2] を返す想定
        return self.model(x)

    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        visibility: torch.Tensor
    ) -> torch.Tensor:
        """
        preds, targets: [B, N, 2]
        visibility:     [B, N]  (0/1/2)
        """
        # フレームごとの誤差: [B, N, 2]
        err = self.loss_fn(preds, targets)

        if self.use_visibility_weighting:
            # visibility==1 (隠れている) を強調
            # 重みベクトル [B, N]
            w = torch.where(visibility == 1,
                            torch.full_like(visibility, self.hidden_weight),
                            torch.ones_like(visibility))
            # [B, N] → [B, N, 2]
            w = w.unsqueeze(-1).expand(-1, -1, 2)
        else:
            w = torch.ones_like(err)

        # 加重 MSE
        loss = (err * w).mean()
        return loss

    def common_step(self, batch, batch_idx, stage="train"):
        frames, coords, visibility = batch
        # frames:    [B, 3, N, H, W]
        # coords:    [B, N, 2]
        # visibility: [B, N]
        preds = self(frames)  # [B, N, 2]
        loss = self.compute_loss(preds, coords, visibility)

        self.log(f"{stage}_loss", loss,
                 on_step=(stage == "train"),
                 on_epoch=True,
                 prog_bar=(stage == "train"),
                 logger=True)
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
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
