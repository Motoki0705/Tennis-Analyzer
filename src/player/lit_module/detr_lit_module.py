import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class DetrLitModule(pl.LightningModule):
    """
    PyTorch Lightning module for DETR-based object detection models.
    
    This module handles training, validation, and testing of DETR-based models
    that are used for player detection.
    
    Attributes:
        model: The DETR-based neural network model
        lr: Learning rate for non-backbone parameters
        lr_backbone: Learning rate for backbone parameters
        weight_decay: Weight decay for optimizer
        optim_t_max: Maximum number of iterations for CosineAnnealingLR
        min_lr: Minimum learning rate
        num_freeze_epoch: Number of epochs to freeze the backbone
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        optim_t_max: int = 50,
        min_lr: float = 1e-6,
        num_freeze_epoch: int = 3,
    ):
        """
        Initialize the DetrLitModule.
        
        Args:
            model: DETR-based model
            lr: Learning rate (except backbone)
            lr_backbone: Learning rate for backbone
            weight_decay: Weight decay
            optim_t_max: T_max for CosineAnnealingLR
            min_lr: Minimum learning rate
            num_freeze_epoch: Number of epochs to freeze backbone
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    def freeze_encoder(self):
        """Freeze the backbone encoder parameters."""
        found = False
        for name, module in self.model.named_modules():
            if "backbone" in name:
                for param in module.parameters():
                    param.requires_grad = False
                found = True
        print("Encoder is frozen")
        if not found:
            raise AttributeError("No module with 'backbone' found in the model.")

    def unfreeze_encoder(self):
        """Unfreeze the backbone encoder parameters."""
        for name, module in self.model.named_modules():
            if "backbone" in name:
                for param in module.parameters():
                    param.requires_grad = True
        print("Encoder is unfrozen")

    def on_train_epoch_start(self):
        """Freeze/unfreeze backbone based on current epoch."""
        if self.current_epoch == 0:
            self.freeze_encoder()
        elif self.current_epoch == self.hparams.num_freeze_epoch:
            self.unfreeze_encoder()

    def forward(self, pixel_values, pixel_mask=None):
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input images
            pixel_mask: Pixel mask (optional)
            
        Returns:
            Model output
        """
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def _step(self, batch, batch_idx, stage: str):
        """
        Common step for training, validation, and testing.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            stage: Current stage ('train', 'val', or 'test')
            
        Returns:
            Calculated loss
        """
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask", None)
        # 各ターゲットをデバイスへ移動
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        # モデルはラベルを与えずに予測を出力（logits, pred_boxes）
        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs["loss"]
        loss_dict = outputs["loss_dict"]
        
        # ログ記録
        self.log(f"{stage}_loss_total", loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"{stage}_loss_{k}", v, prog_bar=(stage == "val"), sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        return self._step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Test loss
        """
        return self._step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and lr_scheduler
        """
        # パラメータをバックボーンとそれ以外で分割
        param_dict = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": self.hparams.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.lr_backbone,
            },
        ]
        optimizer = optim.AdamW(param_dict, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.optim_t_max, eta_min=self.hparams.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        } 