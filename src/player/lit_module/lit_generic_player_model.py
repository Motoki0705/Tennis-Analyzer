"""
Generic LightningModule for Player Detection Models
"""
from typing import Dict, Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class LitGenericPlayerModel(pl.LightningModule):
    """
    A generic LightningModule for player detection that can be configured
    with any model and optimizer settings.
    
    This module is designed to work with object detection models that follow
    the HuggingFace transformers interface (e.g., RT-DETR).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_params: Dict[str, Any],
        scheduler_params: Dict[str, Any],
        criterion: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model (nn.Module): The object detection model to be trained.
            optimizer_params (Dict): Parameters for the optimizer (e.g., lr, lr_backbone, weight_decay).
            scheduler_params (Dict): Parameters for the LR scheduler (e.g., T_max, eta_min, num_freeze_epoch).
            criterion (Dict, optional): Parameters for the loss function (currently unused as loss is handled by model).
        """
        super().__init__()
        
        # Save hyperparameters for reproducibility
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        
        # Store params directly for easier access
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.criterion = criterion or {}

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
        num_freeze_epoch = self.scheduler_params.get("num_freeze_epoch", 3)
        
        if self.current_epoch == 0:
            self.freeze_encoder()
        elif self.current_epoch == num_freeze_epoch:
            self.unfreeze_encoder()

    def forward(self, pixel_values, pixel_mask=None):
        """Forward pass through the model."""
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._common_step(batch, "test")

    def _common_step(self, batch, stage: str):
        """Common logic for training, validation, and test steps."""
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask", None)
        
        # Move labels to device
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        
        # Get model outputs
        outputs = self.model(
            pixel_values=pixel_values, 
            pixel_mask=pixel_mask, 
            labels=labels
        )

        loss = outputs["loss"]
        loss_dict = outputs["loss_dict"]
        
        # Log metrics
        self.log(
            f"{stage}_loss_total",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        for k, v in loss_dict.items():
            self.log(
                f"{stage}_loss_{k}",
                v,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(stage == "val"),
                logger=True,
                sync_dist=True,
            )
        
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Split parameters into backbone and non-backbone
        param_dict = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": self.optimizer_params.get("lr", 1e-4),
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.optimizer_params.get("lr_backbone", 1e-5),
            },
        ]
        
        optimizer = optim.AdamW(
            param_dict, 
            weight_decay=self.optimizer_params.get("weight_decay", 1e-4)
        )
        
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.scheduler_params.get("T_max", 50), 
            eta_min=self.scheduler_params.get("eta_min", 1e-6)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch"
            },
        }