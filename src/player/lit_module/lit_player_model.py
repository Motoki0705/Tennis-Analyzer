"""
Generic LightningModule for Player Detection Models
"""
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn

from src.common.lit_module.base import BaseLitModule


class LitPlayerModel(BaseLitModule):
    """
    A generic LightningModule for player detection that can be configured
    with any model and optimizer settings.
    
    This module is designed to work with object detection models that follow
    the HuggingFace transformers interface (e.g., RT-DETR).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        criterion: Optional[Dict[str, Any]] = None,
        logging_config: Dict[str, Any] = None,
        num_freeze_epochs: int = 3,
    ):
        """
        Args:
            model (nn.Module): The object detection model to be trained.
            optimizer_config (Dict): Parameters for the optimizer (e.g., lr, lr_backbone, weight_decay).
            scheduler_config (Dict): Parameters for the LR scheduler (e.g., T_max, eta_min).
            criterion (Dict, optional): Parameters for the loss function (currently unused as loss is handled by model).
            logging_config (Dict): Configuration for logging and visualization.
            num_freeze_epochs (int): Number of epochs to freeze backbone encoder.
        """
        # Default logging configuration
        if logging_config is None:
            logging_config = {
                "save_validation_outputs": False,
                "save_validation_images": False,
            }
        
        super().__init__(
            model=model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            criterion_config=criterion or {},
            logging_config=logging_config,
        )
        
        # Player detection specific configuration
        self.num_freeze_epochs = num_freeze_epochs

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
        elif self.current_epoch == self.num_freeze_epochs:
            self.unfreeze_encoder()

    def forward(self, batch):
        """Forward pass through the model."""
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask", None)
        labels = batch.get("labels", None)
        
        # Move labels to device if provided
        if labels is not None:
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]
        
        return self.model(
            pixel_values=pixel_values, 
            pixel_mask=pixel_mask, 
            labels=labels
        )

    def compute_loss(self, batch, outputs, stage: str) -> torch.Tensor:
        """Compute loss for player detection task."""
        # For detection models, loss is computed internally by the model
        return outputs["loss"]

    def compute_metrics(self, batch, outputs, stage: str) -> Dict[str, torch.Tensor]:
        """Compute metrics for player detection task."""
        metrics = {}
        
        # Log detailed loss components
        if "loss_dict" in outputs:
            for k, v in outputs["loss_dict"].items():
                metrics[f"loss_{k}"] = v
        
        return metrics

    def _get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for player detection with different learning rates
        for backbone and non-backbone parameters.
        """
        lr = self.optimizer_config.get("lr", 1e-4)
        lr_backbone = self.optimizer_config.get("lr_backbone", 1e-5)
        
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]