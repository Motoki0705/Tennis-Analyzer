"""
Base Lightning Module providing common functionality for all model types.
This module provides standardized interfaces for training, validation, and testing
across all sports analysis tasks (ball, court, event, player, pose).
"""
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

log = logging.getLogger(__name__)


class BaseLitModule(pl.LightningModule, ABC):
    """
    Base LightningModule providing common functionality for all model types.
    
    This class standardizes:
    - Optimizer and scheduler configuration
    - Hyperparameter handling
    - Common training/validation patterns
    - Logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        criterion_config: Optional[Dict[str, Any]] = None,
        logging_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base lightning module.
        
        Args:
            model: The neural network model
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration
            criterion_config: Loss function configuration
            logging_config: Logging and visualization configuration
        """
        super().__init__()
        
        # Save hyperparameters (exclude model to avoid serialization issues)
        self.save_hyperparameters(ignore=['model'])
        
        # Core components
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.criterion_config = criterion_config or {}
        self.logging_config = logging_config or {}
        
        # Validation outputs for logging (subclasses can use this)
        self._validation_outputs = None

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)

    @abstractmethod
    def compute_loss(self, batch: Any, outputs: Any, stage: str) -> torch.Tensor:
        """
        Compute loss for the given batch and model outputs.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            stage: Training stage ('train', 'val', 'test')
            
        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def compute_metrics(self, batch: Any, outputs: Any, stage: str) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for the given batch and model outputs.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            stage: Training stage ('train', 'val', 'test')
            
        Returns:
            Dictionary of metric tensors
        """
        pass

    def _common_step(self, batch: Any, stage: str):
        """Common logic for training, validation, and test steps."""
        # Get model outputs
        outputs = self(batch)
        
        # Compute loss
        loss = self.compute_loss(batch, outputs, stage)
        
        # Log loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        # Compute and log metrics
        if stage in ["val", "test"]:
            metrics = self.compute_metrics(batch, outputs, stage)
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"{stage}_{metric_name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(stage == "val"),
                    logger=True,
                    sync_dist=True,
                )
        
        # Store validation outputs for visualization
        if stage == "val" and self.logging_config.get("save_validation_outputs", False):
            self._validation_outputs = {
                "batch": batch,
                "outputs": outputs,
                "loss": loss,
            }
        
        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._common_step(batch, "test")

    def on_validation_epoch_end(self):
        """Hook called at the end of validation epoch."""
        if self._validation_outputs and self.logging_config.get("save_validation_images", False):
            self._save_validation_images()
        
        # Clear validation outputs
        self._validation_outputs = None

    def _save_validation_images(self):
        """Save validation images (to be implemented by subclasses if needed)."""
        log.debug("Validation image saving not implemented for this module type")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer configuration
        optimizer_type = self.optimizer_config.get("type", "AdamW")
        lr = self.optimizer_config.get("lr", 1e-4)
        weight_decay = self.optimizer_config.get("weight_decay", 1e-4)
        
        # Handle different parameter groups (e.g., backbone vs non-backbone)
        param_groups = self._get_parameter_groups()
        
        # Create optimizer
        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        elif optimizer_type == "SGD":
            momentum = self.optimizer_config.get("momentum", 0.9)
            optimizer = torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Configure scheduler if specified
        scheduler_config = self._configure_scheduler(optimizer)
        
        if scheduler_config:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_config,
            }
        else:
            return optimizer

    def _get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer.
        Can be overridden by subclasses to implement different learning rates
        for different parts of the model (e.g., backbone vs head).
        """
        lr = self.optimizer_config.get("lr", 1e-4)
        return [{"params": self.parameters(), "lr": lr}]

    def _configure_scheduler(self, optimizer) -> Optional[Dict[str, Any]]:
        """Configure learning rate scheduler."""
        scheduler_type = self.scheduler_config.get("type", None)
        
        if not scheduler_type:
            return None
        
        if scheduler_type == "cosine_with_warmup":
            return self._create_cosine_warmup_scheduler(optimizer)
        elif scheduler_type == "cosine":
            return self._create_cosine_scheduler(optimizer)
        elif scheduler_type == "step":
            return self._create_step_scheduler(optimizer)
        else:
            log.warning(f"Unknown scheduler type: {scheduler_type}")
            return None

    def _create_cosine_warmup_scheduler(self, optimizer) -> Dict[str, Any]:
        """Create cosine annealing scheduler with warmup."""
        warmup_epochs = self.scheduler_config.get("warmup_epochs", 5)
        max_epochs = self.scheduler_config.get("max_epochs", 100)
        eta_min_factor = self.scheduler_config.get("eta_min_factor", 1e-2)
        
        base_lr = self.optimizer_config.get("lr", 1e-4)
        
        # Warmup scheduler
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=base_lr * eta_min_factor,
        )
        
        # Sequential scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

    def _create_cosine_scheduler(self, optimizer) -> Dict[str, Any]:
        """Create simple cosine annealing scheduler."""
        max_epochs = self.scheduler_config.get("max_epochs", 100)
        eta_min_factor = self.scheduler_config.get("eta_min_factor", 1e-2)
        base_lr = self.optimizer_config.get("lr", 1e-4)
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=base_lr * eta_min_factor,
        )
        
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

    def _create_step_scheduler(self, optimizer) -> Dict[str, Any]:
        """Create step learning rate scheduler."""
        step_size = self.scheduler_config.get("step_size", 30)
        gamma = self.scheduler_config.get("gamma", 0.1)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

    def freeze_encoder(self):
        """Freeze encoder parameters (to be implemented by subclasses)."""
        log.warning("Encoder freezing not implemented for this module type")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (to be implemented by subclasses)."""
        log.warning("Encoder unfreezing not implemented for this module type")