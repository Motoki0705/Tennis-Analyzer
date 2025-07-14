"""
Generic LightningModule for Ball Detection Models
"""
import os
import random
from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR

class LitGenericBallModel(pl.LightningModule):
    """
    A generic LightningModule for ball detection that can be configured
    with any model, loss function, and optimizer settings.
    
    The model, criterion, and optimizer/scheduler parameters are expected
    to be injected via a configuration framework like Hydra.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_params: Dict[str, Any],
        scheduler_params: Dict[str, Any],
        num_log_images: int = 4,
    ):
        """
        Args:
            model (nn.Module): The neural network model to be trained.
            criterion (nn.Module): The loss function.
            optimizer_params (Dict): Parameters for the optimizer (e.g., lr, weight_decay).
            scheduler_params (Dict): Parameters for the LR scheduler (e.g., warmup_epochs, max_epochs).
            num_log_images (int): Number of images to log during validation.
        """
        super().__init__()
        
        # This saves all the arguments passed to __init__ into self.hparams
        # It allows us to access them with self.hparams.model, self.hparams.criterion, etc.
        # Importantly, it makes the configuration reproducible from a checkpoint.
        self.save_hyperparameters(ignore=['model', 'criterion'])

        self.model = model
        self.criterion = criterion
        
        # Store params directly for easier access
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        
        # For logging validation images
        self._val_batch_outputs_for_log = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Log the first batch for visualization
        if batch_idx == 0:
            frames, heatmaps, _, _ = batch
            logits = self(frames)
            
            # Ensure logits and heatmaps have a channel dimension
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            if heatmaps.dim() == 3:
                heatmaps = heatmaps.unsqueeze(1)
                
            preds = torch.sigmoid(logits)
            
            self._val_batch_outputs_for_log = {
                "frames": frames.cpu(),
                "preds": preds.cpu(),
                "targets": heatmaps.cpu()
            }
            
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._common_step(batch, "test")

    def _common_step(self, batch, stage: str):
        """Common logic for training, validation, and test steps."""
        frames, heatmaps, _, _ = batch
        logits = self(frames)
        
        # Ensure heatmaps and logits have the same dimensions for loss calculation
        if heatmaps.dim() == 3:
            heatmaps = heatmaps.unsqueeze(1)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
            
        # Calculate loss using the injected criterion
        loss = self.criterion(logits, heatmaps)
        
        preds = torch.sigmoid(logits)
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_iou", masked_iou, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        """Log a grid of validation images at the end of the validation epoch."""
        if self._val_batch_outputs_for_log is None or not self.logger or not hasattr(self.logger, 'log_dir'):
            return

        log_dir = self.logger.log_dir
        save_dir = os.path.join(log_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        
        frames = self._val_batch_outputs_for_log["frames"]
        preds = self._val_batch_outputs_for_log["preds"]
        targets = self._val_batch_outputs_for_log["targets"]
        batch_size = frames.size(0)
        
        num_images = min(self.hparams.num_log_images, batch_size)
        selected_indices = random.sample(range(batch_size), num_images)

        for idx, batch_idx in enumerate(selected_indices):
            # Assuming input is [C*T, H, W], take the first 3 channels for visualization
            frame_img = frames[batch_idx, :3] 
            pred_heatmap = preds[batch_idx]
            target_heatmap = targets[batch_idx]

            # Create a grid: [Original Frame, Predicted Heatmap, Ground Truth Heatmap]
            combined_image = torch.cat([
                frame_img,
                pred_heatmap.repeat(3, 1, 1), # Repeat grayscale heatmap for 3 channels
                target_heatmap.repeat(3, 1, 1)
            ], dim=2) # Concatenate vertically

            filename = os.path.join(save_dir, f"epoch_{self.current_epoch:03d}_sample_{idx:02d}.png")
            torchvision.utils.save_image(combined_image, filename)

        # Clear the stored outputs
        self._val_batch_outputs_for_log = None

    @staticmethod
    def compute_masked_iou(
        preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Computes Intersection over Union (IoU) for binary masks."""
        gt_mask = targets > threshold
        pred_mask = preds > threshold
        
        intersection = (gt_mask & pred_mask).float().sum(dim=(1, 2, 3))
        union = (gt_mask | pred_mask).float().sum(dim=(1, 2, 3))
        
        # Add a small epsilon to avoid division by zero
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou.mean()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_params.get("lr", 1e-3),
            weight_decay=self.optimizer_params.get("weight_decay", 1e-4),
        )
        
        warmup_epochs = self.scheduler_params.get("warmup_epochs", 1)
        max_epochs = self.scheduler_params.get("max_epochs", 50)

        # Linear warmup
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=self.optimizer_params.get("lr", 1e-3) * 1e-2
        )
        
        # Chain schedulers
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
