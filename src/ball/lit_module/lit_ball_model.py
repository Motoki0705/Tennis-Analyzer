"""
Generic LightningModule for Ball Detection Models
"""
import os
import random
from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import sigmoid_focal_loss

from src.common.lit_module.base import BaseLitModule

class LitBallModel(BaseLitModule):
    """
    A generic LightningModule for ball detection that can be configured
    with any model, loss function, and optimizer settings.
    
    The model, criterion, and optimizer/scheduler parameters are expected
    to be injected via a configuration framework like Hydra.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        logging_config: Dict[str, Any] = None,
    ):
        """
        Args:
            model (nn.Module): The neural network model to be trained.
            criterion (Dict[str, Any]): Parameters for the loss function.
            optimizer_config (Dict): Parameters for the optimizer (e.g., lr, weight_decay).
            scheduler_config (Dict): Parameters for the LR scheduler (e.g., warmup_epochs, max_epochs).
            logging_config (Dict): Configuration for logging and visualization.
        """
        # Default logging configuration
        if logging_config is None:
            logging_config = {
                "num_log_images": 4,
                "save_validation_outputs": True,
                "save_validation_images": True,
            }
        
        super().__init__(
            model=model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            criterion_config=criterion,
            logging_config=logging_config,
        )

    def compute_loss(self, batch, outputs, stage: str) -> torch.Tensor:
        """Compute loss for ball detection task."""
        frames, heatmaps, _, _ = batch
        logits = outputs
        
        # Handle different output formats from the model
        # For BallDetNet: logits = [B, N, H, W], heatmaps = [B, N, H, W]
        # For other models: logits = [B, H, W] or [B, C, H, W]
        if logits.dim() == 4 and heatmaps.dim() == 4:
            # BallDetNet format: [B, N, H, W] - reshape for loss calculation
            B, N, H, W = logits.shape
            logits = logits.view(B * N, H, W)
            heatmaps = heatmaps.view(B * N, H, W)
            
        # Ensure heatmaps and logits have the same dimensions for loss calculation
        if heatmaps.dim() == 3:
            heatmaps = heatmaps.unsqueeze(1)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
            
        # Calculate loss using focal loss
        loss = sigmoid_focal_loss(
            inputs=logits, 
            targets=heatmaps, 
            alpha=self.criterion_config.get("alpha", 1.0), 
            gamma=self.criterion_config.get("gamma", 2.0), 
            reduction=self.criterion_config.get("reduction", "mean")
        )
        
        return loss

    def compute_metrics(self, batch, outputs, stage: str) -> Dict[str, torch.Tensor]:
        """Compute metrics for ball detection task."""
        # Ball detection typically doesn't have standard accuracy metrics
        # Could add detection accuracy, precision, recall if needed
        return {}

    def validation_step(self, batch, batch_idx):
        """Validation step with visualization preparation."""
        # Prepare visualization data for the first batch
        if batch_idx == 0 and self.logging_config.get("save_validation_outputs", False):
            frames, heatmaps, _, _ = batch
            logits = self(frames)
            
            # Handle different output formats for visualization
            if logits.dim() == 4 and heatmaps.dim() == 4:
                # BallDetNet format: take middle frame for visualization
                middle_frame_idx = logits.size(1) // 2
                logits_vis = logits[:, middle_frame_idx].unsqueeze(1)
                heatmaps_vis = heatmaps[:, middle_frame_idx].unsqueeze(1)
                frames_vis = frames[:, middle_frame_idx]
            else:
                # Standard format: ensure channel dimension
                if logits.dim() == 3:
                    logits_vis = logits.unsqueeze(1)
                else:
                    logits_vis = logits
                if heatmaps.dim() == 3:
                    heatmaps_vis = heatmaps.unsqueeze(1)
                else:
                    heatmaps_vis = heatmaps
                frames_vis = frames
                
            preds_vis = torch.sigmoid(logits_vis)
            
            self._validation_outputs = {
                "frames": frames_vis.cpu(),
                "preds": preds_vis.cpu(),
                "targets": heatmaps_vis.cpu(),
                "batch": batch,
                "outputs": logits,
                "loss": None,  # Will be computed in common step
            }
            
        return super().validation_step(batch, batch_idx)

    def _save_validation_images(self):
        """Save validation images for ball detection visualization."""
        if not self._validation_outputs or not self.logger or not hasattr(self.logger, 'log_dir'):
            return

        log_dir = self.logger.log_dir
        save_dir = os.path.join(log_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        
        frames = self._validation_outputs["frames"]
        preds = self._validation_outputs["preds"]
        targets = self._validation_outputs["targets"]
        batch_size = frames.size(0)
        
        num_images = min(self.logging_config.get("num_log_images", 4), batch_size)
        selected_indices = random.sample(range(batch_size), num_images)

        for idx, batch_idx in enumerate(selected_indices):
            # Handle frame input format properly
            frame_img = frames[batch_idx, :3]  # Take first 3 channels (RGB)
            pred_heatmap = preds[batch_idx]
            target_heatmap = targets[batch_idx]

            # Ensure heatmaps have the right dimensions for visualization
            if pred_heatmap.dim() == 3:
                pred_heatmap = pred_heatmap[0]  # Take first channel
            if target_heatmap.dim() == 3:
                target_heatmap = target_heatmap[0]  # Take first channel

            # Create a grid: [Original Frame, Predicted Heatmap, Ground Truth Heatmap]
            combined_image = torch.cat([
                frame_img,
                pred_heatmap.repeat(3, 1, 1),  # Repeat grayscale heatmap for 3 channels
                target_heatmap.repeat(3, 1, 1)
            ], dim=2)  # Concatenate horizontally

            filename = os.path.join(save_dir, f"epoch_{self.current_epoch:03d}_sample_{idx:02d}.png")
            torchvision.utils.save_image(combined_image, filename)
