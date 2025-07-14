"""
Generic LightningModule for court detection models.
Consolidates functionality from all court-specific LitModules.
"""
from typing import Dict, Any, Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR


class LitGenericCourtModel(pl.LightningModule):
    """
    Generic LightningModule for court detection that accepts any model architecture.
    
    Consolidates common functionality from:
    - LitLiteTracknetFocal
    - LitSwinUNet
    - LitVitUNet
    - LitFPN
    
    Input: Frames [B, C, H, W]
    Output: Heatmaps [B, out_channels, H, W]
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        scheduler_params: Dict[str, Any],
        accuracy_threshold: float = 5.0,
        num_log_images: int = 4,
        use_peak_valley_heatmaps: bool = False,
    ):
        """
        Initialize generic court model.
        
        Args:
            model: The court detection model (LiteTrackNet, SwinUNet, VitUNet, FPN)
            criterion: Loss function configuration
            optimizer_params: Optimizer configuration  
            scheduler_params: Learning rate scheduler configuration
            accuracy_threshold: Distance threshold for accuracy calculation (pixels)
            num_log_images: Number of images to log during validation
            use_peak_valley_heatmaps: Whether to use peak+valley heatmaps format
        """
        super().__init__()
        
        # Store the injected model
        self.model = model
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
        
        # Store configuration
        self.criterion_config = criterion
        self.optimizer_config = optimizer_params
        self.scheduler_config = scheduler_params

    def _find_heatmap_peaks(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Find peak coordinates from batch of heatmaps.
        
        Args:
            heatmaps: Input heatmaps [B, K, H, W]
            
        Returns:
            Peak coordinates [B, K, 2] in [x, y] format
        """
        batch_size, num_kpts, h, w = heatmaps.shape
        flat_heatmaps = heatmaps.view(batch_size, num_kpts, -1)
        _, max_indices = torch.max(flat_heatmaps, dim=2)
        peak_y = max_indices // w
        peak_x = max_indices % w
        peak_coords = torch.stack([peak_x, peak_y], dim=2)
        return peak_coords.float()

    def _calculate_accuracy(self, pred_coords: torch.Tensor, gt_keypoints: torch.Tensor) -> torch.Tensor:
        """
        Calculate accuracy from predicted coordinates and ground truth.
        
        Args:
            pred_coords: Predicted coordinates [B, K, 2]
            gt_keypoints: Ground truth keypoints [B, K, 3] (x, y, visibility)
            
        Returns:
            Accuracy score as tensor
        """
        gt_coords = gt_keypoints[:, :, :2]
        visibility = gt_keypoints[:, :, 2]
        distances = torch.linalg.norm(pred_coords - gt_coords, dim=2)
        correct_preds = distances <= self.hparams.accuracy_threshold
        num_correct = torch.sum(correct_preds * visibility)
        num_visible = torch.sum(visibility)
        
        if num_visible == 0:
            return torch.tensor(0.0, device=self.device)
            
        accuracy = num_correct / num_visible
        return accuracy

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def _common_step(self, batch, stage: str):
        """Common processing step for train/val/test."""
        frames, heatmaps, scaled_keypoints = batch
        logits = self(frames)
        
        # Handle peak+valley heatmaps format conversion if needed
        if self.hparams.use_peak_valley_heatmaps:
            target_heatmaps = (heatmaps + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
        else:
            target_heatmaps = heatmaps  # Already [0, 1]
        
        # Calculate loss using focal loss
        loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=target_heatmaps,
            alpha=self.criterion_config.get('alpha', 1.0),
            gamma=self.criterion_config.get('gamma', 2.0),
            reduction=self.criterion_config.get('reduction', 'mean'),
        )

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Calculate accuracy for validation and test
        if stage in ["val", "test"]:
            pred_heatmaps = torch.sigmoid(logits)
            pred_coords = self._find_heatmap_peaks(pred_heatmaps)
            accuracy = self._calculate_accuracy(pred_coords, scaled_keypoints)
            self.log(f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
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

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer_class = getattr(torch.optim, self.optimizer_config.get('class', 'AdamW'))
        optimizer = optimizer_class(
            self.parameters(),
            lr=self.optimizer_config.get('lr', 1e-4),
            weight_decay=self.optimizer_config.get('weight_decay', 1e-4),
            **{k: v for k, v in self.optimizer_config.items() if k not in ['class', 'lr', 'weight_decay']}
        )

        # Create learning rate scheduler if specified
        if 'scheduler' not in self.scheduler_config:
            return optimizer
            
        scheduler_type = self.scheduler_config['scheduler']
        
        if scheduler_type == 'cosine_with_warmup':
            # Warmup + Cosine scheduler (default for court models)
            warmup_epochs = self.scheduler_config.get('warmup_epochs', 1)
            max_epochs = self.scheduler_config.get('max_epochs', 50)
            
            # Warmup scheduler
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: float(epoch + 1) / float(warmup_epochs),
            )

            # Cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=self.optimizer_config.get('lr', 1e-4) * self.scheduler_config.get('eta_min_factor', 1e-2),
            )

            # Sequential scheduler (Warmup → Cosine)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            
        elif scheduler_type == 'cosine':
            # Simple cosine annealing
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('max_epochs', 50),
                eta_min=self.optimizer_config.get('lr', 1e-4) * self.scheduler_config.get('eta_min_factor', 1e-2),
            )
        else:
            # Fallback to no scheduler
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def freeze_encoder(self):
        """Freeze encoder weights (for models that support this)."""
        if hasattr(self.model, 'freeze_encoder'):
            self.model.freeze_encoder()
            print("Encoder weights frozen.")
        else:
            print("Model does not support encoder freezing.")

    def unfreeze_encoder(self):
        """Unfreeze encoder weights (for models that support this)."""
        if hasattr(self.model, 'unfreeze_encoder'):
            self.model.unfreeze_encoder()
            print("Encoder weights unfrozen.")
        else:
            print("Model does not support encoder unfreezing.")