"""
Generic LightningModule for court detection models.
Consolidates functionality from all court-specific LitModules.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision

from src.common.lit_module.base import BaseLitModule


class LitCourtModel(BaseLitModule):
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
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        accuracy_threshold: float = 5.0,
        use_peak_valley_heatmaps: bool = False,
        logging_config: Dict[str, Any] = None,
    ):
        """
        Initialize generic court model.
        
        Args:
            model: The court detection model (LiteTrackNet, SwinUNet, VitUNet, FPN)
            criterion: Loss function configuration
            optimizer_config: Optimizer configuration  
            scheduler_config: Learning rate scheduler configuration
            accuracy_threshold: Distance threshold for accuracy calculation (pixels)
            use_peak_valley_heatmaps: Whether to use peak+valley heatmaps format
            logging_config: Configuration for logging and visualization
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
        
        # Court-specific configuration
        self.accuracy_threshold = accuracy_threshold
        self.use_peak_valley_heatmaps = use_peak_valley_heatmaps

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

    def compute_loss(self, batch, outputs, stage: str) -> torch.Tensor:
        """Compute loss for court detection task."""
        frames, heatmaps, scaled_keypoints = batch
        logits = outputs
        
        # Handle peak+valley heatmaps format conversion if needed
        if self.use_peak_valley_heatmaps:
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
        
        return loss

    def compute_metrics(self, batch, outputs, stage: str) -> Dict[str, torch.Tensor]:
        """Compute metrics for court detection task."""
        frames, heatmaps, scaled_keypoints = batch
        logits = outputs
        
        # Calculate accuracy for validation and test
        pred_heatmaps = torch.sigmoid(logits)
        pred_coords = self._find_heatmap_peaks(pred_heatmaps)
        accuracy = self._calculate_accuracy(pred_coords, scaled_keypoints)
        
        return {"accuracy": accuracy}

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