"""
LightningModule for Event Detection (Hit/Bounce Detection)
Standardized using the common base class for unified training interface.
"""

import torch
import torch.nn.functional as F
from torchmetrics import AUROC
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from typing import Dict, Any

from src.common.lit_module.base import BaseLitModule
from src.event.model.transformer_v2 import EventTransformerV2


class LitTransformerV2(BaseLitModule):
    """
    Predicts two mutually–exclusive event flags — *hit* and *bounce* —
    for every time-step in a rally.
    """

    def __init__(
        self,
        model: EventTransformerV2,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        criterion_config: Dict[str, Any] = None,
        logging_config: Dict[str, Any] = None,
    ):
        """
        Initialize event detection model.
        
        Args:
            model: The EventTransformerV2 model
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration
            criterion_config: Loss function configuration
            logging_config: Logging and visualization configuration
        """
        # Default criterion configuration
        if criterion_config is None:
            criterion_config = {
                "no_hit_weight": 0.01,    # [0,0]
                "hit_weight": 1.0,        # [1,0]
                "bounce_weight": 1.0,     # [0,1]
                "clarity_weight": 0.02,
            }
        
        # Default logging configuration
        if logging_config is None:
            logging_config = {
                "save_validation_outputs": True,
                "save_validation_images": False,
                "max_samples": 5,
                "peak_thresh": 0.30,
            }
        
        super().__init__(
            model=model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            criterion_config=criterion_config,
            logging_config=logging_config,
        )

        # Initialize metrics
        self.train_f1 = MultilabelF1Score(num_labels=2)
        self.val_f1 = MultilabelF1Score(num_labels=2)
        self.val_precision = MultilabelPrecision(num_labels=2)
        self.val_recall = MultilabelRecall(num_labels=2)
        self.val_auroc = AUROC(task="multilabel", num_labels=2)

        # Sample collection buffers
        self.train_samples = []
        self.val_samples = []

    def forward(self, batch):
        """Forward pass through the event detection model."""
        ball_features, player_bbox_features, player_pose_features, court_features, targets, _ = batch
        return self.model(
            ball_features, player_bbox_features, player_pose_features, court_features
        )

    def compute_loss(self, batch, outputs, stage: str) -> torch.Tensor:
        """Compute loss for event detection task."""
        ball_features, player_bbox_features, player_pose_features, court_features, targets, _ = batch
        logits = outputs
        
        # Binary-cross-entropy + sample weights + clarity regularizer
        logits_flat = logits.view(-1, 2)
        targets_flat = targets.view(-1, 2)

        # BCE
        bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="none")

        # Sample weights
        w = torch.ones_like(targets_flat)
        mask_no = (targets_flat[:, 0] == 0) & (targets_flat[:, 1] == 0)
        mask_hit = (targets_flat[:, 0] == 1) & (targets_flat[:, 1] == 0)
        mask_bounce = (targets_flat[:, 0] == 0) & (targets_flat[:, 1] == 1)
        
        w[mask_no] = self.criterion_config.get("no_hit_weight", 0.01)
        w[mask_hit] = self.criterion_config.get("hit_weight", 1.0)
        w[mask_bounce] = self.criterion_config.get("bounce_weight", 1.0)
        
        w_bce = (bce * w).mean()

        # Clarity: encourage mutually-exclusive predictions
        probs = torch.sigmoid(logits_flat)
        clarity = (probs[:, 0] * probs[:, 1]).mean()
        clarity_weight = self.criterion_config.get("clarity_weight", 0.02)

        return w_bce + clarity_weight * clarity

    def compute_metrics(self, batch, outputs, stage: str) -> Dict[str, torch.Tensor]:
        """Compute metrics for event detection task."""
        ball_features, player_bbox_features, player_pose_features, court_features, targets, _ = batch
        logits = outputs
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        flat_preds = preds.view(-1, 2)
        flat_targets = targets.view(-1, 2).long()

        if stage == "train":
            self.train_f1(flat_preds, flat_targets)
            return {"f1": self.train_f1}
        else:
            self.val_f1(flat_preds, flat_targets)
            self.val_precision(flat_preds, flat_targets)
            self.val_recall(flat_preds, flat_targets)
            self.val_auroc(probs.view(-1, 2), flat_targets)

            return {
                "f1": self.val_f1,
                "precision": self.val_precision,
                "recall": self.val_recall,
                "auroc": self.val_auroc,
            }

    @torch.no_grad()
    def _collect_samples(
        self,
        probs: torch.Tensor,      # [B, S, 2] after sigmoid
        targets: torch.Tensor,    # [B, S, 2] soft labels (Gaussian)
        buffer: list,
    ):
        """Collect sample predictions for logging."""
        max_samples = self.logging_config.get("max_samples", 5)
        peak_thresh = self.logging_config.get("peak_thresh", 0.30)

        if len(buffer) >= max_samples:
            return

        B, S, C = targets.shape  # C=2 (hit, bounce)

        # Utility: 1D tensor local peak detection
        def _find_peaks_1d(seq: torch.Tensor) -> torch.Tensor:
            """Return peak indices (1D LongTensor)"""
            left = torch.cat([seq.new_tensor([-1.0]), seq[:-1]])
            right = torch.cat([seq[1:], seq.new_tensor([-1.0])])
            return ((seq > left) & (seq >= right) & (seq >= peak_thresh)).nonzero(as_tuple=False).squeeze(-1)

        # Scan for peaks
        for b in range(B):
            if len(buffer) >= max_samples:
                break
            for c in range(C):  # 0:hit, 1:bounce
                tgt_seq = targets[b, :, c]  # [S]
                peak_idx = _find_peaks_1d(tgt_seq)  # [...,]
                for idx in peak_idx:
                    if len(buffer) >= max_samples:
                        break
                    buffer.append((
                        probs[b, idx].detach().cpu(),    # [2] predicted probs
                        targets[b, idx].detach().cpu(),  # [2] soft GT at peak
                    ))

    def training_step(self, batch, batch_idx):
        """Training step with sample collection."""
        loss = super().training_step(batch, batch_idx)
        
        # Collect samples for logging
        if self.global_rank == 0 and self.logging_config.get("save_validation_outputs", False):
            _, _, _, _, targets, _ = batch
            logits = self(batch)
            probs = torch.sigmoid(logits)
            self._collect_samples(probs, targets, self.train_samples)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with sample collection."""
        loss = super().validation_step(batch, batch_idx)
        
        # Collect samples for logging
        if self.global_rank == 0 and self.logging_config.get("save_validation_outputs", False):
            _, _, _, _, targets, _ = batch
            logits = self(batch)
            probs = torch.sigmoid(logits)
            self._collect_samples(probs, targets, self.val_samples)
        
        return loss

    def on_train_epoch_start(self):
        """Clear sample buffers at the start of training epoch."""
        super().on_train_epoch_start()
        self.train_samples = []

    def on_validation_epoch_start(self):
        """Clear sample buffers at the start of validation epoch."""
        super().on_validation_epoch_start() 
        self.val_samples = []

    def on_train_epoch_end(self):
        """Pretty print training samples."""
        super().on_train_epoch_end()
        if self.global_rank == 0:
            self._pretty_print(self.train_samples, "train")

    def on_validation_epoch_end(self):
        """Pretty print validation samples."""
        super().on_validation_epoch_end()
        if self.global_rank == 0:
            self._pretty_print(self.val_samples, "val")

    def _pretty_print(self, samples, stage):
        """Pretty print sample predictions."""
        if not samples:
            return
        bar = "=" * 52
        print(f"\n{bar}\nEpoch {self.current_epoch} — {stage} peak predictions")
        for i, (p, t) in enumerate(samples, 1):
            ph, pb = p.tolist()
            th, tb = t.tolist()
            print(
                f"[{i}] pred_hit={ph:5.2f}  pred_bounce={pb:5.2f} | "
                f"GT → hit={th:5.2f}  bounce={tb:5.2f}"
            )
        print(f"{bar}\n", flush=True)
