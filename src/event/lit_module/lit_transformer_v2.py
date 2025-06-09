# event_detection_lit_module.py
# ------------------------------------------------------------
# LightningModule for hit / bounce event detection (multilabel)
# 2025-06-09  (modified to print sample predictions each epoch)
# ------------------------------------------------------------

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics import AUROC
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

# the actual backbone
from src.event.model.transformer_v2 import EventTransformerV2


class LitTransformerV2(pl.LightningModule):
    """
    Predicts two mutually–exclusive event flags — *hit* and *bounce* —
    for every time-step in a rally.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pose_dim: int = 51,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        no_hit_weight: float = 0.01,  # [0,0]
        hit_weight: float = 1.0,      # [1,0]
        bounce_weight: float = 1.0,   # [0,1]
        clarity_weight: float = 0.02,
    ):
        super().__init__()
        self.save_hyperparameters()

        # network
        self.model = EventTransformerV2(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pose_dim=pose_dim,
        )

        # loss weights
        self.no_hit_weight = no_hit_weight
        self.hit_weight = hit_weight
        self.bounce_weight = bounce_weight
        self.clarity_weight = clarity_weight

        # metrics
        self.train_f1 = MultilabelF1Score(num_labels=2)
        self.val_f1 = MultilabelF1Score(num_labels=2)
        self.val_precision = MultilabelPrecision(num_labels=2)
        self.val_recall = MultilabelRecall(num_labels=2)
        self.val_auroc = AUROC(task="multilabel", num_labels=2)

        # ────────── buffers for sample printing ──────────
        self.train_samples = []  # list[tuple[p, t]]
        self.val_samples: list[tuple[torch.Tensor, torch.Tensor]] = []

    # ──────────────────────────── forward ───────────────────────────

    def forward(
        self,
        ball_features,
        player_bbox_features,
        player_pose_features,
        court_features,
    ):
        return self.model(
            ball_features, player_bbox_features, player_pose_features, court_features
        )

    # ──────────────────────────── loss ──────────────────────────────

    def custom_event_loss(self, logits, targets):
        """Binary-cross-entropy + sample weights + clarity regulariser."""
        # flatten
        logits = logits.view(-1, 2)
        targets = targets.view(-1, 2)

        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # sample weights
        w = torch.ones_like(targets)
        mask_no    = (targets[:, 0] == 0) & (targets[:, 1] == 0)
        mask_hit   = (targets[:, 0] != 1) & (targets[:, 1] == 0)
        mask_bnc   = (targets[:, 0] == 0) & (targets[:, 1] != 1)
        w[mask_no]  = self.no_hit_weight
        w[mask_hit] = self.hit_weight
        w[mask_bnc] = self.bounce_weight
        w_bce = (bce * w).mean()

        # clarity: encourage mutually-exclusive predictions
        probs = torch.sigmoid(logits)
        clarity = (probs[:, 0] * probs[:, 1]).mean()

        return w_bce + self.clarity_weight * clarity

    # ─────────────────────── util: collect samples ──────────────────

    @torch.no_grad()
    def _collect_samples(
        self,
        probs: torch.Tensor,      # [B, S, 2] after sigmoid
        targets: torch.Tensor,    # [B, S, 2] soft labels (Gaussian)
        buffer: list,
        max_samples: int = 5,
        peak_thresh: float = 0.30,  # 0.0〜1.0: 「イベントらしさ」の下限
    ):
        """
        Gaussian ラベルのピーク (局所最大) を抽出して保存する。

        条件:
          • target が左右より大きい (local maximum)
          • target ≥ peak_thresh
          • buffer が max_samples に達するまで
        """

        if len(buffer) >= max_samples:
            return

        B, S, C = targets.shape  # C=2 (hit, bounce)

        # ──ユーティリティ: 1Dテンソルのローカルピーク検出──
        def _find_peaks_1d(seq: torch.Tensor) -> torch.Tensor:
            """返り値: peak のインデックス (1D LongTensor)"""
            left  = torch.cat([seq.new_tensor([-1.0]), seq[:-1]])
            right = torch.cat([seq[1:], seq.new_tensor([-1.0])])
            return ((seq > left) & (seq >= right) & (seq >= peak_thresh)).nonzero(as_tuple=False).squeeze(-1)

        # ──走査──
        for b in range(B):
            if len(buffer) >= max_samples:
                break
            for c in range(C):                      # 0:hit, 1:bounce
                tgt_seq = targets[b, :, c]          # [S]
                peak_idx = _find_peaks_1d(tgt_seq)  # [...,]
                for idx in peak_idx:
                    if len(buffer) >= max_samples:
                        break
                    buffer.append((
                        probs  [b, idx].detach().cpu(),   # [2] predicted probs
                        targets[b, idx].detach().cpu(),   # [2] soft GT at peak
                    ))

    # ──────────────────────────── hooks ─────────────────────────────

    # clear buffers
    def on_train_epoch_start(self):
        self.train_samples = []

    def on_validation_epoch_start(self):
        self.val_samples = []

    # pretty print
    def _pretty_print(self, samples, stage):
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


    def on_train_epoch_end(self):
        if self.global_rank == 0:
            self._pretty_print(self.train_samples, "train")

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            self._pretty_print(self.val_samples, "val")

    # ─────────────────────────── core step ──────────────────────────

    def _step(self, batch, stage):
        (
            ball_features,
            player_bbox_features,
            player_pose_features,
            court_features,
            targets,
            _,
        ) = batch

        logits = self(
            ball_features, player_bbox_features, player_pose_features, court_features
        )
        loss = self.custom_event_loss(logits, targets)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        flat_preds = preds.view(-1, 2)
        flat_targets = targets.view(-1, 2).long()

        if stage == "train":
            self.train_f1(flat_preds, flat_targets)
            self.log(
                "train_f1",
                self.train_f1,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.val_f1(flat_preds, flat_targets)
            self.val_precision(flat_preds, flat_targets)
            self.val_recall(flat_preds, flat_targets)
            self.val_auroc(probs.view(-1, 2), flat_targets)

            self.log("val_f1", self.val_f1, prog_bar=True)
            self.log("val_precision", self.val_precision)
            self.log("val_recall", self.val_recall)
            self.log("val_auroc", self.val_auroc)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss, probs, targets

    # ─────────────────────────── steps ──────────────────────────────

    def training_step(self, batch, batch_idx):
        loss, probs, targets = self._step(batch, "train")
        if self.global_rank == 0:
            self._collect_samples(probs, targets, self.train_samples)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, targets = self._step(batch, "val")
        if self.global_rank == 0:
            self._collect_samples(probs, targets, self.val_samples)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, "test")
        return loss

    # ─────────────────────── optim / sched ──────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: min(1.0, float(e + 1) / self.hparams.warmup_epochs),
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 1e-2,
        )
        scheduler = SequentialLR(
            optimizer, [warmup, cosine], milestones=[self.hparams.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
