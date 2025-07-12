"""
LiteTrackNet用LightningModule with Focal Loss.
通常のヒートマップと、ピーク・谷を持つヒートマップの両方に対応。
"""
from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR # LambdaLRもインポート
from torchmetrics.classification import BinaryJaccardIndex
import torchvision

from src.court.models.lite_tracknet import LiteTrackNet


class LitLiteTracknetFocal(pl.LightningModule):
    """
    LiteTrackNet用LightningModule with Focal Loss
    
    入力: Frames [B, C, H, W]
    出力: Heatmaps [B, out_channels, H, W]
    """

    def __init__(
        self,
        # モデル構成パラメータ
        in_channels: int = 3,
        out_channels: int = 15,
        
        # 学習パラメータ
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
        
        # Focal Loss パラメータ
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,

        # 評価パラメータ
        accuracy_threshold: int = 5,
        
        # --- ここからが追加箇所 ---
        # ピーク + 谷のヒートマップを使用するかどうかのフラグ
        use_peak_valley_heatmaps: bool = False,
        # --- ここまでが追加箇所 ---
    ):
        super().__init__()
        
        # モデルの初期化
        self.model = LiteTrackNet(
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        # ハイパーパラメータの保存
        # これにより、self.hparams.lr のようにアクセス可能になる
        self.save_hyperparameters()

    def _find_heatmap_peaks(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        バッチ処理されたヒートマップからピーク座標を見つけるヘルパー関数。
        (B, K, H, W) -> (B, K, 2) [x, y]
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
        予測座標と教師データからAccuracyを計算する。
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
        """順伝播処理"""
        return self.model(x)

    # --- _common_step にロジックを集約し、validation_stepを簡潔にする ---
    def _common_step(self, batch, stage: str):
        """共通処理ステップ"""
        frames, heatmaps, scaled_keypoints = batch
        logits = self(frames)
        
        # --- ここからが修正箇所 ---
        # `use_peak_valley_heatmaps`がTrueの場合、ターゲットを[-1, 1]から[0, 1]に変換
        if self.hparams.use_peak_valley_heatmaps:
            target_heatmaps = (heatmaps + 1.0) / 2.0
        else:
            target_heatmaps = heatmaps # 元々[0, 1]なので何もしない
        # --- ここまでが修正箇所 ---
        
        # Focal Lossで損失を計算
        loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=target_heatmaps, # 変換後のターゲットを使用
            alpha=self.hparams.focal_alpha,
            gamma=self.hparams.focal_gamma,
            reduction="mean",
        )

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Accuracyは検証/テスト時のみ計算
        if stage in ["val", "test"]:
            pred_heatmaps = torch.sigmoid(logits)
            pred_coords = self._find_heatmap_peaks(pred_heatmaps)
            accuracy = self._calculate_accuracy(pred_coords, scaled_keypoints)
            self.log(f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return loss

    def training_step(self, batch, batch_idx):
        """学習ステップ"""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """検証ステップ"""
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """テストステップ"""
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        # Warm-up スケジューラ
        warmup_scheduler = LambdaLR( # ここをLambdaLRに変更
            optimizer,
            lr_lambda=lambda epoch: float(epoch + 1) / float(self.hparams.warmup_epochs),
        )

        # CosineAnnealing スケジューラ
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 1e-2,
        )

        # 順次適用するスケジューラ（Warm-up → Cosine）
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }