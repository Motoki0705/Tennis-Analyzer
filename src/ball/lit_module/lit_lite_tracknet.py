"""
LiteTrackNet用LightningModule
"""
from typing import Dict, Any
import os
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics.classification import BinaryJaccardIndex
import torchvision

from src.ball.models.lite_tracknet import LiteTrackNet


class LitLiteTracknet(pl.LightningModule):
    """
    LiteTrackNet用LightningModule
    
    入力: Frames [B, C, H, W]
    出力: Heatmaps [B, H, W]
    """

    def __init__(
        self,
        # モデル構成パラメータ
        in_channels: int = 9,
        out_channels: int = 1,
        
        # 学習パラメータ
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
        bce_weight: float = 0.7,

        # ===== 追加: 可視化パラメータ =====
        num_log_images: int = 4, # 保存する画像の枚数
    ):
        super().__init__()
        
        # ハイパーパラメータを保存します
        self.save_hyperparameters()

        # モデルの初期化
        self.model = LiteTrackNet(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels
        )
        
        # 損失関数と評価指標
        self.criterion = nn.MSELoss()
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

        # ===== 追加: 検証ステップで可視化のために最初のバッチの出力を保存する変数 =====
        self._val_batch_outputs_for_log = None


    def forward(self, x):
        """順伝播処理"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """学習ステップ"""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """検証ステップ"""
        # ===== 追加: 最初のバッチの入出力を可視化用に保存 =====
        if batch_idx == 0:
            frames, heatmaps, _, _ = batch
            logits = self(frames)
            # 3次元の場合は4次元に変換
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            preds = torch.sigmoid(logits)
            
            # 必要なテンソルをCPUに移動させて保存
            self._val_batch_outputs_for_log = {
                "frames": frames.cpu(),
                "preds": preds.cpu(),
                "targets": heatmaps.cpu()
            }
            
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """テストステップ"""
        return self._common_step(batch, "test")

    def _common_step(self, batch, stage: str):
        """共通処理ステップ"""
        frames, heatmaps, _, _ = batch
        logits = self(frames)
        
        if heatmaps.dim() == 3:
            heatmaps = heatmaps.unsqueeze(1)
        
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
            
        loss = self.criterion(logits, heatmaps)
        preds = torch.sigmoid(logits)
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_iou", masked_iou, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    # ===== 追加: 検証エポックの終わりに画像セットを保存するフック =====
    def on_validation_epoch_end(self):
        """検証エポックの終わりに画像セットを保存"""
        if self._val_batch_outputs_for_log is None:
            return

        # ロガーが利用可能か確認 (TensorBoardLoggerなど)
        if self.logger and hasattr(self.logger, 'log_dir'):
            log_dir = self.logger.log_dir
            save_dir = os.path.join(log_dir, "validation_images")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存するデータの取得
            frames = self._val_batch_outputs_for_log["frames"]
            preds = self._val_batch_outputs_for_log["preds"]
            targets = self._val_batch_outputs_for_log["targets"]

            # バッチサイズの取得
            batch_size = frames.size(0)
            
            # ランダムにサンプルを選択
            available_indices = list(range(batch_size))
            num_images = min(self.hparams.num_log_images, batch_size)
            selected_indices = random.sample(available_indices, num_images)

            for idx, batch_idx in enumerate(selected_indices):
                # 入力フレーム (in_channels=9 のうち最初の3chをRGBと仮定)
                frame_img = frames[batch_idx, :3] 
                
                # 推論ヒートマップ (グレースケール)
                pred_heatmap = preds[batch_idx]

                # ターゲットヒートマップ (グレースケール)
                target_heatmap = targets[batch_idx]
                if target_heatmap.dim() == 2: # チャンネル次元がない場合
                    target_heatmap = target_heatmap.unsqueeze(0)

                # 3つの画像を結合して保存
                # ヒートマップを3チャンネルに変換して結合
                combined_image = torch.cat([
                    frame_img,
                    pred_heatmap.repeat(3, 1, 1),
                    target_heatmap.repeat(3, 1, 1)
                ], dim=2) # 幅方向に結合

                # 画像を保存
                filename = os.path.join(save_dir, f"epoch_{self.current_epoch:03d}_sample_{idx:02d}.png")
                torchvision.utils.save_image(combined_image, filename)

        # 次のエポックのためにリセット
        self._val_batch_outputs_for_log = None

    @staticmethod
    def compute_masked_iou(
        preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """マスクされたIoUを計算"""
        gt_mask = targets > threshold
        pred_mask = preds > threshold
        intersection = (gt_mask & pred_mask).float().sum(dim=(1, 2, 3))
        gt_area = gt_mask.float().sum(dim=(1, 2, 3)) + 1e-8
        iou = intersection / gt_area
        return iou.mean()

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: float(epoch + 1) / float(self.hparams.warmup_epochs)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs - self.hparams.warmup_epochs, eta_min=self.hparams.lr * 1e-2
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.hparams.warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
