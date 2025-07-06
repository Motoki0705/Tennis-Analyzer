"""
RTDETR用LightningModule
"""
from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import RTDetrForObjectDetection
import transformers

class LitRtdetr(pl.LightningModule):
    """
    RTDETR用LightningModule
    
    入力: Images [B, C, H, W]
    出力: Object Detection (logits, boxes)
    """

    def __init__(
        self,
        # モデル構成パラメータ
        pretrained_model_name_or_path: str = "PekingU/rtdetr_v2_r18vd",
        num_labels: int = 1,
        
        # 学習パラメータ
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        optim_t_max: int = 50,
        min_lr: float = 1e-6,
        num_freeze_epoch: int = 3,
    ):
        super().__init__()
        
        # モデルの初期化
        self.model = RTDetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # 学習パラメータの保存
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.optim_t_max = optim_t_max
        self.min_lr = min_lr
        self.num_freeze_epoch = num_freeze_epoch

        # ハイパーパラメータの保存
        self.save_hyperparameters()

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
        elif self.current_epoch == self.num_freeze_epoch:
            self.unfreeze_encoder()

    def forward(self, pixel_values, pixel_mask=None):
        """順伝播処理"""
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def training_step(self, batch, batch_idx):
        """学習ステップ"""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """検証ステップ"""
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """テストステップ"""
        return self._common_step(batch, "test")

    def _common_step(self, batch, stage: str):
        """共通処理ステップ"""
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask", None)
        
        # 各ターゲットをデバイスへ移動
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        
        # モデルの出力を取得
        outputs = self.model(
            pixel_values=pixel_values, 
            pixel_mask=pixel_mask, 
            labels=labels
        )

        loss = outputs["loss"]
        loss_dict = outputs["loss_dict"]
        
        # メトリクスのロギング
        self.log(
            f"{stage}_loss_total",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        for k, v in loss_dict.items():
            self.log(
                f"{stage}_loss_{k}",
                v,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(stage == "val"),
                logger=True,
                sync_dist=True,
            )
        
        return loss

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        # パラメータをバックボーンとそれ以外で分割
        param_dict = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        
        optimizer = optim.AdamW(param_dict, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.optim_t_max, 
            eta_min=self.min_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch"
            },
        } 