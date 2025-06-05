from typing import Callable, Dict, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics.classification import BinaryJaccardIndex

from src.court.lit_module.heatmap_regression_lit_module import HeatmapRegressionLitModule


class SelfTrainingCourtLitModule(HeatmapRegressionLitModule):
    """
    擬似ラベルをサポートするための拡張LightningModule。
    HeatmapRegressionLitModuleを拡張して、擬似ラベルのサンプルに対して
    重み付けした損失計算を行います。
    """

    def __init__(
        self,
        model: Callable,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 1,
        max_epochs: int = 50,
        pseudo_weight: float = 0.5,
    ):
        """
        初期化

        Parameters
        ----------
        model : コート検出モデル
        lr : 学習率
        weight_decay : 重み減衰
        warmup_epochs : ウォームアップエポック数
        max_epochs : 最大エポック数
        pseudo_weight : 擬似ラベルサンプルの損失重み (0.0 ~ 1.0)
        """
        super().__init__(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
        )
        self.pseudo_weight = pseudo_weight
        self.save_hyperparameters(ignore=["model"])

    def _step(self, batch, stage: str):
        """
        学習/検証/テストステップ

        Parameters
        ----------
        batch : バッチデータ
        stage : "train", "val", "test"のいずれか

        Returns
        -------
        loss : 損失値
        """
        # バッチデータの分解
        if len(batch) >= 3:
            frames, heatmaps, is_pseudo = batch
        else:
            frames, heatmaps = batch
            is_pseudo = torch.zeros(frames.size(0), 1, dtype=torch.bool, device=frames.device)

        # モデル予測
        logits = self(frames)
        
        # 損失計算
        if stage == "train" and is_pseudo.any():
            # 擬似ラベルサンプルには重みを適用
            sample_weights = torch.ones(frames.size(0), device=frames.device)
            sample_weights[is_pseudo.view(-1)] = self.pseudo_weight
            
            # 重み付き損失計算
            loss_per_sample = self._compute_loss_per_sample(logits, heatmaps)
            loss = (loss_per_sample * sample_weights).mean()
            
            # 擬似ラベル数と通常ラベル数をログ
            num_pseudo = is_pseudo.sum().item()
            num_real = frames.size(0) - num_pseudo
            self.log(
                f"{stage}_pseudo_ratio",
                num_pseudo / (num_pseudo + num_real),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        else:
            # 通常の損失計算
            loss = self.criterion(logits, heatmaps)

        # 予測をシグモイドして0-1スケールに変換
        preds = torch.sigmoid(logits)

        # IoUの計算
        masked_iou = self.compute_masked_iou(preds, heatmaps)

        # メトリクスのログ
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_IoU",
            masked_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            sync_dist=True,
        )

        return loss

    def _compute_loss_per_sample(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        サンプルごとの損失を計算

        Parameters
        ----------
        logits : モデル出力
        targets : 教師データ

        Returns
        -------
        loss_per_sample : サンプルごとの損失 [B]
        """
        # MSELossをサンプルごとに計算
        mse_per_sample = torch.mean((logits - targets) ** 2, dim=(1, 2, 3))  # [B]
        return mse_per_sample

    def training_epoch_end(self, outputs):
        """
        トレーニングエポック終了時の処理

        Parameters
        ----------
        outputs : 各ステップの出力リスト
        """
        # エポック終了時の追加処理があれば実装
        pass 