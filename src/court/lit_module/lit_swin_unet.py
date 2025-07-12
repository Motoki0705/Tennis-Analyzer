import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR

# 上記で作成したSwinUNetモデルをインポート
from src.court.models.swin_unet import SwinUNet

class LitSwinUNet(pl.LightningModule):
    """
    SwinUNet用LightningModule with Focal Loss and Finetuning Logic.
    """
    def __init__(
        self,
        # Model parameters
        model_name: str = 'swin_tiny_patch4_window7_224',
        in_channels: int = 3,
        out_channels: int = 15,
        
        # Training parameters
        lr: float = 1e-3,
        lr_finetune: float = 1e-5, # ファインチューニング時の学習率
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        unfreeze_epoch: int = 20, # エンコーダを解凍するエポック
        
        # Loss/Metric parameters
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        accuracy_threshold: int = 5,

        # ピーク + 谷のヒートマップの場合
        use_peak_valley_heatmaps: bool = False
    ):
        super().__init__()
        
        # モデルの初期化
        self.model = SwinUNet(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels
        )
        
        # ハイパーパラメータの保存
        self.save_hyperparameters()
        
    def freeze_encoder(self):
        """エンコーダの重みを凍結する"""
        print("Freezing encoder weights.")
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """エンコーダの重みを解凍する"""
        print("Unfreezing encoder weights for finetuning.")
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self):
        """
        各学習エポックの開始時に、現在のエポック数に基づいてエンコーダの状態を決定する。
        これにより、チェックポイントからの再開時にも正しい状態が復元される。
        """
        # unfreeze_epoch より前のエポックでは、エンコーダは常に凍結されているべき
        if self.current_epoch < self.hparams.unfreeze_epoch:
            self.freeze_encoder()
        else:
            # unfreeze_epoch 以降のエポックでは、エンコーダは常に解凍されているべき
            self.unfreeze_encoder()
        
        # unfreeze_epoch に達した瞬間にオプティマイザを再設定する
        # この条件は、学習を再開したエポックがunfreeze_epochと一致する場合にも対応
        if self.current_epoch == self.hparams.unfreeze_epoch:
            print(f"Epoch {self.current_epoch}: Re-setting up optimizers for finetuning.")
            self.trainer.strategy.setup_optimizers(self.trainer)



    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage: str):
        frames, heatmaps, scaled_keypoints = batch
        logits = self(frames)

        # `use_peak_valley_heatmaps`がTrueの場合、ターゲットを[-1, 1]から[0, 1]に変換
        if self.hparams.use_peak_valley_heatmaps:
            target_heatmaps = (heatmaps + 1.0) / 2.0
        else:
            target_heatmaps = heatmaps # 元々[0, 1]なので何もしない
        
        loss = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=heatmaps,
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
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定"""
        # パラメータをエンコーダとそれ以外（デコーダ＋ヘッド）に分割
        encoder_params = self.model.encoder.parameters()
        decoder_params = [p for name, p in self.model.named_parameters() if not name.startswith("encoder.")]
        
        # パラメータグループごとに異なる学習率を設定
        param_groups = [
            {"params": decoder_params, "lr": self.hparams.lr},
            # エンコーダの学習率は最初は0（凍結）か、ファインチューニング用に設定
            {"params": encoder_params, "lr": self.hparams.lr_finetune},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay
        )

        # スケジューラの設計 (Warm-up -> Cosine Annealing)
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: float(epoch + 1) / float(self.hparams.warmup_epochs),
        )
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr_finetune * 0.1, # ファインチューニングLRよりもさらに小さく
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.hparams.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # _find_heatmap_peaks と _calculate_accuracy は LitLiteTracknetFocal からコピー
    def _find_heatmap_peaks(self, heatmaps: torch.Tensor) -> torch.Tensor:
        batch_size, num_kpts, h, w = heatmaps.shape
        flat_heatmaps = heatmaps.view(batch_size, num_kpts, -1)
        _, max_indices = torch.max(flat_heatmaps, dim=2)
        peak_y = max_indices // w
        peak_x = max_indices % w
        return torch.stack([peak_x, peak_y], dim=2).float()

    def _calculate_accuracy(self, pred_coords: torch.Tensor, gt_keypoints: torch.Tensor) -> torch.Tensor:
        gt_coords = gt_keypoints[:, :, :2]
        visibility = gt_keypoints[:, :, 2]
        distances = torch.linalg.norm(pred_coords - gt_coords, dim=2)
        correct_preds = distances <= self.hparams.accuracy_threshold
        num_correct = torch.sum(correct_preds * visibility)
        num_visible = torch.sum(visibility)
        return num_correct / num_visible if num_visible > 0 else torch.tensor(0.0, device=self.device)