import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class DetrBasedTrainer(pl.LightningModule):
    def __init__(
        self,
        detr_based_model: nn.Module,
        lr: float,
        lr_backbone: float,
        weight_decay: float,
        optim_t_max: int,
        min_lr: float,
        num_freeze_epoch: int,
    ):
        """
        Args:
            detr_based_model: DETR系のモデル
            criterion: CustomCriterion インスタンス（distillation=False のもの）
            lr: 学習率（バックボーン以外）
            lr_backbone: バックボーン用学習率
            weight_decay: Weight decay
            optim_t_max: CosineAnnealingLR の T_max
            min_lr: 最小学習率
            num_freeze_epoch: バックボーン層のフリーズ期間
        """
        super().__init__()
        self.model = detr_based_model
        self.save_hyperparameters(ignore=["detr_based_model"])

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.optim_t_max = optim_t_max
        self.min_lr = min_lr
        self.num_freeze_epoch = num_freeze_epoch

    def freeze_encoder(self):
        for name, module in self.model.named_modules():
            if "backbone" in name:
                for param in module.parameters():
                    param.requires_grad = False
                found = True
        print("Encoder is freeze")
        if not found:
            raise AttributeError("No module with 'backbone' found in the model.")

    def unfreeze_encoder(self):
        for name, module in self.model.named_modules():
            if "backbone" in name:
                for param in module.parameters():
                    param.requires_grad = True
        print("Encoder is unfrozen")

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.freeze_encoder()
        elif self.current_epoch == self.num_freeze_epoch:
            self.unfreeze_encoder()

    def forward(self, pixel_values, pixel_mask=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask", None)
        # 各ターゲットをデバイスへ移動
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        # モデルはラベルを与えずに予測を出力（logits, pred_boxes）
        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        return outputs["loss"], outputs["loss_dict"]

    def log_loss_dict(
        self, stage: str, loss: torch.Tensor, loss_dict: dict, prog_bar: bool = False
    ):
        self.log(f"{stage}_total", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"{stage}_{k}", v, prog_bar=prog_bar)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log_loss_dict("train_loss", loss, loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log_loss_dict("val_loss", loss, loss_dict)
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log_loss_dict("test_loss", loss, loss_dict)
        return loss

    def configure_optimizers(self):
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
            optimizer, T_max=self.optim_t_max, eta_min=self.min_lr
        )
        print("Learning rate for backbone:", self.lr_backbone)
        print("Learning rate for other params:", self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
