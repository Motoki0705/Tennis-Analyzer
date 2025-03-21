import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.models.child_model import ChildModel
from src.models.parents_model import ParentModelSimpleConv, ParentModelMaxpool, ParentModelInception

class ChildModelTrainer(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3, in_channels=None, input_shape=None, use_kd=True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.use_kd = use_kd  # 蒸留を使うかのフラグ
        self.child_model = ChildModel(num_classes, input_shape)
        
        if self.use_kd:
            self.parent_model_inception = ParentModelInception(in_channels, num_classes)
            self.parent_model_maxpool = ParentModelMaxpool(in_channels, num_classes)
            self.parent_model_simpleconv = ParentModelSimpleConv(in_channels, num_classes)

            for model in [self.parent_model_inception, self.parent_model_simpleconv, self.parent_model_maxpool]:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

            self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
            self.temperature = 3.0

        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.child_model(x)

    def common_step(self, batch):
        inputs, targets = batch
        child_logits = self.child_model(inputs)
        loss_ce = self.criterion_ce(child_logits, targets)
        
        if self.use_kd:
            with torch.no_grad():
                inception_logits = self.parent_model_inception(inputs)
                maxpool_logits = self.parent_model_maxpool(inputs)
                simple_logits = self.parent_model_simpleconv(inputs)
            mean_parents_logits = (inception_logits + maxpool_logits + simple_logits) / 3

            # Knowledge Distillation Loss (soft targets)
            loss_kd = self.criterion_kd(
                nn.functional.log_softmax(child_logits / self.temperature, dim=1),
                nn.functional.softmax(mean_parents_logits / self.temperature, dim=1)
            )
            loss = loss_ce + loss_kd * (self.temperature ** 2)
        else:
            loss_kd = torch.tensor(0.0, device=self.device)
            loss = loss_ce  # 通常のCrossEntropyLossのみ適用

        return loss, loss_ce, loss_kd

    def training_step(self, batch, batch_idx):
        loss, loss_ce, loss_kd = self.common_step(batch)
        self.log("train_loss", loss)
        self.log("train_ce_loss", loss_ce)
        if self.use_kd:
            self.log("train_kd_loss", loss_kd)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_ce, loss_kd = self.common_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ce_loss", loss_ce, prog_bar=True)
        if self.use_kd:
            self.log("val_kd_loss", loss_kd, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.child_model.parameters(), lr=self.lr)
        return optimizer