import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from src.models.parents_model import ParentModelSimpleConv, ParentModelMaxpool, ParentModelInception

class ParentLightningModule(pl.LightningModule):
    def __init__(self, model_name: str, in_channels: int = None, num_classes: int = None, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        if model_name == "simple_conv":
            self.model = ParentModelSimpleConv(in_channels, num_classes)
        elif model_name == "maxpool":
            self.model = ParentModelMaxpool(in_channels, num_classes)
        elif model_name == "inception":
            self.model = ParentModelInception(in_channels, num_classes)
        else:
            raise ValueError(f"Invalid model name {model_name}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)