import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from BallTrack.src.models.composed_model import ComposedModel

class ModelTrainer(pl.LightningModule):
    def __init__(self, orig_channels, num_keypoints, lr, min_lr, weight_decay, t_max, extractor=None, upsampler=None):
        super().__init__()

        self.model = ComposedModel(
            orig_channels=orig_channels,
            num_keypoints=num_keypoints,
            extractor=extractor,
            upsampler=upsampler
        )
        self.criterion = nn.MSELoss()

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.t_max = t_max

    def forward(self, inputs):
        return  self.model(inputs)
    
    def common_step(self, batch):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.t_max,
            eta_min=self.min_lr
        )
        return [optimizer], [scheduler]
    
if __name__ == '__main__':
    model = ModelTrainer(
        orig_channels=3,
        num_keypoints=30,
        lr=1e-4,
        min_lr=1e-5,
        weight_decay=1e-4,
        t_max=20,
        extractor='segformer',
        upsampler='simple'
        )
    model.eval()
    inputs = torch.rand(2, 3, 512, 512)

    with torch.no_grad():
        output = model(inputs)

    print(output.shape)