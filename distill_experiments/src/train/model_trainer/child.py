import torch
import torch.nn as nn
from collections import OrderedDict
import pytorch_lightning as pl

from src.models.child_model import ChildModel
from src.models.parents_model import ParentModelSimpleConv, ParentModelMaxpool, ParentModelInception

class ChildModelTrainer(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 lr=1e-3,
                 in_channels=None,
                 input_shape=None,
                 use_kd=True,
                 teacher_checkpoint_paths=None,
                 temperature=3,
                 alpha=0.5,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.use_kd = use_kd  # 蒸留を使うかのフラグ
        self.child_model = ChildModel(num_classes, input_shape)
        
        if self.use_kd:
            self.parent_model_simpleconv = ParentModelSimpleConv(in_channels, num_classes)
            self.parent_model_inception = ParentModelInception(in_channels, num_classes)
            self.parent_model_maxpool = ParentModelMaxpool(in_channels, num_classes)

            simpleconv_checkpoint = torch.load(teacher_checkpoint_paths['simple_conv'], map_location=self.device)
            maxpool_checkpoint = torch.load(teacher_checkpoint_paths['maxpool'], map_location=self.device)
            inception_checkpoint = torch.load(teacher_checkpoint_paths['inception'], map_location=self.device)

            replaced_simpleconv_state_dict = self.del_model_from_state_dict_keys(simpleconv_checkpoint)
            replaced_maxpool_state_dict = self.del_model_from_state_dict_keys(maxpool_checkpoint)
            replaced_inception_state_dict = self.del_model_from_state_dict_keys(inception_checkpoint)

            self.parent_model_simpleconv.load_state_dict(replaced_simpleconv_state_dict)
            self.parent_model_maxpool.load_state_dict(replaced_maxpool_state_dict)
            self.parent_model_inception.load_state_dict(replaced_inception_state_dict)

            for model in [self.parent_model_inception, self.parent_model_simpleconv, self.parent_model_maxpool]:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

            self.criterion_kd = nn.KLDivLoss(reduction='batchmean')
            self.temperature = temperature
            self.alpha = alpha

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

            loss_kd = self.criterion_kd(
                nn.functional.log_softmax(child_logits / self.temperature, dim=1),
                nn.functional.softmax(mean_parents_logits / self.temperature, dim=1)
            )
            loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd * (self.temperature ** 2)
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
    
    def del_model_from_state_dict_keys(self, checkpoint):
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_key = k.replace('model.', '')
            new_state_dict[new_key] = v
        return new_state_dict
    
if __name__ == '__main__':
    inputs = torch.rand(1, 224, 224)
    model = ChildModelTrainer(10, 0.0001, 1, (224, 224), True, 3, 0.5)
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    print(output.shape)