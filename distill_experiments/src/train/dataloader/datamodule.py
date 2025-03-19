import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.train.dataloader.dataset import train_dataset, val_dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        elif stage == 'test':
            print('This module is not cover test stage')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
            )
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )