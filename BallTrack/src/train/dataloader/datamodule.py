import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from BallTrack.src.train.dataloader.dataset import TennisDataset
from BallTrack.src.tests.datamodule_tester import DataModuleTester

class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_transform, val_test_transform, batch_size, num_workers):
        super().__init__()
        # dataset params
        self.data_path = data_path
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform

        # detaloader params
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = TennisDataset(
                root_dir=self.data_path,
                transform=self.train_transform,
                mode='train'
                )
            self.val_dataset = TennisDataset(
                root_dir=self.data_path,
                transform=self.val_test_transform,
                mode='val'
                )
        elif stage == 'test':
            self.test_dataset = TennisDataset(
                root_dir=self.data_path,
                transform=self.val_test_transform,
                mode='test'
            )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
if __name__ == '__main__':
    # DataModuleのテストを行います。
    import torchvision.transforms as transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    val_test_transform = train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    data_module = DataModule(
        data_path=r'TrackNet/datasets/tennis',
        train_transform=train_transform,
        val_test_transform=val_test_transform,
        batch_size=2,
        num_workers=0
    )
    datamodule_tester = DataModuleTester(data_module)
    datamodule_tester.test()
