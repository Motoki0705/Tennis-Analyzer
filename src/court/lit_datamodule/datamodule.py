# Placeholder for Court LitDataModule
import pytorch_lightning as pl

class CourtLitDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # TODO: Initialize datasets, transforms etc.

    def setup(self, stage=None):
        # TODO: Assign train/val datasets for use in dataloaders
        pass

    def train_dataloader(self):
        # TODO: Return train dataloader
        return None

    def val_dataloader(self):
        # TODO: Return val dataloader
        return None

    def test_dataloader(self):
        # TODO: Return test dataloader
        return None
