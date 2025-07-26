import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from src.ball.lit_module.lit_ball_model import LitBallModel

@pytest.mark.unit
class TestLitBallModel:
    """Test LitBallModel Lightning module."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple nn.Module for testing."""
        return nn.Sequential(nn.Linear(10, 2))

    @pytest.fixture
    def dummy_dataloader(self):
        """Create a dummy DataLoader for testing."""
        dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 1))
        return DataLoader(dataset, batch_size=2)

    def test_load_model_from_checkpoint(self, simple_model, dummy_dataloader, tmpdir):
        """
        Tests if a LitBallModel can be saved to a checkpoint and loaded back correctly.
        """
        # 1. Arrange: Create and save a LitBallModel checkpoint
        original_model = LitBallModel(
            model=simple_model,
            criterion={"alpha": 0.5, "gamma": 2.0, "reduction": "mean"},
            optimizer_config={"lr": 1e-3},
            scheduler_config={"warmup_epochs": 1, "max_epochs": 10}
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(tmpdir),
            filename='test_checkpoint',
            save_top_k=1,
            monitor='train_loss',
            mode='min'
        )

        # Dummy training step to create a checkpoint
        class DummyDataModule(pl.LightningDataModule):
            def train_dataloader(self):
                return dummy_dataloader
            # Add dummy val_dataloader to avoid warnings/errors
            def val_dataloader(self):
                return dummy_dataloader

        trainer = pl.Trainer(
            default_root_dir=str(tmpdir),
            callbacks=[checkpoint_callback],
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            # Add a dummy validation step to avoid errors
            limit_val_batches=0
        )

        # A dummy training loop to trigger checkpointing
        trainer.fit(original_model, datamodule=DummyDataModule())
        checkpoint_path = checkpoint_callback.best_model_path

        # 2. Act: Load the model from the checkpoint
        loaded_model = LitBallModel.load_from_checkpoint(checkpoint_path)

        # 3. Assert: Check if the loaded model is correct
        assert isinstance(loaded_model, LitBallModel)

        # Check if model weights are the same
        original_state_dict = original_model.model.state_dict()
        loaded_state_dict = loaded_model.model.state_dict()
        for key in original_state_dict:
            assert torch.equal(original_state_dict[key], loaded_state_dict[key])

        # Check if hyperparameters are loaded correctly
        assert loaded_model.hparams.optimizer_config['lr'] == original_model.hparams.optimizer_config['lr']
        assert loaded_model.hparams.scheduler_config['warmup_epochs'] == original_model.hparams.scheduler_config['warmup_epochs']
        assert loaded_model.hparams.criterion['alpha'] == original_model.hparams.criterion['alpha']
