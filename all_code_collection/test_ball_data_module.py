"""Unit tests for TennisBallDataModule."""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.ball.lit_datamodule.ball_data_module import TennisBallDataModule
from src.ball.tests.fixtures.mock_data import MockDataManager


@pytest.mark.unit
class TestTennisBallDataModule:
    """Test TennisBallDataModule functionality."""
    
    @pytest.fixture
    def mock_train_transform(self):
        """Create mock training transform."""
        return A.ReplayCompose([
            A.Resize(height=320, width=640),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))
    
    @pytest.fixture
    def mock_val_transform(self):
        """Create mock validation transform."""
        return A.ReplayCompose([
            A.Resize(height=320, width=640),
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))
    
    def test_data_module_init_heatmap(self, mock_train_transform, mock_val_transform):
        """Test DataModule initialization with heatmap dataset type."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                batch_size=16,
                num_workers=4,
                input_size=[320, 640],
                heatmap_size=[320, 640],
                dataset_type="heatmap"
            )
            
            # Check basic attributes
            assert dm.T == 3
            assert dm.batch_size == 16
            assert dm.num_workers == 4
            assert dm.input_size == [320, 640]
            assert dm.heatmap_size == [320, 640]
            assert dm.dataset_type == "heatmap"
            assert dm.train_transform == mock_train_transform
            assert dm.val_test_transform == mock_val_transform
    
    def test_data_module_init_coord(self, mock_train_transform, mock_val_transform):
        """Test DataModule initialization with coord dataset type."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=5,
                batch_size=8,
                dataset_type="coord",
                input_type="stack",
                output_type="last"
            )
            
            assert dm.dataset_type == "coord"
            assert dm.input_type == "stack"
            assert dm.output_type == "last"
    
    def test_data_module_init_invalid_dataset_type(self, mock_train_transform, mock_val_transform):
        """Test DataModule initialization with invalid dataset type."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            with pytest.raises(AssertionError):
                TennisBallDataModule(
                    train_transform=mock_train_transform,
                    val_test_transform=mock_val_transform,
                    annotation_file=str(annotation_file),
                    image_root=str(image_root),
                    dataset_type="invalid"
                )
    
    def test_data_module_inheritance(self, mock_train_transform, mock_val_transform):
        """Test that DataModule inherits from LightningDataModule."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root)
            )
            
            assert isinstance(dm, pl.LightningDataModule)
    
    def test_setup_fit_stage_heatmap(self, mock_train_transform, mock_val_transform):
        """Test setup method for fit stage with heatmap dataset."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                dataset_type="heatmap"
            )
            
            # Mock SequenceKeypointDataset
            with patch('src.ball.lit_datamodule.ball_data_module.SequenceKeypointDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                
                dm.setup(stage="fit")
                
                # Check that datasets were created
                assert hasattr(dm, 'train_dataset')
                assert hasattr(dm, 'val_dataset')
                
                # Check that SequenceKeypointDataset was called twice (train + val)
                assert mock_dataset.call_count == 2
                
                # Check call arguments
                train_call = mock_dataset.call_args_list[0]
                val_call = mock_dataset.call_args_list[1]
                
                assert train_call[1]['split'] == 'train'
                assert val_call[1]['split'] == 'val'
    
    def test_setup_fit_stage_coord(self, mock_train_transform, mock_val_transform):
        """Test setup method for fit stage with coord dataset."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                dataset_type="coord"
            )
            
            # Mock SequenceCoordDataset
            with patch('src.ball.lit_datamodule.ball_data_module.SequenceCoordDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                
                dm.setup(stage="fit")
                
                # Check that datasets were created
                assert hasattr(dm, 'train_dataset')
                assert hasattr(dm, 'val_dataset')
                
                # Check that SequenceCoordDataset was called
                assert mock_dataset.call_count == 2
    
    def test_setup_test_stage(self, mock_train_transform, mock_val_transform):
        """Test setup method for test stage."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                dataset_type="coord"
            )
            
            with patch('src.ball.lit_datamodule.ball_data_module.SequenceCoordDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                
                dm.setup(stage="test")
                
                # Check that test dataset was created
                assert hasattr(dm, 'test_dataset')
                
                # Check that dataset was called once for test
                assert mock_dataset.call_count == 1
                call_args = mock_dataset.call_args_list[0]
                assert call_args[1]['split'] == 'test'
    
    def test_setup_none_stage(self, mock_train_transform, mock_val_transform):
        """Test setup method with None stage (should setup both fit and test)."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                dataset_type="coord"
            )
            
            with patch('src.ball.lit_datamodule.ball_data_module.SequenceCoordDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                
                dm.setup(stage=None)
                
                # Check that all datasets were created
                assert hasattr(dm, 'train_dataset')
                assert hasattr(dm, 'val_dataset')
                assert hasattr(dm, 'test_dataset')
                
                # Check that dataset was called 3 times (train, val, test)
                assert mock_dataset.call_count == 3
    
    def test_prepare_dataset_heatmap(self, mock_train_transform, mock_val_transform):
        """Test _prepare_dataset method with heatmap type."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                dataset_type="heatmap",
                T=4,
                input_size=[256, 512],
                heatmap_size=[128, 256]
            )
            
            with patch('src.ball.lit_datamodule.ball_data_module.SequenceKeypointDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                
                dataset = dm._prepare_dataset("train", mock_train_transform)
                
                # Check that SequenceKeypointDataset was called with correct parameters
                mock_dataset.assert_called_once_with(
                    annotation_file=annotation_file,
                    image_root=image_root,
                    T=4,
                    input_size=[256, 512],
                    heatmap_size=[128, 256],
                    transform=mock_train_transform,
                    split="train",
                    skip_frames_range=dm.skip_frames_range,
                    input_type=dm.input_type,
                    output_type=dm.output_type
                )
    
    def test_prepare_dataset_coord(self, mock_train_transform, mock_val_transform):
        """Test _prepare_dataset method with coord type."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                dataset_type="coord",
                T=6,
                input_size=[320, 640]
            )
            
            with patch('src.ball.lit_datamodule.ball_data_module.SequenceCoordDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                
                dataset = dm._prepare_dataset("val", mock_val_transform)
                
                # Check that SequenceCoordDataset was called with correct parameters
                mock_dataset.assert_called_once_with(
                    annotation_file=annotation_file,
                    image_root=image_root,
                    T=6,
                    input_size=[320, 640],
                    transform=mock_val_transform,
                    split="val", 
                    skip_frames_range=dm.skip_frames_range,
                    input_type=dm.input_type,
                    output_type=dm.output_type
                )
    
    def test_train_dataloader(self, mock_train_transform, mock_val_transform):
        """Test train_dataloader method."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                batch_size=32,
                num_workers=8
            )
            
            # Mock dataset
            mock_dataset = MagicMock()
            dm.train_dataset = mock_dataset
            
            with patch('src.ball.lit_datamodule.ball_data_module.DataLoader') as mock_dataloader:
                mock_dataloader.return_value = MagicMock()
                
                dataloader = dm.train_dataloader()
                
                # Check that DataLoader was called with correct parameters
                mock_dataloader.assert_called_once_with(
                    dataset=mock_dataset,
                    batch_size=32,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True
                )
    
    def test_val_dataloader(self, mock_train_transform, mock_val_transform):
        """Test val_dataloader method."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                batch_size=16,
                num_workers=4
            )
            
            mock_dataset = MagicMock()
            dm.val_dataset = mock_dataset
            
            with patch('src.ball.lit_datamodule.ball_data_module.DataLoader') as mock_dataloader:
                mock_dataloader.return_value = MagicMock()
                
                dataloader = dm.val_dataloader()
                
                # Check that DataLoader was called with shuffle=False for validation
                mock_dataloader.assert_called_once_with(
                    dataset=mock_dataset,
                    batch_size=16,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
    
    def test_test_dataloader(self, mock_train_transform, mock_val_transform):
        """Test test_dataloader method."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                batch_size=8,
                num_workers=2
            )
            
            mock_dataset = MagicMock()
            dm.test_dataset = mock_dataset
            
            with patch('src.ball.lit_datamodule.ball_data_module.DataLoader') as mock_dataloader:
                mock_dataloader.return_value = MagicMock()
                
                dataloader = dm.test_dataloader()
                
                # Check that DataLoader was called with shuffle=False for test
                mock_dataloader.assert_called_once_with(
                    dataset=mock_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )
    
    def test_make_dataloader(self, mock_train_transform, mock_val_transform):
        """Test _make_dataloader helper method."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                batch_size=64,
                num_workers=16
            )
            
            mock_dataset = MagicMock()
            
            with patch('src.ball.lit_datamodule.ball_data_module.DataLoader') as mock_dataloader:
                mock_dataloader.return_value = MagicMock()
                
                # Test with shuffle=True
                dataloader_shuffle = dm._make_dataloader(mock_dataset, shuffle=True)
                
                # Test with shuffle=False
                dataloader_no_shuffle = dm._make_dataloader(mock_dataset, shuffle=False)
                
                # Check that DataLoader was called twice with different shuffle values
                assert mock_dataloader.call_count == 2
                
                call_args_shuffle = mock_dataloader.call_args_list[0]
                call_args_no_shuffle = mock_dataloader.call_args_list[1]
                
                assert call_args_shuffle[1]['shuffle'] is True
                assert call_args_no_shuffle[1]['shuffle'] is False
                
                # Check common parameters
                for call_args in [call_args_shuffle, call_args_no_shuffle]:
                    assert call_args[1]['dataset'] == mock_dataset
                    assert call_args[1]['batch_size'] == 64
                    assert call_args[1]['num_workers'] == 16
                    assert call_args[1]['pin_memory'] is True
    
    def test_save_hyperparameters(self, mock_train_transform, mock_val_transform):
        """Test that hyperparameters are saved correctly."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=7,
                batch_size=24,
                dataset_type="coord"
            )
            
            # Check that hyperparameters were saved
            assert hasattr(dm, 'hparams')
            
            # Check that basic parameters are in hparams
            assert dm.hparams.T == 7
            assert dm.hparams.batch_size == 24
            assert dm.hparams.dataset_type == "coord"
            
            # Check that transforms were ignored (not in hparams)
            assert 'train_transform' not in dm.hparams
            assert 'val_test_transform' not in dm.hparams
    
    def test_different_parameter_combinations(self, mock_train_transform, mock_val_transform):
        """Test DataModule with different parameter combinations."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Test different parameter combinations
            param_combinations = [
                {
                    "T": 3,
                    "batch_size": 8,
                    "input_type": "cat",
                    "output_type": "all",
                    "dataset_type": "heatmap"
                },
                {
                    "T": 5,
                    "batch_size": 16,
                    "input_type": "stack",
                    "output_type": "last",
                    "dataset_type": "coord"
                },
                {
                    "T": 8,
                    "batch_size": 4,
                    "input_type": "stack",
                    "output_type": "all",
                    "dataset_type": "coord",
                    "skip_frames_range": (1, 3)
                }
            ]
            
            for params in param_combinations:
                dm = TennisBallDataModule(
                    train_transform=mock_train_transform,
                    val_test_transform=mock_val_transform,
                    annotation_file=str(annotation_file),
                    image_root=str(image_root),
                    **params
                )
                
                # Check that all parameters were set correctly
                for key, value in params.items():
                    assert getattr(dm, key) == value
    
    @pytest.mark.slow
    def test_integration_with_real_dataset(self, mock_train_transform, mock_val_transform):
        """Test integration with real dataset classes."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Test with coord dataset type (actual dataset creation)
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                batch_size=2,
                num_workers=0,  # No multiprocessing for testing
                dataset_type="coord"
            )
            
            # Setup datasets
            dm.setup(stage="fit")
            
            # Check that datasets were created
            assert hasattr(dm, 'train_dataset')
            assert hasattr(dm, 'val_dataset')
            
            # Get dataloaders
            train_loader = dm.train_dataloader()
            val_loader = dm.val_dataloader()
            
            # Check that dataloaders are DataLoader instances
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)
            
            # Check basic properties
            assert train_loader.batch_size == 2
            assert val_loader.batch_size == 2
            assert train_loader.dataset == dm.train_dataset
            assert val_loader.dataset == dm.val_dataset
    
    def test_path_handling(self, mock_train_transform, mock_val_transform):
        """Test DataModule with Path objects."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Test with Path objects
            dm = TennisBallDataModule(
                train_transform=mock_train_transform,
                val_test_transform=mock_val_transform,
                annotation_file=annotation_file,  # Path object
                image_root=image_root,            # Path object
                dataset_type="coord"
            )
            
            assert dm.annotation_file == annotation_file
            assert dm.image_root == image_root