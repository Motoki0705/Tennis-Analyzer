"""Unit tests for LitBallModel Lightning module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

from src.ball.lit_module.lit_ball_model import LitBallModel


@pytest.mark.unit
class TestLitBallModel:
    """Test LitBallModel Lightning module."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock(spec=nn.Module)
        model.forward.return_value = torch.randn(2, 1, 64, 64)
        return model
    
    @pytest.fixture
    def criterion_config(self):
        """Create criterion configuration."""
        return {
            "alpha": 0.25,
            "gamma": 2.0,
            "reduction": "mean"
        }
    
    @pytest.fixture
    def optimizer_config(self):
        """Create optimizer configuration."""
        return {
            "lr": 0.001,
            "weight_decay": 1e-4
        }
    
    @pytest.fixture
    def scheduler_config(self):
        """Create scheduler configuration."""
        return {
            "warmup_epochs": 5,
            "max_epochs": 100
        }
    
    @pytest.fixture
    def logging_config(self):
        """Create logging configuration."""
        return {
            "num_log_images": 4,
            "save_validation_outputs": True,
            "save_validation_images": True
        }
    
    def test_lit_ball_model_init(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test LitBallModel initialization."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Check that model is stored
        assert lit_model.model == mock_model
        assert lit_model.criterion_config == criterion_config
        assert lit_model.optimizer_config == optimizer_config
        assert lit_model.scheduler_config == scheduler_config
        
        # Check default logging config
        assert "num_log_images" in lit_model.logging_config
        assert lit_model.logging_config["num_log_images"] == 4
    
    def test_lit_ball_model_init_with_logging(self, mock_model, criterion_config, optimizer_config, scheduler_config, logging_config):
        """Test LitBallModel initialization with custom logging config."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            logging_config=logging_config
        )
        
        assert lit_model.logging_config == logging_config
    
    def test_compute_loss_standard_format(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test compute_loss with standard format inputs."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Create batch: (frames, heatmaps, visibility, image_info)
        frames = torch.randn(2, 3, 64, 64)
        heatmaps = torch.randn(2, 64, 64)  # Ground truth heatmaps
        visibility = torch.ones(2)
        image_info = [{"id": 1}, {"id": 2}]
        batch = (frames, heatmaps, visibility, image_info)
        
        # Mock model output
        outputs = torch.randn(2, 64, 64)  # Logits
        
        with patch('torchvision.ops.sigmoid_focal_loss') as mock_focal_loss:
            mock_focal_loss.return_value = torch.tensor(0.5)
            
            loss = lit_model.compute_loss(batch, outputs, "train")
            
            # Check that focal loss was called
            mock_focal_loss.assert_called_once()
            assert loss == 0.5
    
    def test_compute_loss_balldetnet_format(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test compute_loss with BallDetNet format (4D tensors)."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # BallDetNet format: [B, N, H, W]
        frames = torch.randn(2, 5, 3, 64, 64)
        heatmaps = torch.randn(2, 5, 64, 64)
        visibility = torch.ones(2, 5)
        image_info = [{"id": 1}, {"id": 2}]
        batch = (frames, heatmaps, visibility, image_info)
        
        outputs = torch.randn(2, 5, 64, 64)  # BallDetNet output
        
        with patch('torchvision.ops.sigmoid_focal_loss') as mock_focal_loss:
            mock_focal_loss.return_value = torch.tensor(0.8)
            
            loss = lit_model.compute_loss(batch, outputs, "train")
            
            # Check that focal loss was called with reshaped tensors
            mock_focal_loss.assert_called_once()
            call_args = mock_focal_loss.call_args
            
            # Check that tensors were reshaped to [B*N, H, W] format
            inputs_shape = call_args[1]['inputs'].shape
            targets_shape = call_args[1]['targets'].shape
            
            assert inputs_shape == (10, 1, 64, 64)  # B*N = 2*5 = 10
            assert targets_shape == (10, 1, 64, 64)
            
            assert loss == 0.8
    
    def test_compute_loss_focal_loss_parameters(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test that focal loss is called with correct parameters."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Create simple batch
        frames = torch.randn(1, 3, 32, 32)
        heatmaps = torch.randn(1, 32, 32)
        batch = (frames, heatmaps, torch.ones(1), [{"id": 1}])
        outputs = torch.randn(1, 32, 32)
        
        with patch('torchvision.ops.sigmoid_focal_loss') as mock_focal_loss:
            mock_focal_loss.return_value = torch.tensor(0.3)
            
            lit_model.compute_loss(batch, outputs, "train")
            
            # Check focal loss parameters
            call_args = mock_focal_loss.call_args
            assert call_args[1]['alpha'] == 0.25
            assert call_args[1]['gamma'] == 2.0
            assert call_args[1]['reduction'] == "mean"
    
    def test_compute_metrics(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test compute_metrics method."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Create dummy batch and outputs
        batch = (torch.randn(2, 3, 64, 64), torch.randn(2, 64, 64), torch.ones(2), [{"id": 1}, {"id": 2}])
        outputs = torch.randn(2, 64, 64)
        
        metrics = lit_model.compute_metrics(batch, outputs, "train")
        
        # Currently returns empty dict (can be extended)
        assert isinstance(metrics, dict)
    
    def test_validation_step_standard(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test validation step with standard format."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Mock the parent validation_step
        with patch.object(lit_model.__class__.__bases__[0], 'validation_step') as mock_parent_val:
            mock_parent_val.return_value = {"val_loss": 0.5}
            
            # Create batch
            frames = torch.randn(2, 3, 64, 64)
            heatmaps = torch.randn(2, 64, 64)
            batch = (frames, heatmaps, torch.ones(2), [{"id": 1}, {"id": 2}])
            
            # Mock model forward pass
            lit_model.model.forward.return_value = torch.randn(2, 64, 64)
            
            result = lit_model.validation_step(batch, batch_idx=0)
            
            # Check that validation outputs were stored for first batch
            assert hasattr(lit_model, '_validation_outputs')
            assert 'frames' in lit_model._validation_outputs
            assert 'preds' in lit_model._validation_outputs
            assert 'targets' in lit_model._validation_outputs
            
            # Check shapes
            assert lit_model._validation_outputs['frames'].shape == (2, 3, 64, 64)
            assert lit_model._validation_outputs['preds'].shape == (2, 1, 64, 64)
            assert lit_model._validation_outputs['targets'].shape == (2, 1, 64, 64)
    
    def test_validation_step_balldetnet_format(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test validation step with BallDetNet format."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        with patch.object(lit_model.__class__.__bases__[0], 'validation_step') as mock_parent_val:
            mock_parent_val.return_value = {"val_loss": 0.5}
            
            # BallDetNet format
            frames = torch.randn(2, 5, 3, 64, 64)
            heatmaps = torch.randn(2, 5, 64, 64)
            batch = (frames, heatmaps, torch.ones(2, 5), [{"id": 1}, {"id": 2}])
            
            # Mock BallDetNet output
            lit_model.model.forward.return_value = torch.randn(2, 5, 64, 64)
            
            result = lit_model.validation_step(batch, batch_idx=0)
            
            # Check that middle frame was selected for visualization
            assert lit_model._validation_outputs['frames'].shape == (2, 3, 64, 64)
            assert lit_model._validation_outputs['preds'].shape == (2, 1, 64, 64)
            assert lit_model._validation_outputs['targets'].shape == (2, 1, 64, 64)
    
    def test_validation_step_no_logging(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test validation step with logging disabled."""
        logging_config = {"save_validation_outputs": False}
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            logging_config=logging_config
        )
        
        with patch.object(lit_model.__class__.__bases__[0], 'validation_step') as mock_parent_val:
            mock_parent_val.return_value = {"val_loss": 0.5}
            
            frames = torch.randn(2, 3, 64, 64)
            heatmaps = torch.randn(2, 64, 64)
            batch = (frames, heatmaps, torch.ones(2), [{"id": 1}, {"id": 2}])
            
            result = lit_model.validation_step(batch, batch_idx=0)
            
            # Should not store validation outputs
            assert not hasattr(lit_model, '_validation_outputs') or lit_model._validation_outputs is None
    
    def test_validation_step_non_first_batch(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test validation step for non-first batch (should not store outputs)."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        with patch.object(lit_model.__class__.__bases__[0], 'validation_step') as mock_parent_val:
            mock_parent_val.return_value = {"val_loss": 0.5}
            
            frames = torch.randn(2, 3, 64, 64)
            heatmaps = torch.randn(2, 64, 64)
            batch = (frames, heatmaps, torch.ones(2), [{"id": 1}, {"id": 2}])
            
            # Test with batch_idx > 0
            result = lit_model.validation_step(batch, batch_idx=1)
            
            # Should not store validation outputs for non-first batch
            assert not hasattr(lit_model, '_validation_outputs') or lit_model._validation_outputs is None
    
    def test_save_validation_images(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test saving validation images."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock logger with log_dir
            mock_logger = MagicMock()
            mock_logger.log_dir = temp_dir
            lit_model.logger = mock_logger
            lit_model.current_epoch = 5
            
            # Set validation outputs
            lit_model._validation_outputs = {
                "frames": torch.randn(4, 3, 64, 64),
                "preds": torch.sigmoid(torch.randn(4, 1, 64, 64)),
                "targets": torch.randn(4, 1, 64, 64),
                "batch": None,
                "outputs": None,
                "loss": None
            }
            
            with patch('torchvision.utils.save_image') as mock_save_image:
                lit_model._save_validation_images()
                
                # Check that images were saved
                assert mock_save_image.called
                
                # Check that validation images directory was created
                save_dir = os.path.join(temp_dir, "validation_images")
                assert os.path.exists(save_dir)
    
    def test_save_validation_images_no_outputs(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test saving validation images with no outputs."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # No validation outputs
        lit_model._validation_outputs = None
        
        with patch('torchvision.utils.save_image') as mock_save_image:
            lit_model._save_validation_images()
            
            # Should not save anything
            mock_save_image.assert_not_called()
    
    def test_save_validation_images_no_logger(self, mock_model, criterion_config, optimizer_config, scheduler_config):
        """Test saving validation images with no logger."""
        lit_model = LitBallModel(
            model=mock_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Set validation outputs but no logger
        lit_model._validation_outputs = {
            "frames": torch.randn(2, 3, 64, 64),
            "preds": torch.randn(2, 1, 64, 64),
            "targets": torch.randn(2, 1, 64, 64)
        }
        lit_model.logger = None
        
        with patch('torchvision.utils.save_image') as mock_save_image:
            lit_model._save_validation_images()
            
            # Should not save anything
            mock_save_image.assert_not_called()
    
    @pytest.mark.slow
    def test_forward_pass_integration(self, criterion_config, optimizer_config, scheduler_config):
        """Test integration with real model forward pass."""
        # Create simple real model
        real_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )
        
        lit_model = LitBallModel(
            model=real_model,
            criterion=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )
        
        # Test forward pass
        frames = torch.randn(2, 3, 64, 64)
        outputs = lit_model(frames)
        
        assert outputs.shape == (2, 1, 64, 64)
        assert torch.isfinite(outputs).all()
        
        # Test loss computation
        heatmaps = torch.randn(2, 64, 64)
        batch = (frames, heatmaps, torch.ones(2), [{"id": 1}, {"id": 2}])
        
        loss = lit_model.compute_loss(batch, outputs, "train")
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_different_criterion_configs(self, mock_model, optimizer_config, scheduler_config):
        """Test with different criterion configurations."""
        # Test with different alpha/gamma values
        criterion_configs = [
            {"alpha": 0.5, "gamma": 1.0, "reduction": "sum"},
            {"alpha": 1.0, "gamma": 3.0, "reduction": "none"},
            {"alpha": 0.1, "gamma": 0.5, "reduction": "mean"},
        ]
        
        for criterion_config in criterion_configs:
            lit_model = LitBallModel(
                model=mock_model,
                criterion=criterion_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config
            )
            
            # Test that configuration is stored
            assert lit_model.criterion_config == criterion_config
            
            # Test loss computation
            frames = torch.randn(1, 3, 32, 32)
            heatmaps = torch.randn(1, 32, 32)
            batch = (frames, heatmaps, torch.ones(1), [{"id": 1}])
            outputs = torch.randn(1, 32, 32)
            
            with patch('torchvision.ops.sigmoid_focal_loss') as mock_focal_loss:
                mock_focal_loss.return_value = torch.tensor(0.5)
                
                lit_model.compute_loss(batch, outputs, "train")
                
                # Check that correct parameters were passed
                call_args = mock_focal_loss.call_args
                assert call_args[1]['alpha'] == criterion_config['alpha']
                assert call_args[1]['gamma'] == criterion_config['gamma']
                assert call_args[1]['reduction'] == criterion_config['reduction']