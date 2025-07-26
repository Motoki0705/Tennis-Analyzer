"""Integration tests for checkpoint loading and model restoration."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
import time

from src.ball.models.conv3d_tsm_fpn import BallDetNet
from src.ball.models.lite_tracknet import LiteTrackNet
from src.ball.tests.fixtures.mock_data import create_mock_checkpoint


@pytest.mark.integration
class TestCheckpointLoadingIntegration:
    """Integration tests for model checkpoint loading and restoration."""
    
    @pytest.fixture
    def resource_limits(self):
        """Define resource limits for checkpoint tests."""
        return {
            "max_checkpoint_size_mb": 50,  # 50MB max checkpoint size
            "max_loading_time_seconds": 10,  # 10 second loading timeout
        }
    
    def test_balldetnet_checkpoint_save_and_load(self, resource_limits):
        """Test BallDetNet checkpoint saving and loading cycle."""
        # Create model
        model = BallDetNet(num_frames=3, num_classes=1)
        
        # Create some test data for forward pass verification
        test_input = torch.randn(1, 3, 3, 128, 256)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "balldetnet_test.ckpt"
            
            # Get initial output for comparison
            model.eval()
            with torch.no_grad():
                initial_output = model(test_input)
            
            # Create checkpoint data
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": 10,
                "global_step": 500,
                "pytorch-lightning_version": "2.0.0",
                "hyper_parameters": {
                    "num_frames": 3,
                    "num_classes": 1,
                    "learning_rate": 0.001
                },
                "optimizer_states": [{}],
                "lr_schedulers": []
            }
            
            # Save checkpoint
            start_time = time.time()
            torch.save(checkpoint, checkpoint_path)
            save_time = time.time() - start_time
            
            # Check checkpoint size
            checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            assert checkpoint_size_mb < resource_limits["max_checkpoint_size_mb"], \
                f"Checkpoint too large: {checkpoint_size_mb:.1f}MB"
            
            # Create new model instance
            new_model = BallDetNet(num_frames=3, num_classes=1)
            
            # Load checkpoint
            start_time = time.time()
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            load_time = time.time() - start_time
            
            assert load_time < resource_limits["max_loading_time_seconds"], \
                f"Loading too slow: {load_time:.1f}s"
            
            # Restore model state
            new_model.load_state_dict(loaded_checkpoint["state_dict"])
            
            # Verify restoration
            new_model.eval()
            with torch.no_grad():
                restored_output = new_model(test_input)
            
            # Outputs should be identical
            assert torch.allclose(initial_output, restored_output, atol=1e-6), \
                "Model output changed after checkpoint loading"
            
            # Verify metadata
            assert loaded_checkpoint["epoch"] == 10
            assert loaded_checkpoint["global_step"] == 500
            assert loaded_checkpoint["hyper_parameters"]["num_frames"] == 3
            
            print(f"BallDetNet checkpoint: {checkpoint_size_mb:.1f}MB, "
                  f"save: {save_time:.2f}s, load: {load_time:.2f}s")
    
    def test_lite_tracknet_checkpoint_save_and_load(self, resource_limits):
        """Test LiteTrackNet checkpoint saving and loading cycle."""
        # Create model
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Create test data
        test_input = torch.randn(1, 9, 128, 256)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "lite_tracknet_test.ckpt"
            
            # Get initial output
            model.eval()
            with torch.no_grad():
                initial_output = model(test_input)
            
            # Create and save checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": 25,
                "global_step": 1250,
                "pytorch-lightning_version": "2.0.0",
                "hyper_parameters": {
                    "in_channels": 9,
                    "out_channels": 1,
                    "architecture": "LiteTrackNet"
                }
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Verify checkpoint size
            checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            assert checkpoint_size_mb < resource_limits["max_checkpoint_size_mb"], \
                f"LiteTrackNet checkpoint too large: {checkpoint_size_mb:.1f}MB"
            
            # Load and restore
            new_model = LiteTrackNet(in_channels=9, out_channels=1)
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            new_model.load_state_dict(loaded_checkpoint["state_dict"])
            
            # Verify restoration
            new_model.eval()
            with torch.no_grad():
                restored_output = new_model(test_input)
            
            assert torch.allclose(initial_output, restored_output, atol=1e-6), \
                "LiteTrackNet output changed after checkpoint loading"
    
    def test_partial_checkpoint_loading(self):
        """Test loading checkpoints with missing or extra parameters."""
        model = BallDetNet(num_frames=3, num_classes=1)
        original_state_dict = model.state_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create checkpoint with missing parameters
            partial_state_dict = {k: v for k, v in original_state_dict.items() 
                                 if not k.startswith('head.')}  # Remove head parameters
            
            partial_checkpoint = {
                "state_dict": partial_state_dict,
                "epoch": 5,
                "hyper_parameters": {"num_frames": 3, "num_classes": 1}
            }
            
            checkpoint_path = Path(temp_dir) / "partial_checkpoint.ckpt"
            torch.save(partial_checkpoint, checkpoint_path)
            
            # Load partial checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # This should work with strict=False
            new_model = BallDetNet(num_frames=3, num_classes=1)
            missing_keys, unexpected_keys = new_model.load_state_dict(
                loaded_checkpoint["state_dict"], strict=False
            )
            
            # Should have missing keys (head parameters)
            assert len(missing_keys) > 0, "Should have missing keys"
            assert all('head.' in key for key in missing_keys), "Missing keys should be from head"
            assert len(unexpected_keys) == 0, "Should not have unexpected keys"
            
            # Model should still be functional (head will be randomly initialized)
            test_input = torch.randn(1, 3, 3, 64, 128)
            with torch.no_grad():
                output = new_model(test_input)
            assert output.shape == (1, 3, 64, 128), "Model should produce correct output shape"
    
    def test_checkpoint_compatibility_across_versions(self):
        """Test checkpoint loading compatibility across different scenarios."""
        # Test with different model configurations
        configs = [
            {"num_frames": 3, "num_classes": 1},
            {"num_frames": 5, "num_classes": 1},
            {"num_frames": 3, "num_classes": 2},
        ]
        
        checkpoints_data = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create checkpoints for different configurations
            for i, config in enumerate(configs):
                model = BallDetNet(**config)
                checkpoint_path = Path(temp_dir) / f"model_config_{i}.ckpt"
                
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "hyper_parameters": config,
                    "epoch": i + 1
                }
                
                torch.save(checkpoint, checkpoint_path)
                checkpoints_data[i] = (checkpoint_path, config)
            
            # Test loading compatible checkpoints
            for i, (checkpoint_path, config) in checkpoints_data.items():
                # Load checkpoint
                loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Create model with same config
                model = BallDetNet(**config)
                model.load_state_dict(loaded_checkpoint["state_dict"])
                
                # Should work without issues
                test_input = torch.randn(1, config["num_frames"], 3, 64, 128)
                with torch.no_grad():
                    output = model(test_input)
                
                expected_shape = (1, config["num_frames"], 64, 128) if config["num_classes"] == 1 else \
                                (1, config["num_frames"], config["num_classes"], 64, 128)
                assert output.shape == expected_shape, f"Wrong output shape for config {config}"
    
    def test_checkpoint_loading_with_device_transfer(self):
        """Test checkpoint loading with device transfers (CPU)."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "device_test.ckpt"
            
            # Save checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "hyper_parameters": {"in_channels": 9, "out_channels": 1}
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Test loading to different devices (CPU only in this environment)
            device_configs = [
                {"map_location": 'cpu'},
                {"map_location": torch.device('cpu')},
            ]
            
            for device_config in device_configs:
                # Load checkpoint with device mapping
                loaded_checkpoint = torch.load(checkpoint_path, **device_config)
                
                # Create model and load
                new_model = LiteTrackNet(in_channels=9, out_channels=1)
                new_model.load_state_dict(loaded_checkpoint["state_dict"])
                
                # Verify all parameters are on CPU
                for param in new_model.parameters():
                    assert param.device.type == 'cpu', f"Parameter not on CPU: {param.device}"
                
                # Test functionality
                test_input = torch.randn(1, 9, 64, 128)
                with torch.no_grad():
                    output = new_model(test_input)
                assert output.device.type == 'cpu', "Output not on CPU"
    
    def test_corrupted_checkpoint_handling(self):
        """Test handling of corrupted or invalid checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test various corruption scenarios
            
            # Empty file
            empty_file = Path(temp_dir) / "empty.ckpt"
            empty_file.touch()
            
            with pytest.raises(Exception):  # Should raise some exception
                torch.load(empty_file)
            
            # Invalid format file
            invalid_file = Path(temp_dir) / "invalid.ckpt"
            with open(invalid_file, 'w') as f:
                f.write("This is not a valid checkpoint")
            
            with pytest.raises(Exception):
                torch.load(invalid_file)
            
            # Valid pickle but wrong structure
            wrong_structure_file = Path(temp_dir) / "wrong_structure.ckpt"
            torch.save({"wrong": "structure"}, wrong_structure_file)
            
            loaded = torch.load(wrong_structure_file)
            assert "state_dict" not in loaded  # Missing required key
            
            # Should handle gracefully when trying to load into model
            model = BallDetNet(num_frames=3, num_classes=1)
            with pytest.raises(KeyError):  # Should raise KeyError for missing state_dict
                model.load_state_dict(loaded["state_dict"])
    
    def test_checkpoint_metadata_preservation(self):
        """Test that checkpoint metadata is preserved correctly."""
        model = BallDetNet(num_frames=5, num_classes=2)
        
        # Create comprehensive metadata
        metadata = {
            "epoch": 42,
            "global_step": 2100,
            "best_val_loss": 0.123,
            "pytorch-lightning_version": "2.0.0",
            "timestamp": "2025-07-25T10:00:00",
            "hyper_parameters": {
                "num_frames": 5,
                "num_classes": 2,
                "learning_rate": 0.001,
                "batch_size": 16,
                "optimizer": "Adam",
                "scheduler": "CosineAnnealing"
            },
            "train_config": {
                "max_epochs": 100,
                "early_stopping_patience": 10,
                "gradient_clip_val": 1.0
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "metadata_test.ckpt"
            
            # Create checkpoint with metadata
            checkpoint = {
                "state_dict": model.state_dict(),
                **metadata
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Load and verify all metadata is preserved
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            for key, expected_value in metadata.items():
                assert key in loaded_checkpoint, f"Missing metadata key: {key}"
                assert loaded_checkpoint[key] == expected_value, \
                    f"Metadata mismatch for {key}: expected {expected_value}, got {loaded_checkpoint[key]}"
            
            # Verify state_dict is also present
            assert "state_dict" in loaded_checkpoint
            assert len(loaded_checkpoint["state_dict"]) > 0
    
    @pytest.mark.slow
    def test_large_model_checkpoint_efficiency(self):
        """Test checkpoint efficiency with larger models."""
        # Create larger model for stress testing
        model = BallDetNet(num_frames=8, num_classes=3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "large_model.ckpt"
            
            # Measure save time
            start_time = time.time()
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": 100,
                "hyper_parameters": {"num_frames": 8, "num_classes": 3}
            }
            torch.save(checkpoint, checkpoint_path)
            save_time = time.time() - start_time
            
            # Measure load time
            start_time = time.time()
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            load_time = time.time() - start_time
            
            # Check efficiency benchmarks
            checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            
            # Reasonable performance expectations
            assert save_time < 5.0, f"Save too slow: {save_time:.2f}s"
            assert load_time < 3.0, f"Load too slow: {load_time:.2f}s"
            assert checkpoint_size_mb < 100, f"Checkpoint too large: {checkpoint_size_mb:.1f}MB"
            
            # Verify functionality after loading
            new_model = BallDetNet(num_frames=8, num_classes=3)
            new_model.load_state_dict(loaded_checkpoint["state_dict"])
            
            test_input = torch.randn(1, 8, 3, 128, 256)
            with torch.no_grad():
                output = new_model(test_input)
            
            assert output.shape == (1, 8, 3, 128, 256), "Large model output shape incorrect"
            
            print(f"Large model checkpoint: {checkpoint_size_mb:.1f}MB, "
                  f"save: {save_time:.2f}s, load: {load_time:.2f}s")