"""Integration tests for lightweight training pipeline."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
import time
import psutil
import tempfile
from pathlib import Path

from src.ball.models.conv3d_tsm_fpn import BallDetNet, SigmoidFocalLoss
from src.ball.models.lite_tracknet import LiteTrackNet
from src.ball.dataset.seq_coord_dataset import SequenceCoordDataset
from src.ball.tests.fixtures.mock_data import MockDataManager


@pytest.mark.integration
class TestTrainingPipelineIntegration:
    """Integration tests for complete lightweight training pipeline."""
    
    @pytest.fixture
    def training_config(self):
        """Create lightweight training configuration."""
        return {
            "num_epochs": 2,  # Very short training for integration test
            "batch_size": 2,
            "learning_rate": 0.01,
            "input_size": [64, 128],  # Small size for fast training
            "T": 3,  # Few frames for speed
            "max_time_seconds": 60,  # 1 minute timeout
            "max_memory_mb": 1000,  # 1GB memory limit
        }
    
    @pytest.fixture
    def simple_transform(self):
        """Create simple transform for training tests."""
        return A.ReplayCompose([
            A.Resize(height=64, width=128),
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))
    
    def create_simple_coord_dataset(self, data_dir, transform, split="train"):
        """Create simple coordinate dataset for training."""
        return SequenceCoordDataset(
            annotation_file=str(data_dir / "annotations.json"),
            image_root=str(data_dir / "images"),
            T=3,
            input_size=[64, 128],
            transform=transform,
            split=split,
            input_type="stack",
            output_type="all"
        )
    
    def test_balldetnet_training_loop(self, training_config, simple_transform):
        """Test BallDetNet end-to-end training loop."""
        with MockDataManager() as temp_dir:
            # Create dataset and dataloader
            dataset = self.create_simple_coord_dataset(temp_dir, simple_transform)
            dataloader = DataLoader(
                dataset, 
                batch_size=training_config["batch_size"],
                shuffle=True,
                num_workers=0
            )
            
            # Create model and components
            model = BallDetNet(num_frames=training_config["T"], num_classes=1)
            criterion = SigmoidFocalLoss(alpha=2.0, beta=4.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
            
            # Monitor resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            start_time = time.time()
            
            # Training loop
            model.train()
            training_losses = []
            
            for epoch in range(training_config["num_epochs"]):
                epoch_losses = []
                batch_count = 0
                
                for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                    if batch_count >= 3:  # Limit batches for integration test
                        break
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(frames)
                    
                    # Create dummy heatmaps for loss calculation (since we have coords)
                    # This is a simplified approach for integration testing
                    batch_size, T, H, W = outputs.shape
                    dummy_heatmaps = torch.zeros_like(outputs)
                    
                    # Add some signal for non-zero loss
                    for b in range(batch_size):
                        for t in range(T):
                            if visibility[b, t] > 0:  # If ball is visible
                                x, y = coords[b, t]
                                # Convert normalized coords to pixel coords
                                px = int(x * W)
                                py = int(y * H)
                                # Add small gaussian around ball position
                                if 0 <= px < W and 0 <= py < H:
                                    dummy_heatmaps[b, t, py, px] = 1.0
                    
                    # Calculate loss
                    loss = criterion(outputs, dummy_heatmaps)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    batch_count += 1
                    
                    # Check resource usage
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    assert current_memory - initial_memory < training_config["max_memory_mb"], \
                        f"Memory usage exceeded: {current_memory - initial_memory:.1f}MB"
                    
                    # Check time limit
                    elapsed_time = time.time() - start_time
                    assert elapsed_time < training_config["max_time_seconds"], \
                        f"Training time exceeded: {elapsed_time:.1f}s"
                
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                training_losses.append(avg_loss)
                print(f"Epoch {epoch + 1}/{training_config['num_epochs']}, Loss: {avg_loss:.4f}")
            
            # Verify training completed successfully
            assert len(training_losses) == training_config["num_epochs"]
            assert all(isinstance(loss, float) and loss >= 0 for loss in training_losses)
            
            # Verify model is in correct state
            assert model.training is True  # Should still be in training mode
            
            # Test that model can switch to eval mode and make predictions
            model.eval()
            with torch.no_grad():
                test_frames = torch.randn(1, training_config["T"], 3, 64, 128)
                test_output = model(test_frames)
                assert test_output.shape == (1, training_config["T"], 64, 128)
                assert torch.isfinite(test_output).all()
    
    def test_lite_tracknet_training_loop(self, training_config, simple_transform):
        """Test LiteTrackNet training loop with different input format."""
        with MockDataManager() as temp_dir:
            # Create dataset with cat input type for LiteTrackNet
            dataset = SequenceCoordDataset(
                annotation_file=str(temp_dir / "annotations.json"),
                image_root=str(temp_dir / "images"),
                T=3,
                input_size=[64, 128],
                transform=simple_transform,
                split="train",
                input_type="cat",  # Concatenated input for LiteTrackNet
                output_type="last"  # Only last frame output
            )
            
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            # Create model (input channels = 3 * T)
            model = LiteTrackNet(in_channels=9, out_channels=1)  # 3 * 3 frames
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Training loop
            model.train()
            losses = []
            
            for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                if batch_idx >= 2:  # Limited batches for integration test
                    break
                
                optimizer.zero_grad()
                outputs = model(frames)  # frames shape: [B, 9, H, W]
                
                # Create simple target (binary mask)
                batch_size, H, W = outputs.shape
                targets = torch.zeros_like(outputs)
                
                # Add targets based on coordinates
                for b in range(batch_size):
                    if visibility[b] > 0:  # Last frame visibility
                        x, y = coords[b]  # Last frame coordinates
                        px, py = int(x * W), int(y * H)
                        if 0 <= px < W and 0 <= py < H:
                            targets[b, py, px] = 1.0
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            assert len(losses) > 0
            assert all(isinstance(loss, float) and loss >= 0 for loss in losses)
    
    def test_training_with_checkpointing(self, training_config, simple_transform):
        """Test training pipeline with checkpoint saving and loading."""
        with MockDataManager() as data_dir, tempfile.TemporaryDirectory() as checkpoint_dir:
            # Create dataset
            dataset = self.create_simple_coord_dataset(data_dir, simple_transform)
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            # Create model
            model = BallDetNet(num_frames=3, num_classes=1)
            criterion = SigmoidFocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            checkpoint_path = Path(checkpoint_dir) / "training_checkpoint.ckpt"
            
            # Training for first epoch
            model.train()
            initial_loss = None
            
            for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                if batch_idx >= 2:
                    break
                
                optimizer.zero_grad()
                outputs = model(frames)
                
                # Simple loss calculation
                dummy_targets = torch.zeros_like(outputs)
                loss = criterion(dummy_targets, dummy_targets)  # Zero loss for simplicity
                
                if initial_loss is None:
                    initial_loss = loss.item()
                
                loss.backward()
                optimizer.step()
            
            # Save checkpoint
            checkpoint = {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": initial_loss,
                "hyper_parameters": {"num_frames": 3, "num_classes": 1}
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create new model and load checkpoint
            new_model = BallDetNet(num_frames=3, num_classes=1)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
            
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
            new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
            
            # Verify models are equivalent
            test_input = torch.randn(1, 3, 3, 64, 128)
            model.eval()
            new_model.eval()
            
            with torch.no_grad():
                original_output = model(test_input)
                loaded_output = new_model(test_input)
            
            assert torch.allclose(original_output, loaded_output, atol=1e-6), \
                "Model outputs differ after checkpoint loading"
    
    def test_training_memory_management(self, simple_transform):
        """Test memory management during training."""
        with MockDataManager() as temp_dir:
            dataset = self.create_simple_coord_dataset(temp_dir, simple_transform)
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            model = BallDetNet(num_frames=3, num_classes=1)
            criterion = SigmoidFocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            process = psutil.Process()
            memory_samples = []
            
            model.train()
            
            for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                if batch_idx >= 5:  # Multiple batches to check memory behavior
                    break
                
                # Record memory before batch
                memory_before = process.memory_info().rss / (1024 * 1024)
                
                optimizer.zero_grad()
                outputs = model(frames)
                
                # Simple loss
                dummy_targets = torch.zeros_like(outputs)
                loss = criterion(outputs, dummy_targets)
                loss.backward()
                optimizer.step()
                
                # Explicit cleanup
                del outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Record memory after batch
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_samples.append((memory_before, memory_after))
            
            # Check that memory doesn't grow unboundedly
            if len(memory_samples) > 2:
                first_batch_memory = memory_samples[0][1]
                last_batch_memory = memory_samples[-1][1]
                memory_growth = last_batch_memory - first_batch_memory
                
                # Should not grow by more than 200MB during training
                assert memory_growth < 200, f"Excessive memory growth: {memory_growth:.1f}MB"
    
    def test_gradient_flow_and_optimization(self, simple_transform):
        """Test gradient flow and optimization during training."""
        with MockDataManager() as temp_dir:
            dataset = self.create_simple_coord_dataset(temp_dir, simple_transform)
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            model = BallDetNet(num_frames=3, num_classes=1)
            criterion = SigmoidFocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Get initial parameters for comparison
            initial_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.clone().detach()
            
            model.train()
            
            for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                if batch_idx >= 2:
                    break
                
                optimizer.zero_grad()
                outputs = model(frames)
                
                # Create meaningful loss (non-zero)
                batch_size, T, H, W = outputs.shape
                targets = torch.rand_like(outputs) * 0.1  # Small random targets
                
                loss = criterion(outputs, targets)
                assert loss.item() > 0, "Loss should be positive"
                
                loss.backward()
                
                # Check that gradients exist and are non-zero
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append(grad_norm)
                        assert grad_norm > 0, f"Zero gradient for {name}"
                
                assert len(grad_norms) > 0, "No gradients computed"
                
                optimizer.step()
            
            # Check that parameters actually changed
            params_changed = 0
            for name, param in model.named_parameters():
                if param.requires_grad and name in initial_params:
                    if not torch.allclose(param, initial_params[name], atol=1e-7):
                        params_changed += 1
            
            assert params_changed > 0, "No parameters were updated during training"
    
    def test_training_pipeline_error_recovery(self, simple_transform):
        """Test training pipeline robustness and error recovery."""
        with MockDataManager() as temp_dir:
            dataset = self.create_simple_coord_dataset(temp_dir, simple_transform)
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            model = BallDetNet(num_frames=3, num_classes=1)
            criterion = SigmoidFocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            model.train()
            successful_steps = 0
            
            for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                if batch_idx >= 3:
                    break
                
                try:
                    optimizer.zero_grad()
                    
                    # Simulate occasional issues but recover
                    if batch_idx == 1:
                        # Simulate NaN in input (should be handled gracefully)
                        test_frames = frames.clone()
                        if torch.rand(1).item() > 0.5:  # Randomly introduce issue
                            test_frames = torch.where(torch.rand_like(test_frames) > 0.99, 
                                                    torch.tensor(float('inf')), test_frames)
                        outputs = model(test_frames)
                    else:
                        outputs = model(frames)
                    
                    # Check for NaN/Inf in outputs
                    if not torch.isfinite(outputs).all():
                        print(f"Non-finite outputs detected in batch {batch_idx}, skipping...")
                        continue
                    
                    dummy_targets = torch.zeros_like(outputs)
                    loss = criterion(outputs, dummy_targets)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        successful_steps += 1
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed with error: {e}, continuing...")
                    continue
            
            # Should have completed at least some successful steps
            assert successful_steps > 0, "No successful training steps completed"
    
    @pytest.mark.slow
    def test_extended_training_stability(self, simple_transform):
        """Test training stability over extended period."""
        with MockDataManager() as temp_dir:
            dataset = self.create_simple_coord_dataset(temp_dir, simple_transform)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
            
            model = BallDetNet(num_frames=3, num_classes=1)
            criterion = SigmoidFocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lower LR for stability
            
            # Run multiple epochs
            model.train()
            all_losses = []
            
            for epoch in range(3):  # 3 epochs for extended test
                epoch_losses = []
                
                for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                    if batch_idx >= 4:  # More batches per epoch
                        break
                    
                    optimizer.zero_grad()
                    outputs = model(frames)
                    
                    # Gradually changing targets to test adaptation
                    targets = torch.rand_like(outputs) * (0.1 + epoch * 0.05)
                    loss = criterion(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
                all_losses.append(epoch_avg_loss)
                print(f"Extended training epoch {epoch + 1}: avg_loss = {epoch_avg_loss:.4f}")
            
            # Verify training remained stable
            assert len(all_losses) == 3
            assert all(loss > 0 and loss < 10 for loss in all_losses), "Losses out of reasonable range"
            
            # Check that model still functions correctly
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(1, 3, 3, 64, 128)
                final_output = model(test_input)
                assert torch.isfinite(final_output).all(), "Model outputs non-finite values after training"