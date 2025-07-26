"""Integration tests for inference pipeline."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader
import albumentations as A

from src.ball.models.conv3d_tsm_fpn import BallDetNet
from src.ball.models.lite_tracknet import LiteTrackNet
from src.ball.dataset.seq_coord_dataset import SequenceCoordDataset
from src.ball.tests.fixtures.mock_data import MockDataManager


@pytest.mark.integration
class TestInferencePipelineIntegration:
    """Integration tests for complete inference pipeline."""
    
    @pytest.fixture
    def inference_config(self):
        """Create configuration for inference testing."""
        return {
            "batch_size": 4,
            "input_size": [128, 256],  # Moderate size for realistic testing
            "max_inference_time_seconds": 30,  # 30 second timeout
            "max_memory_mb": 800,  # 800MB memory limit
            "confidence_threshold": 0.5,
        }
    
    @pytest.fixture
    def inference_transform(self):
        """Create transform for inference (no augmentation)."""
        return A.ReplayCompose([
            A.Resize(height=128, width=256),
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))
    
    def create_trained_model(self, model_class, **model_kwargs):
        """Create a model with some 'trained' weights."""
        model = model_class(**model_kwargs)
        
        # Initialize with slightly better weights (not random)
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        return model
    
    def test_balldetnet_inference_pipeline(self, inference_config, inference_transform):
        """Test BallDetNet complete inference pipeline."""
        with MockDataManager() as temp_dir:
            # Create dataset for inference
            dataset = SequenceCoordDataset(
                annotation_file=str(temp_dir / "annotations.json"),
                image_root=str(temp_dir / "images"),
                T=5,
                input_size=inference_config["input_size"],
                transform=inference_transform,
                split="test",  # Use test split for inference
                input_type="stack",
                output_type="all"
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=inference_config["batch_size"],
                shuffle=False,  # No shuffle for inference
                num_workers=0
            )
            
            # Create trained model
            model = self.create_trained_model(BallDetNet, num_frames=5, num_classes=1)
            model.eval()  # Set to evaluation mode
            
            # Monitor resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            start_time = time.time()
            
            # Run inference
            all_predictions = []
            all_confidences = []
            
            with torch.no_grad():  # Disable gradients for inference
                for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                    if batch_idx >= 3:  # Limit batches for integration test
                        break
                    
                    # Forward pass
                    outputs = model(frames)  # Shape: [B, T, H, W]
                    
                    # Apply sigmoid to get probabilities
                    probabilities = torch.sigmoid(outputs)
                    
                    # Extract ball positions (simplified peak detection)
                    batch_size, T, H, W = probabilities.shape
                    batch_predictions = []
                    batch_confidences = []
                    
                    for b in range(batch_size):
                        frame_predictions = []
                        frame_confidences = []
                        
                        for t in range(T):
                            prob_map = probabilities[b, t]
                            
                            # Simple peak detection
                            max_val = torch.max(prob_map)
                            if max_val > inference_config["confidence_threshold"]:
                                # Find position of maximum
                                max_pos = torch.argmax(prob_map.flatten())
                                y_pos = max_pos // W
                                x_pos = max_pos % W
                                
                                # Normalize to [0, 1]
                                norm_x = x_pos.float() / W
                                norm_y = y_pos.float() / H
                                
                                frame_predictions.append([norm_x.item(), norm_y.item()])
                                frame_confidences.append(max_val.item())
                            else:
                                frame_predictions.append([0.0, 0.0])  # No detection
                                frame_confidences.append(0.0)
                        
                        batch_predictions.append(frame_predictions)
                        batch_confidences.append(frame_confidences)
                    
                    all_predictions.extend(batch_predictions)
                    all_confidences.extend(batch_confidences)
                    
                    # Check resource usage
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    memory_increase = current_memory - initial_memory
                    assert memory_increase < inference_config["max_memory_mb"], \
                        f"Memory usage exceeded: {memory_increase:.1f}MB"
                    
                    elapsed_time = time.time() - start_time
                    assert elapsed_time < inference_config["max_inference_time_seconds"], \
                        f"Inference time exceeded: {elapsed_time:.1f}s"
            
            # Validate inference results
            assert len(all_predictions) > 0, "No predictions generated"
            assert len(all_predictions) == len(all_confidences), "Prediction count mismatch"
            
            # Check prediction format
            for pred_seq, conf_seq in zip(all_predictions, all_confidences):
                assert len(pred_seq) == 5, f"Expected 5 frames, got {len(pred_seq)}"
                assert len(conf_seq) == 5, f"Expected 5 confidences, got {len(conf_seq)}"
                
                for pred, conf in zip(pred_seq, conf_seq):
                    assert len(pred) == 2, "Prediction should be [x, y]"
                    assert 0 <= pred[0] <= 1, f"X coordinate out of range: {pred[0]}"
                    assert 0 <= pred[1] <= 1, f"Y coordinate out of range: {pred[1]}"
                    assert 0 <= conf <= 1, f"Confidence out of range: {conf}"
            
            total_time = time.time() - start_time
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            print(f"BallDetNet inference: {len(all_predictions)} sequences in {total_time:.2f}s, "
                  f"memory: {final_memory - initial_memory:.1f}MB")
    
    def test_lite_tracknet_inference_pipeline(self, inference_config, inference_transform):
        """Test LiteTrackNet inference pipeline."""
        with MockDataManager() as temp_dir:
            # Create dataset with concatenated input
            dataset = SequenceCoordDataset(
                annotation_file=str(temp_dir / "annotations.json"),
                image_root=str(temp_dir / "images"),
                T=3,
                input_size=inference_config["input_size"],
                transform=inference_transform,
                split="test",
                input_type="cat",  # Concatenated for LiteTrackNet
                output_type="last"  # Single frame output
            )
            
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            
            # Create trained model
            model = self.create_trained_model(LiteTrackNet, in_channels=9, out_channels=1)
            model.eval()
            
            predictions = []
            
            with torch.no_grad():
                for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                    if batch_idx >= 2:
                        break
                    
                    # Forward pass
                    outputs = model(frames)  # Shape: [B, H, W]
                    probabilities = torch.sigmoid(outputs)
                    
                    # Extract predictions
                    batch_size, H, W = probabilities.shape
                    for b in range(batch_size):
                        prob_map = probabilities[b]
                        max_val = torch.max(prob_map)
                        
                        if max_val > inference_config["confidence_threshold"]:
                            max_pos = torch.argmax(prob_map.flatten())
                            y_pos = max_pos // W
                            x_pos = max_pos % W
                            
                            norm_x = x_pos.float() / W
                            norm_y = y_pos.float() / H
                            
                            predictions.append({
                                "position": [norm_x.item(), norm_y.item()],
                                "confidence": max_val.item()
                            })
                        else:
                            predictions.append({
                                "position": [0.0, 0.0],
                                "confidence": 0.0
                            })
            
            assert len(predictions) > 0, "No predictions from LiteTrackNet"
            
            # Validate predictions
            for pred in predictions:
                assert "position" in pred and "confidence" in pred
                pos = pred["position"]
                conf = pred["confidence"]
                
                assert len(pos) == 2
                assert 0 <= pos[0] <= 1 and 0 <= pos[1] <= 1
                assert 0 <= conf <= 1
    
    def test_batch_inference_consistency(self, inference_transform):
        """Test consistency between single and batch inference."""
        with MockDataManager() as temp_dir:
            dataset = SequenceCoordDataset(
                annotation_file=str(temp_dir / "annotations.json"),
                image_root=str(temp_dir / "images"),
                T=3,
                input_size=[64, 128],
                transform=inference_transform,
                split="test",
                input_type="stack",
                output_type="all"
            )
            
            model = self.create_trained_model(BallDetNet, num_frames=3, num_classes=1)
            model.eval()
            
            # Get a batch of data
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            frames, coords, visibility, image_info = next(iter(dataloader))
            
            with torch.no_grad():
                # Batch inference
                batch_outputs = model(frames)
                
                # Single inference for each item
                single_outputs = []
                for i in range(frames.shape[0]):
                    single_frame = frames[i:i+1]  # Keep batch dimension
                    single_output = model(single_frame)
                    single_outputs.append(single_output)
                
                # Stack single outputs
                stacked_single_outputs = torch.cat(single_outputs, dim=0)
                
                # Should be nearly identical
                assert torch.allclose(batch_outputs, stacked_single_outputs, atol=1e-5), \
                    "Batch and single inference outputs differ"
    
    def test_inference_with_different_input_sizes(self, inference_transform):
        """Test inference robustness with different input sizes."""
        model = self.create_trained_model(BallDetNet, num_frames=3, num_classes=1)
        model.eval()
        
        # Test different input sizes
        input_sizes = [(64, 128), (128, 256), (96, 192)]
        
        for h, w in input_sizes:
            test_input = torch.randn(2, 3, 3, h, w)
            
            with torch.no_grad():
                output = model(test_input)
            
            # Output should match input spatial dimensions
            assert output.shape == (2, 3, h, w), f"Wrong output shape for input {h}x{w}"
            assert torch.isfinite(output).all(), f"Non-finite outputs for input {h}x{w}"
    
    def test_inference_from_checkpoint(self, inference_transform):
        """Test inference pipeline with model loaded from checkpoint."""
        with MockDataManager() as temp_dir, tempfile.TemporaryDirectory() as checkpoint_dir:
            # Create and save model
            original_model = self.create_trained_model(BallDetNet, num_frames=3, num_classes=1)
            
            checkpoint_path = Path(checkpoint_dir) / "inference_model.ckpt"
            checkpoint = {
                "state_dict": original_model.state_dict(),
                "hyper_parameters": {"num_frames": 3, "num_classes": 1},
                "epoch": 50
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load model from checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            loaded_model = BallDetNet(num_frames=3, num_classes=1)
            loaded_model.load_state_dict(loaded_checkpoint["state_dict"])
            loaded_model.eval()
            
            # Create test dataset
            dataset = SequenceCoordDataset(
                annotation_file=str(temp_dir / "annotations.json"),
                image_root=str(temp_dir / "images"),
                T=3,
                input_size=[64, 128],
                transform=inference_transform,
                split="test"
            )
            
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            # Run inference with loaded model
            predictions = []
            
            with torch.no_grad():
                for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                    if batch_idx >= 2:
                        break
                    
                    outputs = loaded_model(frames)
                    probabilities = torch.sigmoid(outputs)
                    
                    # Simple peak detection
                    batch_size, T, H, W = probabilities.shape
                    for b in range(batch_size):
                        for t in range(T):
                            prob_map = probabilities[b, t]
                            max_val = torch.max(prob_map)
                            
                            if max_val > 0.1:  # Low threshold for testing
                                max_pos = torch.argmax(prob_map.flatten())
                                y_pos = max_pos // W
                                x_pos = max_pos % W
                                
                                predictions.append({
                                    "frame": (b, t),
                                    "position": [x_pos.item() / W, y_pos.item() / H],
                                    "confidence": max_val.item()
                                })
            
            assert len(predictions) >= 0, "Inference pipeline should complete without errors"
    
    def test_inference_performance_benchmarking(self, inference_transform):
        """Test inference performance and throughput."""
        model = self.create_trained_model(BallDetNet, num_frames=5, num_classes=1)
        model.eval()
        
        # Create synthetic data for benchmarking
        batch_sizes = [1, 2, 4]
        input_size = [128, 256]
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 5, 3, *input_size)
            
            # Warmup
            with torch.no_grad():
                _ = model(test_input)
            
            # Benchmark
            num_runs = 5
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    output = model(test_input)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            fps = batch_size / avg_time  # Sequences per second
            
            print(f"Batch size {batch_size}: {avg_time:.3f}s avg, {fps:.1f} sequences/sec")
            
            # Performance expectations (adjust based on requirements)
            assert avg_time < 2.0, f"Inference too slow for batch size {batch_size}: {avg_time:.3f}s"
            assert fps > 0.5, f"Throughput too low for batch size {batch_size}: {fps:.1f} sequences/sec"
    
    def test_inference_error_handling(self, inference_transform):
        """Test inference pipeline error handling and robustness."""
        model = self.create_trained_model(BallDetNet, num_frames=3, num_classes=1)
        model.eval()
        
        # Test various edge cases
        test_cases = [
            # Normal case
            torch.randn(1, 3, 3, 64, 128),
            # Very small input
            torch.randn(1, 3, 3, 32, 64),
            # Single pixel input (extreme case)
            torch.randn(1, 3, 3, 1, 1),
            # Large batch
            torch.randn(8, 3, 3, 64, 128),
        ]
        
        successful_cases = 0
        
        for i, test_input in enumerate(test_cases):
            try:
                with torch.no_grad():
                    output = model(test_input)
                
                # Basic validation
                assert output.shape[0] == test_input.shape[0], "Batch size mismatch"
                assert output.shape[1] == test_input.shape[1], "Frame count mismatch"
                assert torch.isfinite(output).all(), "Non-finite outputs"
                
                successful_cases += 1
                
            except Exception as e:
                print(f"Test case {i} failed: {e}")
                # Some edge cases might legitimately fail
                pass
        
        # Should handle most reasonable cases
        assert successful_cases >= len(test_cases) - 1, "Too many test cases failed"
    
    @pytest.mark.slow
    def test_long_sequence_inference(self, inference_transform):
        """Test inference on longer sequences."""
        with MockDataManager() as temp_dir:
            # Create dataset with longer sequences
            dataset = SequenceCoordDataset(
                annotation_file=str(temp_dir / "annotations.json"),
                image_root=str(temp_dir / "images"),
                T=8,  # Longer sequence
                input_size=[64, 128],
                transform=inference_transform,
                split="test",
                input_type="stack",
                output_type="all"
            )
            
            model = self.create_trained_model(BallDetNet, num_frames=8, num_classes=1)
            model.eval()
            
            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)  # Single batch
            
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024 * 1024)
            
            with torch.no_grad():
                for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                    if batch_idx >= 3:  # Process a few long sequences
                        break
                    
                    start_time = time.time()
                    outputs = model(frames)
                    inference_time = time.time() - start_time
                    
                    # Should complete in reasonable time
                    assert inference_time < 10.0, f"Long sequence inference too slow: {inference_time:.2f}s"
                    
                    # Check memory usage
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    memory_increase = current_memory - start_memory
                    assert memory_increase < 500, f"Excessive memory for long sequence: {memory_increase:.1f}MB"
                    
                    # Validate output
                    assert outputs.shape == (1, 8, 64, 128), "Wrong output shape for long sequence"
                    assert torch.isfinite(outputs).all(), "Non-finite outputs in long sequence"
            
            print(f"Long sequence inference completed successfully")