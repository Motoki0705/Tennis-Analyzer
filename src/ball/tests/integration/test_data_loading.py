"""Integration tests for data loading pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader
import albumentations as A
import time
import psutil
import os
from pathlib import Path

from src.ball.dataset.seq_coord_dataset import SequenceCoordDataset
from src.ball.tests.fixtures.mock_data import MockDataManager


@pytest.mark.integration
class TestDataLoadingIntegration:
    """Integration tests for complete data loading pipeline."""
    
    @pytest.fixture
    def basic_transform(self):
        """Create basic transform for testing."""
        return A.ReplayCompose([
            A.Resize(height=128, width=256),  # Small size for fast testing
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))
    
    @pytest.fixture
    def resource_limits(self):
        """Define resource limits for integration tests."""
        return {
            "max_memory_mb": 500,  # 500MB memory limit
            "max_time_seconds": 30,  # 30 second timeout
            "max_batch_size": 4,   # Small batch for CPU testing
        }
    
    def test_dataset_creation_and_basic_loading(self, basic_transform):
        """Test dataset creation and basic data loading."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Create dataset
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[128, 256],
                transform=basic_transform,
                split="train"
            )
            
            # Test basic properties
            assert len(dataset) > 0, "Dataset should contain data"
            
            # Test single item loading
            frames, coords, visibility, image_info = dataset[0]
            
            # Validate output shapes and types
            assert isinstance(frames, torch.Tensor), "Frames should be tensor"
            assert isinstance(coords, torch.Tensor), "Coords should be tensor"
            assert isinstance(visibility, torch.Tensor), "Visibility should be tensor"
            assert isinstance(image_info, dict), "Image info should be dict"
            
            # Validate tensor shapes
            assert frames.shape == (3, 3, 128, 256), f"Expected (3,3,128,256), got {frames.shape}"
            assert coords.shape == (3, 2), f"Expected (3,2), got {coords.shape}"
            assert visibility.shape == (3,), f"Expected (3,), got {visibility.shape}"
    
    def test_dataloader_integration(self, basic_transform, resource_limits):
        """Test DataLoader integration with resource monitoring."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[128, 256],
                transform=basic_transform,
                split="train"
            )
            
            # Create DataLoader with resource-conscious settings
            dataloader = DataLoader(
                dataset,
                batch_size=resource_limits["max_batch_size"],
                shuffle=True,
                num_workers=0,  # Single-threaded for predictable resource usage
                pin_memory=False  # Reduce memory usage
            )
            
            # Monitor resource usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            start_time = time.time()
            
            batch_count = 0
            max_batches = 3  # Limit number of batches for integration test
            
            for batch_idx, (frames, coords, visibility, image_info) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                batch_count += 1
                
                # Validate batch shapes
                batch_size = frames.shape[0]
                assert frames.shape == (batch_size, 3, 3, 128, 256)
                assert coords.shape == (batch_size, 3, 2)
                assert visibility.shape == (batch_size, 3)
                
                # Check data types
                assert frames.dtype == torch.float32
                assert coords.dtype == torch.float32
                assert visibility.dtype == torch.float32
                
                # Check data ranges
                assert torch.all(coords >= 0) and torch.all(coords <= 1), "Coords should be normalized"
                assert torch.all(visibility >= 0) and torch.all(visibility <= 2), "Visibility should be COCO format"
                
                # Monitor memory usage
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_increase = current_memory - initial_memory
                assert memory_increase < resource_limits["max_memory_mb"], \
                    f"Memory usage increased by {memory_increase:.1f}MB (limit: {resource_limits['max_memory_mb']}MB)"
                
                # Check execution time
                elapsed_time = time.time() - start_time
                assert elapsed_time < resource_limits["max_time_seconds"], \
                    f"Execution time {elapsed_time:.1f}s exceeded limit {resource_limits['max_time_seconds']}s"
            
            assert batch_count > 0, "Should have processed at least one batch"
            
            # Final resource check
            final_time = time.time() - start_time
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            print(f"Integration test completed: {batch_count} batches in {final_time:.2f}s, "
                  f"peak memory: {final_memory - initial_memory:.1f}MB")
    
    def test_different_dataset_configurations(self, basic_transform):
        """Test data loading with different dataset configurations."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Test different configurations
            configs = [
                {"T": 3, "input_type": "stack", "output_type": "all"},
                {"T": 5, "input_type": "cat", "output_type": "last"},
                {"T": 2, "input_type": "stack", "output_type": "last"},
            ]
            
            for config in configs:
                dataset = SequenceCoordDataset(
                    annotation_file=str(annotation_file),
                    image_root=str(image_root),
                    input_size=[64, 128],  # Smaller for faster testing
                    transform=basic_transform,
                    split="train",
                    **config
                )
                
                dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
                
                # Get one batch to verify configuration
                frames, coords, visibility, image_info = next(iter(dataloader))
                batch_size = 2
                
                # Verify shapes based on configuration
                if config["input_type"] == "stack":
                    expected_frame_shape = (batch_size, config["T"], 3, 64, 128)
                else:  # cat
                    expected_frame_shape = (batch_size, 3 * config["T"], 64, 128)
                
                if config["output_type"] == "all":
                    expected_coord_shape = (batch_size, config["T"], 2)
                else:  # last
                    expected_coord_shape = (batch_size, 2)
                
                assert frames.shape == expected_frame_shape, \
                    f"Config {config}: expected {expected_frame_shape}, got {frames.shape}"
                assert coords.shape == expected_coord_shape, \
                    f"Config {config}: expected {expected_coord_shape}, got {coords.shape}"
    
    def test_data_consistency_across_epochs(self, basic_transform):
        """Test data consistency across multiple epochs."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[64, 128],
                transform=basic_transform,
                split="val",  # Use validation split for deterministic behavior
                skip_frames_range=(1, 1)  # No frame skipping for consistency
            )
            
            dataloader = DataLoader(
                dataset, 
                batch_size=2, 
                shuffle=False,  # No shuffle for consistency
                num_workers=0
            )
            
            # Collect data from first epoch
            epoch1_data = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 2:  # Limit for integration test
                    break
                epoch1_data.append(batch)
            
            # Collect data from second epoch
            epoch2_data = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 2:
                    break
                epoch2_data.append(batch)
            
            # Verify consistency (should be identical for validation set)
            assert len(epoch1_data) == len(epoch2_data)
            
            for i, (batch1, batch2) in enumerate(zip(epoch1_data, epoch2_data)):
                frames1, coords1, vis1, info1 = batch1
                frames2, coords2, vis2, info2 = batch2
                
                # Should be identical across epochs for validation
                assert torch.allclose(frames1, frames2, atol=1e-6), \
                    f"Frames not consistent across epochs for batch {i}"
                assert torch.allclose(coords1, coords2, atol=1e-6), \
                    f"Coords not consistent across epochs for batch {i}"
                assert torch.allclose(vis1, vis2, atol=1e-6), \
                    f"Visibility not consistent across epochs for batch {i}"
    
    def test_error_recovery_and_robustness(self, basic_transform):
        """Test data loading robustness and error recovery."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[64, 128],
                transform=basic_transform,
                split="train"
            )
            
            # Test with different DataLoader configurations
            dataloader_configs = [
                {"batch_size": 1, "shuffle": True, "drop_last": False},
                {"batch_size": 3, "shuffle": False, "drop_last": True},
                {"batch_size": 2, "shuffle": True, "drop_last": False},
            ]
            
            for config in dataloader_configs:
                dataloader = DataLoader(dataset, num_workers=0, **config)
                
                # Should be able to iterate without errors
                batch_count = 0
                try:
                    for batch in dataloader:
                        batch_count += 1
                        if batch_count >= 2:  # Limit for integration test
                            break
                        
                        # Basic validation
                        frames, coords, visibility, image_info = batch
                        assert frames.numel() > 0, "Frames should not be empty"
                        assert coords.numel() > 0, "Coords should not be empty"
                        
                except Exception as e:
                    pytest.fail(f"DataLoader failed with config {config}: {e}")
                
                assert batch_count > 0, f"No batches processed with config {config}"
    
    @pytest.mark.slow
    def test_memory_leak_detection(self, basic_transform):
        """Test for memory leaks during extended data loading."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[64, 128],
                transform=basic_transform,
                split="train"
            )
            
            dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            # Run multiple epochs to detect memory leaks
            max_epochs = 5
            batches_per_epoch = 3
            
            for epoch in range(max_epochs):
                epoch_start_memory = process.memory_info().rss / (1024 * 1024)
                
                batch_count = 0
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= batches_per_epoch:
                        break
                    
                    # Process batch (simulate training step)
                    frames, coords, visibility, image_info = batch
                    _ = frames.mean()  # Simple operation
                    del frames, coords, visibility, image_info  # Explicit cleanup
                
                epoch_end_memory = process.memory_info().rss / (1024 * 1024)
                memory_increase_per_epoch = epoch_end_memory - epoch_start_memory
                
                # Memory increase per epoch should be minimal (<50MB)
                assert memory_increase_per_epoch < 50, \
                    f"Potential memory leak: epoch {epoch} increased memory by {memory_increase_per_epoch:.1f}MB"
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            total_increase = final_memory - initial_memory
            
            # Total memory increase should be reasonable (<100MB for 5 epochs)
            assert total_increase < 100, \
                f"Memory leak detected: total increase {total_increase:.1f}MB over {max_epochs} epochs"
    
    def test_concurrent_data_access(self, basic_transform):
        """Test concurrent data access patterns."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Create multiple datasets for different splits
            datasets = {}
            for split in ["train", "val"]:
                datasets[split] = SequenceCoordDataset(
                    annotation_file=str(annotation_file),
                    image_root=str(image_root),
                    T=3,
                    input_size=[64, 128],
                    transform=basic_transform,
                    split=split
                )
            
            # Create DataLoaders
            dataloaders = {}
            for split, dataset in datasets.items():
                dataloaders[split] = DataLoader(
                    dataset, 
                    batch_size=2, 
                    shuffle=(split == "train"),
                    num_workers=0
                )
            
            # Simulate concurrent access (alternating between train/val)
            train_iter = iter(dataloaders["train"])
            val_iter = iter(dataloaders["val"])
            
            for i in range(3):  # Limited iterations for integration test
                try:
                    # Get train batch
                    train_batch = next(train_iter)
                    train_frames, train_coords, _, _ = train_batch
                    assert train_frames.shape[0] == 2  # batch_size
                    
                    # Get val batch
                    val_batch = next(val_iter)
                    val_frames, val_coords, _, _ = val_batch
                    assert val_frames.shape[0] == 2  # batch_size
                    
                    # Should be able to access both without interference
                    assert train_frames.shape == val_frames.shape
                    assert train_coords.shape == val_coords.shape
                    
                except StopIteration:
                    break  # One of the iterators is exhausted