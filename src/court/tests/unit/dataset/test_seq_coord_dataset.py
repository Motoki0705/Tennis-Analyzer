"""Unit tests for SequenceCoordDataset."""

import pytest
import torch
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from PIL import Image
import albumentations as A

from src.ball.dataset.seq_coord_dataset import SequenceCoordDataset
from src.ball.tests.fixtures.mock_data import MockDataManager, create_mock_coco_annotation, create_mock_images


@pytest.mark.unit
class TestSequenceCoordDataset:
    """Test SequenceCoordDataset functionality."""
    
    @pytest.fixture
    def mock_transform(self):
        """Create mock Albumentations transform."""
        return A.ReplayCompose([
            A.Resize(height=360, width=640),
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))
    
    @pytest.fixture
    def sample_annotation(self):
        """Create sample annotation for testing."""
        return create_mock_coco_annotation(num_images=12, num_clips=2, image_width=640, image_height=360)
    
    def test_dataset_init_basic(self, mock_transform):
        """Test basic dataset initialization."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform,
                split="train"
            )
            
            # Check basic attributes
            assert dataset.T == 3
            assert dataset.input_size == (360, 640)
            assert dataset.split == "train"
            assert dataset.input_type == "stack"  # default
            assert dataset.output_type == "all"   # default
    
    def test_dataset_init_parameters(self, mock_transform):
        """Test dataset initialization with different parameters."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=5,
                input_size=[320, 640],
                transform=mock_transform,
                split="val",
                input_type="cat",
                output_type="last",
                skip_frames_range=(1, 3)
            )
            
            assert dataset.T == 5
            assert dataset.input_size == (320, 640)
            assert dataset.split == "val"
            assert dataset.input_type == "cat"
            assert dataset.output_type == "last"
            assert dataset.skip_min == 1
            assert dataset.skip_max == 3
    
    def test_dataset_annotation_loading(self, mock_transform):
        """Test annotation file loading and parsing."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
            
            # Check that annotations were loaded
            assert len(dataset.images) > 0
            assert len(dataset.anns_by_image) > 0
            
            # Check annotation structure
            for img_id, img_info in dataset.images.items():
                assert "id" in img_info
                assert "file_name" in img_info
                assert "width" in img_info
                assert "height" in img_info
                assert "game_id" in img_info
                assert "clip_id" in img_info
    
    def test_dataset_clip_grouping(self, mock_transform):
        """Test clip-based grouping of images."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
            
            # Check clip grouping
            assert hasattr(dataset, 'clip_groups')
            assert len(dataset.clip_groups) > 0
            
            # Each clip should have multiple images
            for clip_key, image_ids in dataset.clip_groups.items():
                assert len(image_ids) >= dataset.T  # At least T frames per clip
    
    def test_dataset_window_generation(self, mock_transform):
        """Test sliding window generation."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
            
            # Check windows were created
            assert len(dataset.windows) > 0
            
            # Each window should be (clip_key, start_index)
            for clip_key, start_idx in dataset.windows:
                assert clip_key in dataset.clip_groups
                assert isinstance(start_idx, int)
                assert start_idx >= 0
    
    def test_dataset_len(self, mock_transform):
        """Test dataset length."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root), 
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
            
            assert len(dataset) == len(dataset.windows)
            assert len(dataset) > 0
    
    def test_dataset_getitem_stack_all(self, mock_transform):
        """Test __getitem__ with stack input and all output."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform,
                input_type="stack",
                output_type="all"
            )
            
            # Get first item
            frames, coords, visibility, image_info = dataset[0]
            
            # Check frames shape: [T, C, H, W]
            assert frames.shape == (3, 3, 360, 640)
            assert frames.dtype == torch.float32
            
            # Check coords shape: [T, 2] (x, y normalized)
            assert coords.shape == (3, 2)
            assert coords.dtype == torch.float32
            
            # Check visibility shape: [T]
            assert visibility.shape == (3,)
            assert visibility.dtype == torch.float32
            
            # Check image_info
            assert isinstance(image_info, dict)
            assert "id" in image_info
    
    def test_dataset_getitem_cat_last(self, mock_transform):
        """Test __getitem__ with cat input and last output."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=4,
                input_size=[320, 640],
                transform=mock_transform,
                input_type="cat",
                output_type="last"
            )
            
            frames, coords, visibility, image_info = dataset[0]
            
            # Check frames shape: [C*T, H, W]
            assert frames.shape == (12, 320, 640)  # 3*4 channels
            
            # Check coords shape: [2] (single frame)
            assert coords.shape == (2,)
            
            # Check visibility shape: [1]
            assert visibility.shape == (1,)
    
    def test_dataset_coordinate_normalization(self, mock_transform):
        """Test coordinate normalization to [0,1] range."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform,
                output_type="all"
            )
            
            frames, coords, visibility, image_info = dataset[0]
            
            # Coordinates should be normalized to [0, 1]
            assert torch.all(coords >= 0)
            assert torch.all(coords <= 1)
            
            # Check that coordinates are not all zeros (unless no ball)
            visible_mask = visibility > 0
            if visible_mask.any():
                visible_coords = coords[visible_mask]
                assert not torch.allclose(visible_coords, torch.zeros_like(visible_coords))
    
    def test_dataset_skip_frames_training(self, mock_transform):
        """Test frame skipping during training."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Training dataset with frame skipping
            dataset_train = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform,
                split="train",
                skip_frames_range=(1, 3)
            )
            
            # Get multiple samples to test randomness
            frames1, _, _, _ = dataset_train[0]
            frames2, _, _, _ = dataset_train[0]
            
            # With frame skipping, frames might be different due to randomness
            # (This is stochastic so might occasionally be same)
            assert frames1.shape == frames2.shape == (3, 3, 360, 640)
    
    def test_dataset_no_skip_validation(self, mock_transform):
        """Test no frame skipping during validation."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Validation dataset (no frame skipping)
            dataset_val = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform,
                split="val",
                skip_frames_range=(1, 3)
            )
            
            # Get same sample multiple times
            frames1, _, _, _ = dataset_val[0]
            frames2, _, _, _ = dataset_val[0]
            
            # Should be identical for validation
            assert torch.allclose(frames1, frames2)
    
    def test_dataset_clip_keypoints_handling(self, mock_transform):
        """Test clipping of out-of-bounds keypoints."""
        # Test the static method
        clipped = SequenceCoordDataset._clip_keypoints((650, 370), (360, 640))
        
        # Should clip to image bounds
        assert clipped[0] <= 640 - 1e-3  # width bound
        assert clipped[1] <= 360 - 1e-3  # height bound
        assert clipped[0] >= 0
        assert clipped[1] >= 0
        
        # Test with negative coordinates
        clipped_neg = SequenceCoordDataset._clip_keypoints((-10, -5), (360, 640))
        assert clipped_neg[0] == 0
        assert clipped_neg[1] == 0
    
    def test_dataset_group_split_clip(self):
        """Test clip-based data splitting."""
        clip_keys = [('game1', 'clip1'), ('game1', 'clip2'), ('game2', 'clip1'), ('game2', 'clip2')]
        
        train_keys, val_keys, test_keys = SequenceCoordDataset.group_split_clip(
            clip_keys, train_ratio=0.5, val_ratio=0.25, seed=42
        )
        
        # Check that all clips are assigned
        total_clips = len(train_keys) + len(val_keys) + len(test_keys)
        assert total_clips == len(clip_keys)
        
        # Check no overlap
        all_assigned = set(train_keys) | set(val_keys) | set(test_keys)
        assert len(all_assigned) == len(clip_keys)
        
        # Check approximate ratios (might not be exact due to small sample)
        assert len(train_keys) >= 1  # At least some training data
    
    def test_dataset_transform_replay(self, mock_transform):
        """Test that transforms are replayed consistently across frames."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
            
            frames, coords, visibility, image_info = dataset[0]
            
            # All frames should have same transform applied (same output size)
            assert frames.shape == (3, 3, 360, 640)
            
            # Frames should be valid tensors
            assert torch.isfinite(frames).all()
            assert not torch.allclose(frames, torch.zeros_like(frames))
    
    def test_dataset_missing_annotation(self, mock_transform):
        """Test handling of frames without ball annotations."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            # Load and modify annotation to remove some annotations
            with open(annotation_file) as f:
                annotation_data = json.load(f)
            
            # Remove half the annotations
            annotation_data["annotations"] = annotation_data["annotations"][:len(annotation_data["annotations"])//2]
            
            # Save modified annotation
            modified_annotation_file = temp_dir / "modified_annotations.json"
            with open(modified_annotation_file, 'w') as f:
                json.dump(annotation_data, f)
            
            dataset = SequenceCoordDataset(
                annotation_file=str(modified_annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
            
            frames, coords, visibility, image_info = dataset[0]
            
            # Should handle missing annotations gracefully
            assert frames.shape == (3, 3, 360, 640)
            assert coords.shape == (3, 2)
            assert visibility.shape == (3,)
            
            # Missing annotations should have 0 visibility and 0 coordinates
            invisible_mask = visibility == 0
            if invisible_mask.any():
                invisible_coords = coords[invisible_mask]
                assert torch.allclose(invisible_coords, torch.zeros_like(invisible_coords))
    
    def test_dataset_error_handling(self, mock_transform):
        """Test dataset error handling."""
        # Test with non-existent annotation file
        with pytest.raises(FileNotFoundError):
            SequenceCoordDataset(
                annotation_file="non_existent.json",
                image_root="/tmp",
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
        
        # Test with invalid input_type
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            with pytest.raises(AssertionError):
                SequenceCoordDataset(
                    annotation_file=str(annotation_file),
                    image_root=str(image_root),
                    T=3,
                    input_size=[360, 640],
                    transform=mock_transform,
                    input_type="invalid"
                )
    
    @pytest.mark.slow
    def test_dataset_performance(self, mock_transform):
        """Test dataset loading performance."""
        import time
        
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            dataset = SequenceCoordDataset(
                annotation_file=str(annotation_file),
                image_root=str(image_root),
                T=3,
                input_size=[360, 640],
                transform=mock_transform
            )
            
            # Time multiple sample loads
            start_time = time.time()
            for i in range(min(5, len(dataset))):
                _ = dataset[i]
            end_time = time.time()
            
            # Should be reasonably fast (< 2 seconds for 5 samples)
            assert (end_time - start_time) < 2.0
    
    def test_dataset_different_splits(self, mock_transform):
        """Test dataset with different data splits."""
        with MockDataManager() as temp_dir:
            annotation_file = temp_dir / "annotations.json"
            image_root = temp_dir / "images"
            
            splits = ["train", "val", "test"]
            datasets = {}
            
            for split in splits:
                datasets[split] = SequenceCoordDataset(
                    annotation_file=str(annotation_file),
                    image_root=str(image_root),
                    T=3,
                    input_size=[360, 640],
                    transform=mock_transform,
                    split=split,
                    train_ratio=0.6,
                    val_ratio=0.2  # test gets remaining 0.2
                )
            
            # Check that splits have data
            for split, dataset in datasets.items():
                assert len(dataset) > 0, f"Split {split} has no data"
            
            # Check that splits are different (in most cases)
            total_windows = sum(len(dataset) for dataset in datasets.values())
            # Total should be reasonable
            assert total_windows > 0