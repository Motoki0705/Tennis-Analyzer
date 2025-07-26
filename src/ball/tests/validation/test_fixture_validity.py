"""Test fixture validity for ball detection tests.

This module validates that all mock data, fixtures, and test resources
used in the test suite are properly formatted and contain valid data.
"""

import pytest
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from src.ball.tests.fixtures.mock_data import (
    create_mock_coco_annotation,
    create_mock_images,
    create_mock_checkpoint,
    MockDataManager
)


class TestFixtureValidity:
    """Test suite for validating test fixtures and mock data."""
    
    def test_mock_coco_annotation_structure(self):
        """Test that mock COCO annotation has correct structure."""
        annotation = create_mock_coco_annotation(num_images=6, num_clips=2)
        
        # Check required top-level keys
        required_keys = {"images", "annotations", "categories"}
        assert all(key in annotation for key in required_keys), \
            f"Missing required keys: {required_keys - set(annotation.keys())}"
        
        # Check categories structure
        assert len(annotation["categories"]) == 1
        category = annotation["categories"][0]
        assert category["id"] == 1
        assert category["name"] == "ball"
        assert "supercategory" in category
        
        # Check images structure
        assert len(annotation["images"]) == 6
        for img in annotation["images"]:
            required_img_keys = {"id", "file_name", "width", "height", "game_id", "clip_id", "frame_id"}
            assert all(key in img for key in required_img_keys), \
                f"Image missing required keys: {required_img_keys - set(img.keys())}"
            
            # Validate data types and ranges
            assert isinstance(img["id"], int) and img["id"] > 0
            assert isinstance(img["width"], int) and img["width"] > 0
            assert isinstance(img["height"], int) and img["height"] > 0
            assert isinstance(img["file_name"], str) and img["file_name"].endswith('.jpg')
        
        # Check annotations structure
        assert len(annotation["annotations"]) == 6
        for ann in annotation["annotations"]:
            required_ann_keys = {"id", "image_id", "category_id", "keypoints", "bbox", "area"}
            assert all(key in ann for key in required_ann_keys), \
                f"Annotation missing required keys: {required_ann_keys - set(ann.keys())}"
            
            # Validate keypoints format
            assert len(ann["keypoints"]) == 3  # x, y, visibility
            x, y, v = ann["keypoints"]
            assert isinstance(x, (int, float)) and x >= 0
            assert isinstance(y, (int, float)) and y >= 0
            assert v in [0, 1, 2]  # COCO visibility values
    
    def test_mock_coco_annotation_coordinate_validity(self):
        """Test that coordinates in mock annotation are within image bounds."""
        width, height = 640, 360
        annotation = create_mock_coco_annotation(
            num_images=10, num_clips=2, 
            image_width=width, image_height=height
        )
        
        for ann in annotation["annotations"]:
            x, y, v = ann["keypoints"]
            
            # Find corresponding image
            img = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
            
            # Check coordinates are within bounds
            assert 0 <= x < img["width"], \
                f"X coordinate {x} out of bounds [0, {img['width']})"
            assert 0 <= y < img["height"], \
                f"Y coordinate {y} out of bounds [0, {img['height']})"
            
            # Check bbox validity
            bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
            assert bbox_x >= 0 and bbox_x + bbox_w <= img["width"]
            assert bbox_y >= 0 and bbox_y + bbox_h <= img["height"]
    
    def test_mock_coco_annotation_temporal_consistency(self):
        """Test temporal consistency of ball positions across frames."""
        annotation = create_mock_coco_annotation(num_images=20, num_clips=2)
        
        # Group annotations by clip
        clips = {}
        for ann in annotation["annotations"]:
            img = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
            clip_id = img["clip_id"]
            if clip_id not in clips:
                clips[clip_id] = []
            clips[clip_id].append((img["frame_id"], ann["keypoints"][:2]))
        
        for clip_id, frames in clips.items():
            # Sort by frame_id
            frames.sort(key=lambda x: x[0])
            
            # Check that ball doesn't teleport (reasonable movement between frames)
            for i in range(1, len(frames)):
                prev_x, prev_y = frames[i-1][1]
                curr_x, curr_y = frames[i][1]
                
                # Maximum reasonable movement per frame (adjust based on video fps)
                max_movement = 100  # pixels
                distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                
                assert distance <= max_movement, \
                    f"Ball moved too far between frames: {distance} pixels in clip {clip_id}"
    
    def test_mock_images_creation(self):
        """Test that mock images are created correctly."""
        annotation = create_mock_coco_annotation(num_images=4, num_clips=1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            create_mock_images(annotation, temp_path)
            
            images_dir = temp_path / "images"
            assert images_dir.exists(), "Images directory not created"
            
            # Check that all images were created
            for img_info in annotation["images"]:
                img_path = images_dir / img_info["file_name"]
                assert img_path.exists(), f"Image {img_info['file_name']} not created"
                
                # Load and validate image
                img = Image.open(img_path)
                assert img.size == (img_info["width"], img_info["height"])
                assert img.mode == "RGB"
                
                # Check that image data is not all zeros
                img_array = np.array(img)
                assert not np.all(img_array == 0), "Image appears to be all black"
                assert img_array.shape == (img_info["height"], img_info["width"], 3)
    
    def test_mock_checkpoint_structure(self):
        """Test that mock checkpoint has correct PyTorch Lightning structure."""
        checkpoint_data = create_mock_checkpoint("Conv3dTsmFpn")
        
        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(suffix='.ckpt') as tmp_file:
            with open(tmp_file.name, 'wb') as f:
                f.write(checkpoint_data)
            
            checkpoint = torch.load(tmp_file.name, map_location='cpu')
            
            # Check required Lightning checkpoint keys
            required_keys = {
                "state_dict", "epoch", "global_step", 
                "pytorch-lightning_version", "hyper_parameters"
            }
            assert all(key in checkpoint for key in required_keys), \
                f"Checkpoint missing required keys: {required_keys - set(checkpoint.keys())}"
            
            # Check state_dict structure
            state_dict = checkpoint["state_dict"]
            assert isinstance(state_dict, dict)
            assert len(state_dict) > 0
            
            # Check that all state_dict values are tensors
            for key, value in state_dict.items():
                assert torch.is_tensor(value), f"State dict value {key} is not a tensor"
                assert value.dtype in [torch.float32, torch.float64], \
                    f"Unexpected tensor dtype for {key}: {value.dtype}"
            
            # Check hyperparameters
            hyper_params = checkpoint["hyper_parameters"]
            assert isinstance(hyper_params, dict)
            assert "model_name" in hyper_params
    
    def test_mock_checkpoint_different_models(self):
        """Test that different model checkpoints have different structures."""
        models = ["Conv3dTsmFpn", "LiteTrackNet", "GenericModel"]
        checkpoints = {}
        
        for model_name in models:
            checkpoint_data = create_mock_checkpoint(model_name)
            
            with tempfile.NamedTemporaryFile(suffix='.ckpt') as tmp_file:
                with open(tmp_file.name, 'wb') as f:
                    f.write(checkpoint_data)
                
                checkpoint = torch.load(tmp_file.name, map_location='cpu')
                checkpoints[model_name] = checkpoint["state_dict"]
        
        # Check that different models have different state dict keys
        conv3d_keys = set(checkpoints["Conv3dTsmFpn"].keys())
        lite_keys = set(checkpoints["LiteTrackNet"].keys())
        
        # They should have some different keys
        assert conv3d_keys != lite_keys, "Different models should have different state dict structures"
    
    def test_mock_data_manager_lifecycle(self):
        """Test MockDataManager setup and cleanup."""
        manager = MockDataManager()
        
        # Test setup
        temp_dir = manager.setup()
        
        try:
            # Check that all files were created
            assert temp_dir.exists()
            assert manager.annotation_file.exists()
            assert manager.images_dir.exists() 
            assert manager.checkpoint_file.exists()
            
            # Check annotation file is valid JSON
            with open(manager.annotation_file) as f:
                annotation = json.load(f)
            assert "images" in annotation
            assert "annotations" in annotation
            
            # Check checkpoint file is valid
            checkpoint = torch.load(manager.checkpoint_file, map_location='cpu')
            assert "state_dict" in checkpoint
            
        finally:
            # Test cleanup
            manager.cleanup()
            assert not temp_dir.exists()
    
    def test_mock_data_manager_context_manager(self):
        """Test MockDataManager as context manager."""
        temp_dir_path = None
        
        with MockDataManager() as temp_dir:
            temp_dir_path = temp_dir
            assert temp_dir.exists()
            
            # Files should exist within context
            annotation_file = temp_dir / "annotations.json"
            images_dir = temp_dir / "images"
            checkpoint_file = temp_dir / "model.ckpt"
            
            assert annotation_file.exists()
            assert images_dir.exists()
            assert checkpoint_file.exists()
        
        # Files should be cleaned up after context
        assert not temp_dir_path.exists()
    
    def test_fixture_data_sizes(self):
        """Test that fixture data has reasonable sizes."""
        # Test annotation size
        annotation = create_mock_coco_annotation(num_images=100, num_clips=5)
        annotation_json = json.dumps(annotation)
        
        # Should be reasonable size (not too large for tests)
        assert len(annotation_json) < 1_000_000, "Annotation JSON too large for test fixture"
        
        # Test checkpoint size
        checkpoint_data = create_mock_checkpoint("Conv3dTsmFpn")
        assert len(checkpoint_data) < 10_000_000, "Checkpoint too large for test fixture"  # <10MB
        
        # Test image creation doesn't consume too much memory
        with MockDataManager() as temp_dir:
            images_dir = temp_dir / "images"
            total_size = sum(f.stat().st_size for f in images_dir.glob("*.jpg"))
            assert total_size < 50_000_000, "Total image size too large for test fixture"  # <50MB