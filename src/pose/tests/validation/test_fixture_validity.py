"""Test fixture validity for pose estimation tests."""

import pytest
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile

from src.pose.tests.fixtures.mock_data import (
    create_mock_coco_annotation,
    create_mock_images,
    create_mock_checkpoint,
    MockDataManager
)


class TestFixtureValidity:
    """Test suite for validating test fixtures and mock data for pose estimation."""
    
    def test_mock_coco_annotation_structure(self):
        """Test that mock COCO annotation for pose has correct structure."""
        annotation = create_mock_coco_annotation(num_images=4, num_clips=1, num_players=2)
        
        required_keys = {"images", "annotations", "categories"}
        assert all(key in annotation for key in required_keys)
        
        # Validate category structure for pose
        assert len(annotation["categories"]) == 1
        category = annotation["categories"][0]
        assert category["name"] == "person"
        assert "keypoints" in category and len(category["keypoints"]) == 17
        assert "skeleton" in category and len(category["skeleton"]) > 0
        
        # Validate image and annotation structures
        assert len(annotation["images"]) == 4
        assert len(annotation["annotations"]) == 4 * 2 # 4 images, 2 players
        
        for ann in annotation["annotations"]:
            required_ann_keys = {"id", "image_id", "category_id", "keypoints", "bbox", "area", "num_keypoints"}
            assert all(key in ann for key in required_ann_keys)
            assert len(ann["keypoints"]) == 17 * 3 # 17 keypoints, (x,y,v)

    def test_mock_coco_annotation_coordinate_validity(self):
        """Test that keypoints and bboxes are within image bounds."""
        width, height = 1280, 720
        annotation = create_mock_coco_annotation(image_width=width, image_height=height)
        
        for ann in annotation["annotations"]:
            kps = np.array(ann["keypoints"]).reshape(-1, 3)
            img = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
            
            for x, y, v in kps:
                if v > 0:
                    assert 0 <= x < img["width"], f"X keypoint {x} out of bounds"
                    assert 0 <= y < img["height"], f"Y keypoint {y} out of bounds"
            
            bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
            if bbox_w > 0 and bbox_h > 0:
                assert bbox_x + bbox_w <= img["width"]
                assert bbox_y + bbox_h <= img["height"]

    def test_mock_images_creation(self):
        """Test that mock images with skeletons are created correctly."""
        annotation = create_mock_coco_annotation(num_images=2, num_clips=1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            create_mock_images(annotation, temp_path)
            
            images_dir = temp_path / "images"
            assert images_dir.exists()
            
            for img_info in annotation["images"]:
                img_path = images_dir / img_info["file_name"]
                assert img_path.exists()
                img = Image.open(img_path)
                assert img.size == (img_info["width"], img_info["height"])
                assert img.mode == "RGB"

    def test_mock_checkpoint_structure(self):
        """Test that mock pose checkpoint has correct structure."""
        checkpoint_data = create_mock_checkpoint("PoseResNet")
        
        with tempfile.NamedTemporaryFile(suffix='.ckpt') as tmp_file:
            with open(tmp_file.name, 'wb') as f:
                f.write(checkpoint_data)
            checkpoint = torch.load(tmp_file.name, map_location='cpu')
            
            required_keys = {"state_dict", "epoch", "global_step", "hyper_parameters"}
            assert all(key in checkpoint for key in required_keys)
            assert "num_keypoints" in checkpoint["hyper_parameters"]
            assert checkpoint["hyper_parameters"]["num_keypoints"] == 17

    def test_mock_data_manager_lifecycle(self):
        """Test MockDataManager setup and cleanup for pose data."""
        manager = MockDataManager()
        try:
            temp_dir = manager.setup()
            assert temp_dir.exists()
            assert manager.annotation_file.exists()
            assert manager.images_dir.exists()
            assert manager.checkpoint_file.exists()
        finally:
            manager.cleanup()
            assert not temp_dir.exists()
