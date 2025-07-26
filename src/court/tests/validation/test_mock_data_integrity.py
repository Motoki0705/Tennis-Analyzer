"""Test mock data integrity and realistic properties.

This module validates that mock data used in tests has realistic properties
and maintains internal consistency for effective testing.
"""

import pytest
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile

from src.court.tests.fixtures.mock_data import (
    create_mock_coco_annotation,
    create_mock_images,
    create_mock_checkpoint,
    MockDataManager
)


class TestMockDataIntegrity:
    """Test suite for validating mock data integrity and realism."""
    
    def test_annotation_coordinate_distribution(self):
        """Test that court keypoints have a realistic, structured distribution."""
        annotation = create_mock_coco_annotation(num_images=20, num_clips=2)
        
        all_keypoints = []
        for ann in annotation["annotations"]:
            all_keypoints.append(np.array(ann["keypoints"]).reshape(-1, 3))
        
        all_keypoints = np.array(all_keypoints)
        
        # Test that keypoints are not all identical (i.e., there is some variation)
        assert np.std(all_keypoints[:, :, :2]) > 1.0, "Keypoints are too static"

        # Test the overall spread of the court across the image
        img_width = annotation["images"][0]["width"]
        img_height = annotation["images"][0]["height"]
        
        mean_court_x = np.mean(all_keypoints[:, :, 0])
        mean_court_y = np.mean(all_keypoints[:, :, 1])

        # The center of the court should be roughly in the center of the image
        assert img_width * 0.2 < mean_court_x < img_width * 0.8
        assert img_height * 0.3 < mean_court_y < img_height * 0.7

    def test_clip_temporal_continuity(self):
        """Test temporal continuity of court keypoints within clips."""
        annotation = create_mock_coco_annotation(num_images=30, num_clips=3)
        
        clips = {}
        for ann in annotation["annotations"]:
            img = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
            clip_id = img["clip_id"]
            if clip_id not in clips:
                clips[clip_id] = []
            clips[clip_id].append((img["frame_id"], np.array(ann["keypoints"]).reshape(-1, 3)))

        for clip_id, frames in clips.items():
            frames.sort(key=lambda x: x[0])
            if len(frames) < 2:
                continue

            # Calculate frame-to-frame displacement of keypoints
            displacements = []
            for i in range(1, len(frames)):
                prev_kps = frames[i-1][1][:, :2]
                curr_kps = frames[i][1][:, :2]
                # Calculate the mean distance between corresponding keypoints
                dist = np.mean(np.linalg.norm(curr_kps - prev_kps, axis=1))
                displacements.append(dist)
            
            # The average displacement should be small, indicating a stable court
            assert np.mean(displacements) < 20.0, f"Court movement too erratic in clip {clip_id}"

    def test_court_visibility_realism(self):
        """Test that court keypoints are always visible."""
        annotation = create_mock_coco_annotation(num_images=50, num_clips=5)
        
        visibility_values = []
        for ann in annotation["annotations"]:
            keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
            visibility_values.extend(keypoints[:, 2])
        
        # In mock data, all court keypoints should be visible (2)
        assert all(v == 2 for v in visibility_values), "Not all court keypoints are marked as visible"

    def test_bbox_keypoint_consistency(self):
        """Test that bounding boxes are consistent with keypoints."""
        annotation = create_mock_coco_annotation(num_images=20, num_clips=2)
        
        for ann in annotation["annotations"]:
            keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
            bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
            
            # Bbox should enclose all keypoints
            visible_kps = keypoints[keypoints[:, 2] > 0]
            if len(visible_kps) > 0:
                min_x, min_y = visible_kps[:, 0].min(), visible_kps[:, 1].min()
                max_x, max_y = visible_kps[:, 0].max(), visible_kps[:, 1].max()
                
                assert bbox_x <= min_x + 1e-4
                assert bbox_y <= min_y + 1e-4
                assert (bbox_x + bbox_w) >= max_x - 1e-4
                assert (bbox_y + bbox_h) >= max_y - 1e-4

    def test_image_pixel_value_distribution(self):
        """Test that generated images have realistic pixel distributions."""
        annotation = create_mock_coco_annotation(num_images=5, num_clips=1)
        
        with MockDataManager() as temp_dir:
            images_dir = temp_dir / "images"
            
            for img_info in annotation["images"]:
                img_path = images_dir / img_info["file_name"]
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Test individual image properties
                assert np.std(img_array) > 10, f"Image too uniform (std={np.std(img_array)})"
                assert len(np.unique(img_array)) > 5, "Image has too few unique pixel values"

    def test_court_lines_in_images(self):
        """Test that court lines are visible in the generated images."""
        annotation = create_mock_coco_annotation(num_images=5, num_clips=1)
        
        with MockDataManager() as temp_dir:
            images_dir = temp_dir / "images"
            
            for ann in annotation["annotations"]:
                img_info = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
                img_path = images_dir / img_info["file_name"]
                img = Image.open(img_path).convert('L') # Convert to grayscale
                img_array = np.array(img)

                # The white lines should be the brightest part of the image
                assert np.max(img_array) > 250, "Court lines are not drawn in white"
                # The background should be dark green
                assert np.mean(img_array) < 150, "Image background is too bright"
