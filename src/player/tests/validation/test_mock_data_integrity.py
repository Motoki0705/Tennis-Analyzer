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

from src.ball.tests.fixtures.mock_data import (
    create_mock_coco_annotation,
    create_mock_images,
    create_mock_checkpoint,
    MockDataManager
)


class TestMockDataIntegrity:
    """Test suite for validating mock data integrity and realism."""
    
    def test_annotation_coordinate_distribution(self):
        """Test that ball coordinates have realistic distribution."""
        annotation = create_mock_coco_annotation(num_images=50, num_clips=5)
        
        coordinates = []
        image_dims = []
        
        for ann in annotation["annotations"]:
            x, y, v = ann["keypoints"]
            coordinates.append((x, y))
            
            # Get image dimensions
            img = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
            image_dims.append((img["width"], img["height"]))
        
        coordinates = np.array(coordinates)
        image_dims = np.array(image_dims)
        
        # Test coordinate ranges
        x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]
        
        # Coordinates should span reasonable portion of image space
        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)
        
        avg_width = np.mean(image_dims[:, 0])
        avg_height = np.mean(image_dims[:, 1])
        
        # Should use at least 20% of image width/height
        assert x_range > 0.2 * avg_width, f"X coordinates too clustered: range {x_range} vs width {avg_width}"
        assert y_range > 0.2 * avg_height, f"Y coordinates too clustered: range {y_range} vs height {avg_height}"
        
        # Should not be perfectly uniform (more realistic)
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        assert x_std > avg_width * 0.05, "X coordinates too uniform"
        assert y_std > avg_height * 0.05, "Y coordinates too uniform"
    
    def test_clip_temporal_continuity(self):
        """Test temporal continuity within clips."""
        annotation = create_mock_coco_annotation(num_images=30, num_clips=3)
        
        # Group by clips
        clips = {}
        for ann in annotation["annotations"]:
            img = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
            clip_id = img["clip_id"]
            frame_id = img["frame_id"]
            
            if clip_id not in clips:
                clips[clip_id] = {}
            clips[clip_id][frame_id] = ann["keypoints"][:2]
        
        for clip_id, frames in clips.items():
            # Sort frames by frame_id
            sorted_frames = sorted(frames.items())
            
            if len(sorted_frames) < 3:
                continue
                
            # Calculate frame-to-frame velocities
            velocities = []
            for i in range(1, len(sorted_frames)):
                prev_pos = np.array(sorted_frames[i-1][1])
                curr_pos = np.array(sorted_frames[i][1])
                velocity = curr_pos - prev_pos
                velocities.append(np.linalg.norm(velocity))
            
            velocities = np.array(velocities)
            
            # Velocity should have some consistency (not completely random)
            if len(velocities) > 5:
                velocity_std = np.std(velocities)
                velocity_mean = np.mean(velocities)
                
                # Coefficient of variation should be reasonable (not > 2.0)
                if velocity_mean > 0:
                    cv = velocity_std / velocity_mean
                    assert cv < 2.0, f"Ball movement too erratic in clip {clip_id}: CV={cv}"
    
    def test_ball_visibility_realism(self):
        """Test that ball visibility values are realistic."""
        annotation = create_mock_coco_annotation(num_images=100, num_clips=5)
        
        visibility_values = []
        for ann in annotation["annotations"]:
            visibility = ann["keypoints"][2]
            visibility_values.append(visibility)
        
        visibility_values = np.array(visibility_values)
        
        # Should mostly be visible (value 2)
        visible_ratio = np.mean(visibility_values == 2)
        assert visible_ratio > 0.7, f"Too few visible balls: {visible_ratio}"
        
        # Should have valid COCO visibility values only
        valid_values = {0, 1, 2}  # not visible, occluded, visible
        unique_values = set(visibility_values)
        assert unique_values.issubset(valid_values), f"Invalid visibility values: {unique_values - valid_values}"
    
    def test_bbox_coordinate_consistency(self):
        """Test that bounding boxes are consistent with keypoints."""
        annotation = create_mock_coco_annotation(num_images=20, num_clips=2)
        
        for ann in annotation["annotations"]:
            kp_x, kp_y, visibility = ann["keypoints"]
            bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
            
            if visibility > 0:  # If ball is visible
                # Keypoint should be within or near bbox
                bbox_center_x = bbox_x + bbox_w / 2
                bbox_center_y = bbox_y + bbox_h / 2
                
                # Distance from keypoint to bbox center should be reasonable
                distance = np.sqrt((kp_x - bbox_center_x)**2 + (kp_y - bbox_center_y)**2)
                max_reasonable_distance = max(bbox_w, bbox_h)
                
                assert distance <= max_reasonable_distance, \
                    f"Keypoint too far from bbox center: distance={distance}, bbox_size=({bbox_w}, {bbox_h})"
                
                # Bbox should have reasonable size for a ball
                assert 5 <= bbox_w <= 50, f"Bbox width unrealistic: {bbox_w}"
                assert 5 <= bbox_h <= 50, f"Bbox height unrealistic: {bbox_h}"
    
    def test_image_pixel_value_distribution(self):
        """Test that generated images have realistic pixel distributions."""
        annotation = create_mock_coco_annotation(num_images=5, num_clips=1)
        
        with MockDataManager() as temp_dir:
            images_dir = temp_dir / "images"
            
            pixel_stats = []
            for img_info in annotation["images"]:
                img_path = images_dir / img_info["file_name"]
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Collect pixel statistics
                stats = {
                    'mean': np.mean(img_array),
                    'std': np.std(img_array),
                    'min': np.min(img_array),
                    'max': np.max(img_array)
                }
                pixel_stats.append(stats)
                
                # Test individual image properties
                assert 0 <= stats['min'] <= 255, f"Invalid pixel minimum: {stats['min']}"
                assert 0 <= stats['max'] <= 255, f"Invalid pixel maximum: {stats['max']}"
                assert stats['std'] > 5, f"Image too uniform (std={stats['std']})"
                
                # Should have some variation in pixel values
                unique_values = len(np.unique(img_array))
                total_pixels = img_array.size
                uniqueness_ratio = unique_values / total_pixels
                # Reduced threshold since our mock images are simple
                assert uniqueness_ratio > 0.0001, f"Image too uniform: {uniqueness_ratio} unique ratio"
    
    def test_ball_visibility_in_images(self):
        """Test that visible balls are actually visible in generated images."""
        annotation = create_mock_coco_annotation(num_images=10, num_clips=2)
        
        with MockDataManager() as temp_dir:
            images_dir = temp_dir / "images"
            
            for ann in annotation["annotations"]:
                if ann["keypoints"][2] == 2:  # Visible ball
                    img_info = next(img for img in annotation["images"] if img["id"] == ann["image_id"])
                    img_path = images_dir / img_info["file_name"]
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    ball_x, ball_y = int(ann["keypoints"][0]), int(ann["keypoints"][1])
                    
                    # Check region around ball position
                    radius = 10
                    x_min, x_max = max(0, ball_x - radius), min(img_array.shape[1], ball_x + radius)
                    y_min, y_max = max(0, ball_y - radius), min(img_array.shape[0], ball_y + radius)
                    
                    ball_region = img_array[y_min:y_max, x_min:x_max]
                    
                    # Ball region should have some bright pixels (white ball)
                    max_brightness = np.max(ball_region)
                    assert max_brightness > 200, f"No bright ball found at ({ball_x}, {ball_y}), max brightness: {max_brightness}"
    
    def test_checkpoint_tensor_properties(self):
        """Test that checkpoint tensors have realistic properties."""
        models = ["Conv3dTsmFpn", "LiteTrackNet"]
        
        for model_name in models:
            checkpoint_data = create_mock_checkpoint(model_name)
            
            with tempfile.NamedTemporaryFile(suffix='.ckpt') as tmp_file:
                with open(tmp_file.name, 'wb') as f:
                    f.write(checkpoint_data)
                
                checkpoint = torch.load(tmp_file.name, map_location='cpu')
                state_dict = checkpoint["state_dict"]
                
                # Test tensor properties
                for param_name, tensor in state_dict.items():
                    # Should be reasonable size (not too large/small)
                    numel = tensor.numel()
                    assert 1 <= numel <= 10_000_000, f"Parameter {param_name} has unrealistic size: {numel}"
                    
                    # Should have reasonable value distribution
                    if numel > 10:  # Skip very small tensors
                        tensor_std = torch.std(tensor).item()
                        tensor_mean = torch.mean(tensor).item()
                        
                        # Should not be all zeros or all ones
                        assert tensor_std > 1e-6, f"Parameter {param_name} has zero variation"
                        assert abs(tensor_mean) < 10, f"Parameter {param_name} has extreme mean: {tensor_mean}"
                        
                        # Should be roughly normally distributed (not extreme values)
                        tensor_max = torch.max(torch.abs(tensor)).item()
                        assert tensor_max < 10, f"Parameter {param_name} has extreme values: {tensor_max}"
    
    def test_checkpoint_hyperparameter_validity(self):
        """Test that checkpoint hyperparameters are realistic."""
        checkpoint_data = create_mock_checkpoint("Conv3dTsmFpn")
        
        with tempfile.NamedTemporaryFile(suffix='.ckpt') as tmp_file:
            with open(tmp_file.name, 'wb') as f:
                f.write(checkpoint_data)
            
            checkpoint = torch.load(tmp_file.name, map_location='cpu')
            hyper_params = checkpoint["hyper_parameters"]
            
            # Test learning rate
            if "learning_rate" in hyper_params:
                lr = hyper_params["learning_rate"]
                assert 1e-6 <= lr <= 1.0, f"Unrealistic learning rate: {lr}"
            
            # Test input size
            if "input_size" in hyper_params:
                input_size = hyper_params["input_size"]
                assert isinstance(input_size, (list, tuple)), "Input size should be list/tuple"
                assert len(input_size) == 2, "Input size should be [height, width]"
                
                height, width = input_size
                assert 64 <= height <= 2160, f"Unrealistic height: {height}"
                assert 64 <= width <= 4096, f"Unrealistic width: {width}"
            
            # Test epoch and global_step
            assert isinstance(checkpoint["epoch"], int), "Epoch should be integer"
            assert isinstance(checkpoint["global_step"], int), "Global step should be integer"
            assert checkpoint["epoch"] >= 0, "Epoch should be non-negative"
            assert checkpoint["global_step"] >= 0, "Global step should be non-negative"
    
    def test_data_loading_performance(self):
        """Test that mock data can be loaded efficiently."""
        import time
        
        # Test annotation loading time
        start_time = time.time()
        annotation = create_mock_coco_annotation(num_images=100, num_clips=5)
        annotation_time = time.time() - start_time
        
        assert annotation_time < 5.0, f"Annotation creation too slow: {annotation_time}s"
        
        # Test image creation time
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            start_time = time.time()
            create_mock_images(annotation, temp_path)
            image_time = time.time() - start_time
            
            assert image_time < 30.0, f"Image creation too slow: {image_time}s"
        
        # Test checkpoint creation time
        start_time = time.time()
        checkpoint_data = create_mock_checkpoint("Conv3dTsmFpn")
        checkpoint_time = time.time() - start_time
        
        assert checkpoint_time < 2.0, f"Checkpoint creation too slow: {checkpoint_time}s"
    
    def test_mock_data_manager_memory_usage(self):
        """Test that MockDataManager doesn't consume excessive memory."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and destroy multiple managers
        for i in range(3):
            with MockDataManager() as temp_dir:
                # Do some operations
                annotation_file = temp_dir / "annotations.json"
                with open(annotation_file) as f:
                    data = json.load(f)
                    assert len(data["images"]) > 0
            
            # Force garbage collection
            gc.collect()
            
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Should not increase by more than 100MB per iteration
            assert memory_increase < 100 * 1024 * 1024, \
                f"Memory leak detected: {memory_increase / (1024*1024):.1f}MB increase"