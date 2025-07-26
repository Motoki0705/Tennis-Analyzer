"""Mock data generators for ball detection tests."""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import os
from PIL import Image


def create_mock_coco_annotation(
    num_images: int = 10,
    num_clips: int = 2,
    image_width: int = 640,
    image_height: int = 360
) -> Dict[str, Any]:
    """Create a mock COCO annotation for ball detection.
    
    Args:
        num_images: Number of images to create
        num_clips: Number of clips
        image_width: Width of images
        image_height: Height of images
        
    Returns:
        COCO-format annotation dictionary
    """
    # COCO base structure
    annotation = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "ball",
                "supercategory": "object"
            }
        ]
    }
    
    game_id = "test_game_001"
    images_per_clip = num_images // num_clips
    
    for clip_id in range(num_clips):
        for i in range(images_per_clip):
            image_id = clip_id * images_per_clip + i + 1
            
            # Image entry
            image_info = {
                "id": image_id,
                "file_name": f"test_frame_{image_id:06d}.jpg",
                "width": image_width,
                "height": image_height,
                "game_id": game_id,
                "clip_id": f"clip_{clip_id:03d}",
                "frame_id": i
            }
            annotation["images"].append(image_info)
            
            # Create annotation with valid ball coordinates
            # Simulate ball movement across frames
            base_x = image_width * 0.3 + (i * 20) % (image_width * 0.4)
            base_y = image_height * 0.5 + np.sin(i * 0.5) * 50
            
            # Ensure coordinates are within image bounds
            x = max(1, min(base_x, image_width - 1))
            y = max(1, min(base_y, image_height - 1))
            
            ann_info = {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": [float(x), float(y), 2.0],  # x, y, visibility (2=visible)
                "bbox": [x-5, y-5, 10, 10],  # x, y, width, height
                "area": 100
            }
            annotation["annotations"].append(ann_info)
    
    return annotation


def create_mock_images(annotation_data: Dict[str, Any], temp_dir: Path) -> None:
    """Create mock images based on annotation data.
    
    Args:
        annotation_data: COCO annotation data
        temp_dir: Directory to save images
    """
    images_dir = temp_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for img_info in annotation_data["images"]:
        width = img_info["width"]
        height = img_info["height"]
        
        # Create a simple test image (green background with noise)
        img_array = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        img_array[:, :, 1] = np.random.randint(100, 150, (height, width))  # Green channel
        
        # Add a white circle at ball position if annotation exists
        ann = next((a for a in annotation_data["annotations"] if a["image_id"] == img_info["id"]), None)
        if ann:
            x, y = int(ann["keypoints"][0]), int(ann["keypoints"][1])
            # Draw a simple white circle (5 pixel radius)
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    if dx*dx + dy*dy <= 25:  # Circle equation
                        px, py = x + dx, y + dy
                        if 0 <= px < width and 0 <= py < height:
                            img_array[py, px] = [255, 255, 255]  # White ball
        
        # Save image
        img = Image.fromarray(img_array)
        img_path = images_dir / img_info["file_name"]
        img.save(img_path)


def create_mock_checkpoint(model_class: str = "Conv3dTsmFpn") -> bytes:
    """Create a mock PyTorch checkpoint.
    
    Args:
        model_class: Name of the model class
        
    Returns:
        Serialized checkpoint data
    """
    # Create mock state dict with realistic tensor shapes
    if model_class == "Conv3dTsmFpn":
        state_dict = {
            "backbone.conv1.weight": torch.randn(64, 9, 7, 7),  # Conv3D layer
            "backbone.conv1.bias": torch.randn(64),
            "fpn.conv1.weight": torch.randn(256, 64, 3, 3),
            "fpn.conv1.bias": torch.randn(256),
            "head.final_conv.weight": torch.randn(1, 256, 1, 1),
            "head.final_conv.bias": torch.randn(1),
        }
    elif model_class == "LiteTrackNet": 
        state_dict = {
            "backbone.features.conv1.weight": torch.randn(64, 9, 3, 3),
            "backbone.features.conv1.bias": torch.randn(64),
            "head.conv.weight": torch.randn(1, 64, 1, 1),
            "head.conv.bias": torch.randn(1),
        }
    else:
        # Generic model
        state_dict = {
            "conv1.weight": torch.randn(64, 3, 3, 3),
            "conv1.bias": torch.randn(64),
            "final_layer.weight": torch.randn(1, 64, 1, 1),
            "final_layer.bias": torch.randn(1),
        }
    
    checkpoint = {
        "state_dict": state_dict,
        "epoch": 42,
        "global_step": 1000,
        "pytorch-lightning_version": "2.0.0",
        "lr_schedulers": [],
        "optimizer_states": [{}],
        "hyper_parameters": {
            "model_name": model_class,
            "learning_rate": 0.001,
            "input_size": [360, 640]
        }
    }
    
    # Serialize to bytes
    import io
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


class MockDataManager:
    """Manager for creating and cleaning up mock test data."""
    
    def __init__(self):
        self.temp_dir = None
        self.annotation_file = None
        self.images_dir = None
        self.checkpoint_file = None
    
    def setup(self) -> Path:
        """Setup temporary directory with mock data.
        
        Returns:
            Path to temporary directory
        """
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ball_test_"))
        
        # Create mock annotation
        annotation_data = create_mock_coco_annotation()
        self.annotation_file = self.temp_dir / "annotations.json"
        with open(self.annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        # Create mock images
        create_mock_images(annotation_data, self.temp_dir)
        self.images_dir = self.temp_dir / "images"
        
        # Create mock checkpoint
        checkpoint_data = create_mock_checkpoint()
        self.checkpoint_file = self.temp_dir / "model.ckpt"
        with open(self.checkpoint_file, 'wb') as f:
            f.write(checkpoint_data)
        
        return self.temp_dir
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def __enter__(self):
        return self.setup()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def get_mock_config_paths() -> Dict[str, str]:
    """Get paths to actual config files for testing.
    
    Returns:
        Dictionary of config file paths
    """
    # Check if running in Docker container or local environment
    if Path("/workspace").exists():
        base_path = Path("/workspace")
    else:
        base_path = Path("/mnt/c/Users/kamim/code/tennis_systems")
    
    return {
        "litmodule_config": str(base_path / "configs/train/ball/litmodule/default.yaml"),
        "model_config": str(base_path / "configs/train/ball/model/conv3d_tsm_fpn.yaml"),
        "main_config": str(base_path / "configs/train/ball/conv3d_tsm_fpn.yaml"),
        "datamodule_config": str(base_path / "configs/train/ball/litdatamodule/N_frames_stack_all.yaml"),
    }