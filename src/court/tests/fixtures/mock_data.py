"""Mock data generators for court detection tests."""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import os
from PIL import Image, ImageDraw


def create_mock_coco_annotation(
    num_images: int = 10,
    num_clips: int = 2,
    image_width: int = 1280,
    image_height: int = 720,
    num_keypoints: int = 14
) -> Dict[str, Any]:
    """Create a mock COCO annotation for court detection.
    
    Args:
        num_images: Number of images to create
        num_clips: Number of clips
        image_width: Width of images
        image_height: Height of images
        num_keypoints: Number of court keypoints
        
    Returns:
        COCO-format annotation dictionary
    """
    annotation = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "court",
                "supercategory": "object"
            }
        ]
    }
    
    game_id = "test_game_001"
    images_per_clip = num_images // num_clips
    
    for clip_id in range(num_clips):
        # Define a simple court layout for this clip
        # This simulates a static camera perspective for the clip's duration
        p_top_left = (image_width * (0.2 + clip_id * 0.1), image_height * 0.3)
        p_top_right = (image_width * (0.8 - clip_id * 0.1), image_height * 0.3)
        p_bottom_left = (image_width * 0.1, image_height * 0.9)
        p_bottom_right = (image_width * 0.9, image_height * 0.9)

        for i in range(images_per_clip):
            image_id = clip_id * images_per_clip + i + 1
            
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
            
            # Simulate slight camera jitter/movement per frame
            jitter_x = np.sin(i * 0.2) * 5
            jitter_y = np.cos(i * 0.2) * 5

            # Generate 14 keypoints based on the court layout
            keypoints = []
            # Simple interpolation to generate 14 points
            for j in range(num_keypoints):
                # Interpolate between corners
                alpha = j / (num_keypoints - 1)
                if j % 2 == 0:
                    # Mix between top-left and bottom-left
                    x = p_top_left[0] * (1 - alpha) + p_bottom_left[0] * alpha
                    y = p_top_left[1] * (1 - alpha) + p_bottom_left[1] * alpha
                else:
                    # Mix between top-right and bottom-right
                    x = p_top_right[0] * (1 - alpha) + p_bottom_right[0] * alpha
                    y = p_top_right[1] * (1 - alpha) + p_bottom_right[1] * alpha

                # Add jitter and ensure bounds
                kpt_x = max(0, min(x + jitter_x, image_width - 1))
                kpt_y = max(0, min(y + jitter_y, image_height - 1))
                
                keypoints.extend([float(kpt_x), float(kpt_y), 2.0]) # All visible

            # Create a bounding box around all keypoints
            kps_array = np.array(keypoints).reshape(-1, 3)
            min_x, min_y = kps_array[:, 0].min(), kps_array[:, 1].min()
            max_x, max_y = kps_array[:, 0].max(), kps_array[:, 1].max()
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            
            ann_info = {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
                "bbox": bbox,
                "area": bbox[2] * bbox[3]
            }
            annotation["annotations"].append(ann_info)
    
    return annotation


def create_mock_images(annotation_data: Dict[str, Any], temp_dir: Path) -> None:
    """Create mock images with court lines based on annotation data.
    
    Args:
        annotation_data: COCO annotation data
        temp_dir: Directory to save images
    """
    images_dir = temp_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for img_info in annotation_data["images"]:
        width = img_info["width"]
        height = img_info["height"]
        
        # Create a simple test image (green background)
        img_array = np.full((height, width, 3), (50, 150, 50), dtype=np.uint8)
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        
        # Draw court lines if annotation exists
        ann = next((a for a in annotation_data["annotations"] if a["image_id"] == img_info["id"]), None)
        if ann:
            keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
            # Draw lines between some keypoints to simulate a court
            # This is a simplified representation
            for i in range(0, len(keypoints) - 2, 2):
                p1 = (keypoints[i][0], keypoints[i][1])
                p2 = (keypoints[i+1][0], keypoints[i+1][1])
                p3 = (keypoints[i+2][0], keypoints[i+2][1])
                draw.line([p1, p2], fill="white", width=3)
                draw.line([p1, p3], fill="white", width=2)

        # Save image
        img_path = images_dir / img_info["file_name"]
        img.save(img_path)


def create_mock_checkpoint(model_class: str = "FPN") -> bytes:
    """Create a mock PyTorch checkpoint for court models.
    
    Args:
        model_class: Name of the model class (e.g., "FPN", "LiteTrackNet")
        
    Returns:
        Serialized checkpoint data
    """
    if model_class == "FPN":
        state_dict = {
            "backbone.conv1.weight": torch.randn(64, 3, 7, 7),
            "fpn.layer4.conv.weight": torch.randn(256, 2048, 1, 1),
            "head.final_conv.weight": torch.randn(14, 256, 3, 3), # 14 keypoints
        }
    elif model_class == "LiteTrackNet": 
        state_dict = {
            "enc1.0.dw.weight": torch.randn(32, 1, 3, 3),
            "bottleneck.0.pw.weight": torch.randn(512, 256, 1, 1),
            "head.conv.weight": torch.randn(14, 32, 1, 1), # 14 keypoints
        }
    else: # Generic
        state_dict = {
            "encoder.block1.weight": torch.randn(64, 3, 3, 3),
            "decoder.block4.weight": torch.randn(64, 128, 3, 3),
            "head.output.weight": torch.randn(14, 64, 1, 1), # 14 keypoints
        }
    
    checkpoint = {
        "state_dict": state_dict,
        "epoch": 15,
        "global_step": 800,
        "pytorch-lightning_version": "2.1.0",
        "lr_schedulers": [],
        "optimizer_states": [{}],
        "hyper_parameters": {
            "model_name": model_class,
            "learning_rate": 0.0005,
            "input_size": [720, 1280],
            "num_keypoints": 14
        }
    }
    
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
        """Setup temporary directory with mock data."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="court_test_"))
        
        annotation_data = create_mock_coco_annotation()
        self.annotation_file = self.temp_dir / "annotations.json"
        with open(self.annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        create_mock_images(annotation_data, self.temp_dir)
        self.images_dir = self.temp_dir / "images"
        
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
    if Path("/workspace").exists():
        base_path = Path("/workspace")
    else:
        base_path = Path("/mnt/c/Users/kamim/code/tennis_systems")
    
    return {
        "litmodule_config": str(base_path / "configs/train/court/litmodule/default.yaml"),
        "model_config": str(base_path / "configs/train/court/model/fpn.yaml"),
        "main_config": str(base_path / "configs/train/court/fpn_generic.yaml"),
        "datamodule_config": str(base_path / "configs/train/court/litdatamodule/court_data.yaml"),
    }
