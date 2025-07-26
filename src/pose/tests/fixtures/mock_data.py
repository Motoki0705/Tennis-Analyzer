"""Mock data generators for pose estimation tests."""

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
    num_players: int = 2,
) -> Dict[str, Any]:
    """Create a mock COCO annotation for pose estimation.
    
    Args:
        num_images: Number of images to create
        num_clips: Number of clips
        image_width: Width of images
        image_height: Height of images
        num_players: Number of players to annotate
        
    Returns:
        COCO-format annotation dictionary
    """
    # COCO base structure for pose estimation
    annotation = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ],
                "skeleton": [
                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                ]
            }
        ]
    }
    
    game_id = "test_game_pose_001"
    images_per_clip = num_images // num_clips
    annotation_id_counter = 1

    for clip_id in range(num_clips):
        for i in range(images_per_clip):
            image_id = clip_id * images_per_clip + i + 1
            
            image_info = {
                "id": image_id,
                "file_name": f"test_frame_pose_{image_id:06d}.jpg",
                "width": image_width,
                "height": image_height,
                "game_id": game_id,
                "clip_id": f"clip_{clip_id:03d}",
                "frame_id": i
            }
            annotation["images"].append(image_info)
            
            for player_id in range(num_players):
                # Simulate player movement
                base_x = (image_width * 0.25) + (player_id * image_width * 0.5) + (i * 5)
                base_y = image_height * 0.6
                
                # Generate a simple standing pose
                keypoints = []
                # Head
                keypoints.extend([base_x, base_y - 100, 2])  # nose
                keypoints.extend([base_x - 10, base_y - 110, 2]) # left_eye
                keypoints.extend([base_x + 10, base_y - 110, 2]) # right_eye
                # Shoulders
                keypoints.extend([base_x - 30, base_y - 80, 2]) # left_shoulder
                keypoints.extend([base_x + 30, base_y - 80, 2]) # right_shoulder
                # Arms
                keypoints.extend([base_x - 50, base_y - 40, 2]) # left_elbow
                keypoints.extend([base_x + 50, base_y - 40, 2]) # right_elbow
                keypoints.extend([base_x - 70, base_y, 2]) # left_wrist
                keypoints.extend([base_x + 70, base_y, 2]) # right_wrist
                # Hips
                keypoints.extend([base_x - 25, base_y, 2]) # left_hip
                keypoints.extend([base_x + 25, base_y, 2]) # right_hip
                # Legs
                keypoints.extend([base_x - 30, base_y + 50, 2]) # left_knee
                keypoints.extend([base_x + 30, base_y + 50, 2]) # right_knee
                keypoints.extend([base_x - 35, base_y + 100, 2]) # left_ankle
                keypoints.extend([base_x + 35, base_y + 100, 2]) # right_ankle
                
                # Flatten and add ears (often not visible)
                flat_keypoints = [0]* (5*3) # 5 head keypoints
                flat_keypoints[0:3] = keypoints[0:3]
                flat_keypoints[3:6] = keypoints[3:6]
                flat_keypoints[6:9] = keypoints[6:9]
                flat_keypoints.extend(keypoints[9:]) # Rest of the body
                
                # Bounding box from keypoints
                visible_kps = np.array(flat_keypoints).reshape(-1, 3)
                visible_kps = visible_kps[visible_kps[:, 2] > 0]
                if len(visible_kps) > 0:
                    x_min, y_min, _ = np.min(visible_kps, axis=0)
                    x_max, y_max, _ = np.max(visible_kps, axis=0)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    area = bbox[2] * bbox[3]
                else:
                    bbox = [0, 0, 0, 0]
                    area = 0

                ann_info = {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": flat_keypoints,
                    "bbox": bbox,
                    "area": area,
                    "num_keypoints": len(visible_kps),
                    "iscrowd": 0
                }
                annotation["annotations"].append(ann_info)
                annotation_id_counter += 1
    
    return annotation


def create_mock_images(annotation_data: Dict[str, Any], temp_dir: Path) -> None:
    """Create mock images with skeletons based on annotation data.
    
    Args:
        annotation_data: COCO annotation data
        temp_dir: Directory to save images
    """
    images_dir = temp_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    skeleton = annotation_data["categories"][0]["skeleton"]
    
    for img_info in annotation_data["images"]:
        width = img_info["width"]
        height = img_info["height"]
        
        img_array = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
        img_array[:, :, 1] = np.random.randint(100, 200, (height, width)) # Greenish court
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        anns = [a for a in annotation_data["annotations"] if a["image_id"] == img_info["id"]]
        for ann in anns:
            kps = np.array(ann["keypoints"]).reshape(-1, 3)
            
            # Draw skeleton
            for p1_idx, p2_idx in skeleton:
                if kps[p1_idx-1, 2] > 0 and kps[p2_idx-1, 2] > 0:
                    p1 = tuple(kps[p1_idx-1, :2])
                    p2 = tuple(kps[p2_idx-1, :2])
                    draw.line([p1, p2], fill=(255, 255, 255), width=2)
            
            # Draw keypoints
            for x, y, v in kps:
                if v > 0:
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0))

        img_path = images_dir / img_info["file_name"]
        img.save(img_path)


def create_mock_checkpoint(model_class: str = "PoseResNet") -> bytes:
    """Create a mock PyTorch checkpoint for a pose model.
    
    Args:
        model_class: Name of the model class
        
    Returns:
        Serialized checkpoint data
    """
    if model_class == "PoseResNet":
        state_dict = {
            "backbone.conv1.weight": torch.randn(64, 3, 7, 7),
            "backbone.layer1.0.conv1.weight": torch.randn(64, 64, 3, 3),
            "deconv_layers.0.weight": torch.randn(256, 256, 4, 4),
            "final_layer.weight": torch.randn(17, 256, 1, 1), # 17 keypoints
        }
    else: # Generic
        state_dict = {
            "encoder.block1.conv.weight": torch.randn(32, 3, 3, 3),
            "decoder.block1.conv.weight": torch.randn(32, 64, 3, 3),
            "head.weight": torch.randn(17, 32, 1, 1),
        }
    
    checkpoint = {
        "state_dict": state_dict,
        "epoch": 15,
        "global_step": 500,
        "pytorch-lightning_version": "2.0.0",
        "hyper_parameters": {
            "model_name": model_class,
            "learning_rate": 0.0005,
            "num_keypoints": 17
        }
    }
    
    import io
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


class MockDataManager:
    """Manager for creating and cleaning up mock test data for pose estimation."""
    
    def __init__(self):
        self.temp_dir = None
        self.annotation_file = None
        self.images_dir = None
        self.checkpoint_file = None
    
    def setup(self) -> Path:
        """Setup temporary directory with mock data."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pose_test_"))
        
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
    
    # NOTE: Training configs for pose do not exist yet.
    # We will use inference configs for schema validation.
    return {
        "infer_config": str(base_path / "configs/infer/pose/default.yaml"),
        "model_config": str(base_path / "configs/infer/pose/model/yolov8l-pose.yaml"),
    }
