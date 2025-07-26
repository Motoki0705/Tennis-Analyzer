"""
Unified dataset visualization system for all model types.
Provides standardized interfaces for visualizing dataset samples across ball, court, event, and player tasks.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import hydra
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class DatasetVisualizer:
    """
    Unified dataset visualizer for all model types.
    Handles visualization of dataset samples including:
    - Ball detection (heatmaps and coordinates)
    - Court detection (keypoint heatmaps)
    - Event classification (sequence data)
    - Player detection (bounding boxes)
    """

    def __init__(self, config: DictConfig, module_name: str, output_dir: str = "visualizations"):
        """
        Initialize dataset visualizer.
        
        Args:
            config: Hydra configuration for the specific task
            module_name: Module name (ball, court, event, player)
            output_dir: Output directory for saving visualizations
        """
        self.config = config
        self.module_name = module_name
        self.output_dir = Path(output_dir) / module_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = self._get_class_names()
        
    def _get_class_names(self) -> Dict[int, str]:
        """Get class names for the specific task."""
        class_maps = {
            "ball": {0: "ball"},
            "court": {i: f"keypoint_{i}" for i in range(15)},
            "event": {0: "serve", 1: "return", 2: "rally", 3: "other"},
            "player": {0: "player", 1: "non_player_person"}
        }
        return class_maps.get(self.module_name, {})
    
    def _denormalize_image(self, image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Denormalize image from standard ImageNet normalization.
        
        Args:
            image: Image tensor or array in format [C, H, W] or [H, W, C]
            
        Returns:
            Denormalized image as numpy array in format [H, W, C]
        """
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu().numpy()
        else:
            img = image.copy()
            
        # Ensure CHW format
        if img.ndim == 3 and img.shape[-1] in [1, 3]:
            img = np.transpose(img, (2, 0, 1))
            
        # Denormalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean
        img = np.clip(img, 0.0, 1.0)
        
        # Convert to HWC format
        img = np.transpose(img, (1, 2, 0))
        
        return img
    
    def visualize_ball_sample(
        self, 
        frames: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray, Dict],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize ball detection dataset sample.
        
        Args:
            frames: Input frames [T*C, H, W] or [T, C, H, W]
            target: Target heatmap or coordinates
            sample_idx: Sample index for naming
            save_path: Optional path to save the visualization
        """
        # Handle different input formats
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
            
        if frames.ndim == 3:  # [T*C, H, W] -> extract last frame
            if frames.shape[0] % 3 == 0:
                T = frames.shape[0] // 3
                last_frame = frames[-3:, :, :]  # Last 3 channels (RGB)
            else:
                last_frame = frames[-3:, :, :] if frames.shape[0] >= 3 else frames
        else:  # [T, C, H, W]
            last_frame = frames[-1, :, :, :]
            
        # Denormalize image
        img = self._denormalize_image(last_frame)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Image with target overlay
        axes[1].imshow(img)
        
        if isinstance(target, dict) and "heatmap" in target:
            # Heatmap target
            heatmap = target["heatmap"]
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
            heatmap = np.squeeze(heatmap)
            
            # Normalize heatmap
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            axes[1].imshow(heatmap, alpha=0.5, cmap="jet")
            axes[1].set_title("Image + Heatmap Target")
            
        elif isinstance(target, dict) and "coords" in target:
            # Coordinate target
            coords = target["coords"]
            visibility = target.get("visibility", [1] * len(coords))
            
            if isinstance(coords, torch.Tensor):
                coords = coords.detach().cpu().numpy()
            if isinstance(visibility, torch.Tensor):
                visibility = visibility.detach().cpu().numpy()
                
            H, W = img.shape[:2]
            for i, (coord, vis) in enumerate(zip(coords, visibility)):
                if vis > 0:
                    x, y = coord[0] * W, coord[1] * H
                    circle = plt.Circle((x, y), 5, color="red", fill=False, linewidth=2)
                    axes[1].add_patch(circle)
                    
            axes[1].set_title("Image + Coordinate Target")
        else:
            # Direct heatmap/coordinate array
            if isinstance(target, torch.Tensor):
                target = target.detach().cpu().numpy()
            
            if target.ndim == 2:  # Heatmap
                if target.max() > target.min():
                    target = (target - target.min()) / (target.max() - target.min())
                axes[1].imshow(target, alpha=0.5, cmap="jet")
                axes[1].set_title("Image + Heatmap Target")
            elif target.ndim == 1 and len(target) == 2:  # Single coordinate
                H, W = img.shape[:2]
                x, y = target[0] * W, target[1] * H
                circle = plt.Circle((x, y), 5, color="red", fill=False, linewidth=2)
                axes[1].add_patch(circle)
                axes[1].set_title("Image + Coordinate Target")
        
        axes[1].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            log.info(f"Ball sample visualization saved: {save_path}")
        else:
            default_path = self.output_dir / f"ball_sample_{sample_idx:04d}.png"
            plt.savefig(default_path, bbox_inches="tight", dpi=150)
            log.info(f"Ball sample visualization saved: {default_path}")
            
        plt.close()
    
    def visualize_court_sample(
        self,
        image: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize court detection dataset sample.
        
        Args:
            image: Input image [C, H, W]
            target: Target keypoint heatmaps [K, H, W] where K is number of keypoints
            sample_idx: Sample index for naming
            save_path: Optional path to save the visualization
        """
        # Denormalize image
        img = self._denormalize_image(image)
        
        # Process target heatmaps
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        if target.ndim == 3:  # Multiple keypoint heatmaps
            combined_heatmap = np.max(target, axis=0)  # Take max across keypoints
        else:
            combined_heatmap = target
            
        # Normalize heatmap
        if combined_heatmap.max() > combined_heatmap.min():
            combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Image with heatmap overlay
        axes[1].imshow(img)
        axes[1].imshow(combined_heatmap, alpha=0.5, cmap="jet")
        axes[1].set_title("Image + Court Keypoints")
        axes[1].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            log.info(f"Court sample visualization saved: {save_path}")
        else:
            default_path = self.output_dir / f"court_sample_{sample_idx:04d}.png"
            plt.savefig(default_path, bbox_inches="tight", dpi=150)
            log.info(f"Court sample visualization saved: {default_path}")
            
        plt.close()
    
    def visualize_player_sample(
        self,
        image: Union[torch.Tensor, np.ndarray],
        target: Dict[str, Any],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize player detection dataset sample.
        
        Args:
            image: Input image [C, H, W]
            target: Target dictionary containing boxes, labels, etc.
            sample_idx: Sample index for naming
            save_path: Optional path to save the visualization
        """
        # Denormalize image
        img = self._denormalize_image(image)
        H, W = img.shape[:2]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)
        ax.set_title("Image + Player Detections")
        ax.axis("off")
        
        # Draw bounding boxes
        if "boxes" in target and "labels" in target:
            boxes = target["boxes"]
            labels = target["labels"]
            
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
                
            for box, label in zip(boxes, labels):
                if len(box) == 4:
                    # Handle different box formats
                    if np.all(box <= 1.0):  # Normalized coordinates
                        if box[2] < 1.0 and box[3] < 1.0:  # [cx, cy, w, h] normalized
                            cx, cy, bw, bh = box
                            x1 = (cx - bw/2) * W
                            y1 = (cy - bh/2) * H
                            width = bw * W
                            height = bh * H
                        else:  # [x1, y1, x2, y2] normalized
                            x1, y1, x2, y2 = box * np.array([W, H, W, H])
                            width = x2 - x1
                            height = y2 - y1
                    else:  # Absolute coordinates
                        if box[2] > box[0]:  # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1
                        else:  # [x, y, w, h]
                            x1, y1, width, height = box
                    
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor="red", facecolor="none"
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    class_name = self.class_names.get(int(label), f"class_{int(label)}")
                    ax.text(
                        x1, y1-5, class_name,
                        fontsize=12, color="white",
                        bbox=dict(facecolor="red", alpha=0.7, pad=2)
                    )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            log.info(f"Player sample visualization saved: {save_path}")
        else:
            default_path = self.output_dir / f"player_sample_{sample_idx:04d}.png"
            plt.savefig(default_path, bbox_inches="tight", dpi=150)
            log.info(f"Player sample visualization saved: {default_path}")
            
        plt.close()
    
    def visualize_event_sample(
        self,
        sequence_data: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray, int],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize event classification dataset sample.
        
        Args:
            sequence_data: Input sequence data [T, F] where T is time steps, F is features
            target: Target class label or sequence
            sample_idx: Sample index for naming
            save_path: Optional path to save the visualization
        """
        if isinstance(sequence_data, torch.Tensor):
            sequence_data = sequence_data.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot sequence data
        if sequence_data.ndim == 2:
            T, F = sequence_data.shape
            for f in range(min(F, 5)):  # Plot up to 5 features
                axes[0].plot(sequence_data[:, f], label=f"Feature {f}")
            axes[0].set_title("Input Sequence Features")
            axes[0].set_xlabel("Time Steps")
            axes[0].set_ylabel("Feature Values")
            axes[0].legend()
            axes[0].grid(True)
        else:
            axes[0].plot(sequence_data.flatten())
            axes[0].set_title("Input Sequence")
            axes[0].set_xlabel("Time Steps")
            axes[0].set_ylabel("Values")
            axes[0].grid(True)
        
        # Plot target
        if np.isscalar(target):
            # Single class label
            class_name = self.class_names.get(int(target), f"class_{int(target)}")
            axes[1].bar([class_name], [1.0], color="blue", alpha=0.7)
            axes[1].set_title("Target Class")
            axes[1].set_ylabel("Probability")
        else:
            # Sequence or multi-class target
            if target.ndim == 1:
                if len(target) <= 10:  # Class probabilities
                    class_names = [self.class_names.get(i, f"class_{i}") for i in range(len(target))]
                    axes[1].bar(class_names, target, color="blue", alpha=0.7)
                    axes[1].set_title("Target Class Probabilities")
                    axes[1].set_ylabel("Probability")
                    plt.setp(axes[1].get_xticklabels(), rotation=45)
                else:  # Target sequence
                    axes[1].plot(target, color="red")
                    axes[1].set_title("Target Sequence")
                    axes[1].set_xlabel("Time Steps")
                    axes[1].set_ylabel("Target Values")
                    axes[1].grid(True)
            else:
                # 2D target (e.g., sequence of class probabilities)
                axes[1].imshow(target.T, aspect="auto", cmap="viridis")
                axes[1].set_title("Target Sequence (2D)")
                axes[1].set_xlabel("Time Steps")
                axes[1].set_ylabel("Classes")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            log.info(f"Event sample visualization saved: {save_path}")
        else:
            default_path = self.output_dir / f"event_sample_{sample_idx:04d}.png"
            plt.savefig(default_path, bbox_inches="tight", dpi=150)
            log.info(f"Event sample visualization saved: {default_path}")
            
        plt.close()
    
    def visualize_sample(
        self,
        data: Any,
        target: Any,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """
        Unified interface for visualizing dataset samples.
        Automatically determines the appropriate visualization method based on module type.
        
        Args:
            data: Input data (varies by task)
            target: Target data (varies by task)
            sample_idx: Sample index for naming
            save_path: Optional path to save the visualization
        """
        try:
            if self.module_name == "ball":
                self.visualize_ball_sample(data, target, sample_idx, save_path)
            elif self.module_name == "court":
                self.visualize_court_sample(data, target, sample_idx, save_path)
            elif self.module_name == "player":
                self.visualize_player_sample(data, target, sample_idx, save_path)
            elif self.module_name == "event":
                self.visualize_event_sample(data, target, sample_idx, save_path)
            else:
                log.warning(f"Unknown module type: {self.module_name}")
                
        except Exception as e:
            log.error(f"Error visualizing sample {sample_idx}: {e}")
            log.error("Stack trace:", exc_info=True)
    
    def create_datamodule(self):
        """Create the data module for loading dataset samples."""
        log.info(f"Creating DataModule for {self.module_name}...")
        return hydra.utils.instantiate(self.config.litdatamodule)
    
    def visualize_dataset_samples(
        self,
        num_samples: int = 10,
        split: str = "train"
    ) -> None:
        """
        Visualize multiple dataset samples.
        
        Args:
            num_samples: Number of samples to visualize
            split: Dataset split to use ("train", "val", "test")
        """
        log.info(f"Visualizing {num_samples} samples from {split} split...")
        
        try:
            # Create datamodule
            datamodule = self.create_datamodule()
            datamodule.setup("fit")
            
            # Get appropriate dataloader
            if split == "train":
                dataloader = datamodule.train_dataloader()
            elif split == "val":
                dataloader = datamodule.val_dataloader()
            elif split == "test":
                dataloader = datamodule.test_dataloader()
            else:
                raise ValueError(f"Unknown split: {split}")
            
            # Visualize samples
            sample_count = 0
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    data, targets = batch[0], batch[1]
                else:
                    log.warning(f"Unexpected batch format: {type(batch)}")
                    continue
                
                batch_size = data.shape[0] if hasattr(data, 'shape') else len(data)
                
                for i in range(min(batch_size, num_samples - sample_count)):
                    if hasattr(data, '__getitem__'):
                        sample_data = data[i]
                    else:
                        sample_data = data
                        
                    if hasattr(targets, '__getitem__'):
                        sample_target = targets[i]
                    else:
                        sample_target = targets
                    
                    self.visualize_sample(
                        sample_data,
                        sample_target,
                        sample_idx=sample_count,
                        save_path=None
                    )
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
                
                if sample_count >= num_samples:
                    break
                    
            log.info(f"Successfully visualized {sample_count} samples")
            
        except Exception as e:
            log.error(f"Error during dataset visualization: {e}")
            log.error("Stack trace:", exc_info=True)
            raise