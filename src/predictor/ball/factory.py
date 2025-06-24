"""
Factory for creating ball detectors.

This module provides a factory function for creating ball detector instances
with automatic model type detection and configuration.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ..base.detector import BaseBallDetector
from .lite_tracknet_detector import LiteTrackNetDetector
from .wasb_sbdt_detector import WASBSBDTDetector

logger = logging.getLogger(__name__)


def create_ball_detector(
    model_path: str,
    model_type: str = "auto",
    config_path: Optional[str] = None,
    device: str = "auto",
    **kwargs
) -> BaseBallDetector:
    """Create a ball detector instance.
    
    Args:
        model_path: Path to the trained model file
        model_type: Type of model ('lite_tracknet', 'wasb_sbdt', or 'auto')
        config_path: Path to configuration file (optional)
        device: Device for inference ('cuda', 'cpu', or 'auto')
        **kwargs: Additional arguments passed to detector constructor
        
    Returns:
        Ball detector instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model type cannot be determined or is unsupported
        RuntimeError: If detector creation fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Auto-detect model type if not specified
    if model_type == "auto":
        model_type = detect_model_type(model_path)
    
    # Create appropriate detector
    if model_type == "lite_tracknet":
        return LiteTrackNetDetector(
            model_path=model_path,
            device=device,
            **kwargs
        )
    elif model_type == "wasb_sbdt":
        return WASBSBDTDetector(
            model_path=model_path,
            config_path=config_path,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def detect_model_type(model_path: str) -> str:
    """Auto-detect model type from file extension and path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Detected model type ('lite_tracknet' or 'wasb_sbdt')
        
    Raises:
        ValueError: If model type cannot be determined
    """
    path = Path(model_path)
    extension = path.suffix.lower()
    
    # Check file extension
    if extension == ".ckpt":
        return "lite_tracknet"
    elif extension in [".pth", ".tar"]:
        return "wasb_sbdt"
    
    # Check path components for hints
    path_str = str(path).lower()
    if "lite_tracknet" in path_str or "tracknet" in path_str:
        return "lite_tracknet"
    elif "wasb" in path_str or "sbdt" in path_str:
        return "wasb_sbdt"
    
    # Default fallback based on common patterns
    if extension == ".ckpt":
        logger.warning(f"Assuming LiteTrackNet for .ckpt file: {model_path}")
        return "lite_tracknet"
    elif extension in [".pth", ".pth.tar"]:
        logger.warning(f"Assuming WASB-SBDT for .pth file: {model_path}")
        return "wasb_sbdt"
    
    raise ValueError(
        f"Cannot determine model type from path: {model_path}. "
        f"Please specify model_type explicitly."
    )


def get_available_models(checkpoints_dir: str = "checkpoints/ball") -> Dict[str, list]:
    """Get available model files organized by type.
    
    Args:
        checkpoints_dir: Directory to search for model files
        
    Returns:
        Dictionary with model types as keys and file paths as values
    """
    available_models = {
        "lite_tracknet": [],
        "wasb_sbdt": [],
        "unknown": []
    }
    
    if not os.path.exists(checkpoints_dir):
        logger.warning(f"Checkpoints directory not found: {checkpoints_dir}")
        return available_models
    
    # Search for model files
    for root, dirs, files in os.walk(checkpoints_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                model_type = detect_model_type(file_path)
                available_models[model_type].append(file_path)
            except ValueError:
                available_models["unknown"].append(file_path)
    
    return available_models


def validate_model_compatibility(model_path: str, model_type: str) -> bool:
    """Validate that a model file is compatible with the specified type.
    
    Args:
        model_path: Path to the model file
        model_type: Expected model type
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        detected_type = detect_model_type(model_path)
        return detected_type == model_type
    except ValueError:
        return False