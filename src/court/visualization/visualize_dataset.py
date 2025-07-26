#!/usr/bin/env python
"""
Court Detection Dataset Visualization Script
Visualizes court detection dataset samples with keypoint heatmaps.
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.common.visualization import DatasetVisualizer

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs/train/court", config_name="lite_tracknet_generic", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for visualizing court detection dataset samples.
    
    Args:
        cfg: Hydra configuration
    """
    log.info("Starting court dataset visualization...")
    
    # Create output directory
    output_dir = "visualizations/court"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = DatasetVisualizer(
        config=cfg,
        module_name="court",
        output_dir=output_dir
    )
    
    # Get parameters from config or defaults
    num_samples = getattr(cfg, 'num_samples', 20)
    split = getattr(cfg, 'split', 'train')
    
    log.info(f"Visualizing {num_samples} samples from {split} split...")
    
    try:
        # Visualize dataset samples
        visualizer.visualize_dataset_samples(
            num_samples=num_samples,
            split=split
        )
        
        log.info(f"Court dataset visualization completed. Check output in: {output_dir}")
        
    except Exception as e:
        log.error(f"Court dataset visualization failed: {e}")
        log.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    main()