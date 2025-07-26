#!/usr/bin/env python
"""
Event Classification Dataset Visualization Script
Visualizes event classification dataset samples with sequence data.
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.common.visualization import DatasetVisualizer

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs/train/event", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for visualizing event classification dataset samples.
    
    Args:
        cfg: Hydra configuration
    """
    log.info("Starting event dataset visualization...")
    
    # Create output directory
    output_dir = "visualizations/event"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = DatasetVisualizer(
        config=cfg,
        module_name="event",
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
        
        log.info(f"Event dataset visualization completed. Check output in: {output_dir}")
        
    except Exception as e:
        log.error(f"Event dataset visualization failed: {e}")
        log.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    main()