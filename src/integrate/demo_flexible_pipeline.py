"""
Demo script for the flexible tennis analysis pipeline.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

from .core.flexible_pipeline import FlexiblePipeline

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs/infer/integrate", config_name="flexible_pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Demo script for the flexible tennis analysis pipeline.
    
    Usage:
        # Basic usage with default config
        python demo_flexible_pipeline.py io.input_video=/path/to/video.mp4
        
        # Enable specific tasks only
        python demo_flexible_pipeline.py io.input_video=/path/to/video.mp4 \
            tasks.0.enabled=true tasks.1.enabled=false tasks.2.enabled=false tasks.3.enabled=false
        
        # Single-threaded mode with output video
        python demo_flexible_pipeline.py io.input_video=/path/to/video.mp4 \
            io.output_video=/path/to/output.mp4 threading.mode=single
        
        # Multi-threaded mode with larger batch size
        python demo_flexible_pipeline.py io.input_video=/path/to/video.mp4 \
            threading.mode=multi batch_size=8
    """
    
    # Print configuration for debugging
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Validate required configuration
    if not hasattr(cfg.io, 'input_video') or not cfg.io.input_video:
        raise ValueError("Input video path is required. Please set io.input_video.")
    
    if not os.path.exists(cfg.io.input_video):
        raise FileNotFoundError(f"Input video not found: {cfg.io.input_video}")
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.get('level', 'INFO').upper()),
        format=cfg.logging.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    )
    
    # Log enabled tasks
    enabled_tasks = [task for task in cfg.tasks if task.get('enabled', True)]
    log.info(f"Enabled tasks: {[task['name'] for task in enabled_tasks]}")
    
    try:
        # Create and run pipeline
        pipeline = FlexiblePipeline(cfg)
        pipeline.run()
        
        log.info("Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        
    except Exception as e:
        log.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()