"""
Standardized training API for all model types.
Provides unified interface for training ball, court, event, player, and pose models.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

log = logging.getLogger(__name__)


class TrainingManager:
    """
    Unified training manager for all model types.
    Handles common training workflow including:
    - Configuration validation
    - Checkpoint management
    - Callback setup
    - Training execution
    - Results reporting
    """

    def __init__(self, config: DictConfig, module_name: str):
        """
        Initialize training manager.
        
        Args:
            config: Hydra configuration
            module_name: Module name (ball, court, event, player, pose)
        """
        self.config = config
        self.module_name = module_name
        self.callbacks = []
        self.best_model_path = None
        self.best_score = None

    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint file in the given directory.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            
        Returns:
            Path to latest checkpoint file, or None if not found
        """
        try:
            checkpoint_path = Path(checkpoint_dir)
            
            if not checkpoint_path.exists():
                log.info(f"Checkpoint directory does not exist: {checkpoint_dir}")
                return None
                
            # Search for .ckpt files
            ckpt_files = list(checkpoint_path.glob("*.ckpt"))
            
            if not ckpt_files:
                log.info(f"No checkpoint files found in: {checkpoint_dir}")
                return None
                
            # Get the most recent file
            latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
            log.info(f"Found latest checkpoint: {latest_ckpt}")
            
            return str(latest_ckpt)
            
        except Exception as e:
            log.error(f"Error searching for checkpoint: {e}")
            return None

    def setup_callbacks(self) -> List[pl.Callback]:
        """
        Set up training callbacks.
        
        Returns:
            List of configured callbacks
        """
        callbacks = []
        
        try:
            # Model checkpoint callback
            if "checkpoint" in self.config.callbacks:
                log.info("Setting up ModelCheckpoint callback...")
                checkpoint_callback = hydra.utils.instantiate(self.config.callbacks.checkpoint)
                callbacks.append(checkpoint_callback)
            
            # Early stopping callback
            if "early_stopping" in self.config.callbacks:
                log.info("Setting up EarlyStopping callback...")
                early_stopping = hydra.utils.instantiate(self.config.callbacks.early_stopping)
                callbacks.append(early_stopping)
            
            # Learning rate monitor callback
            if "lr_monitor" in self.config.callbacks:
                log.info("Setting up LearningRateMonitor callback...")
                lr_monitor = hydra.utils.instantiate(self.config.callbacks.lr_monitor)
                callbacks.append(lr_monitor)
            
            # Module-specific callbacks
            module_callbacks = self._setup_module_specific_callbacks()
            callbacks.extend(module_callbacks)
                
        except Exception as e:
            log.error(f"Error setting up callbacks: {e}")
            raise
            
        return callbacks

    def _setup_module_specific_callbacks(self) -> List[pl.Callback]:
        """
        Set up module-specific callbacks.
        Can be overridden by subclasses for specialized callbacks.
        
        Returns:
            List of module-specific callbacks
        """
        callbacks = []
        
        # Court module has a special heatmap visualization callback
        if self.module_name == "court":
            try:
                from src.court.callbacks.save_heatmap import HeatmapVisualizerCallback
                log.info("Setting up HeatmapVisualizerCallback for court module...")
                heatmap_visualizer = HeatmapVisualizerCallback()
                callbacks.append(heatmap_visualizer)
            except ImportError:
                log.warning("Could not import HeatmapVisualizerCallback")
        
        return callbacks

    def get_best_model_info(self, callbacks: List[pl.Callback]) -> Tuple[Optional[str], Optional[float]]:
        """
        Get information about the best model from callbacks.
        
        Args:
            callbacks: List of training callbacks
            
        Returns:
            Tuple of (best_model_path, best_score)
        """
        best_model_path = None
        best_score = None
        
        try:
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint) and hasattr(callback, "best_model_path"):
                    best_model_path = callback.best_model_path
                    best_score = callback.best_model_score.item() if callback.best_model_score else None
                    break
                    
        except Exception as e:
            log.error(f"Error getting best model info: {e}")
            
        return best_model_path, best_score

    def validate_configuration(self):
        """Validate the training configuration."""
        required_sections = ["litdatamodule", "litmodule", "trainer", "callbacks"]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        log.info("Configuration validation passed")

    def create_datamodule(self):
        """Create the data module."""
        log.info("Creating DataModule...")
        return hydra.utils.instantiate(self.config.litdatamodule)

    def create_litmodule(self):
        """Create the lightning module."""
        log.info("Creating LightningModule...")
        return hydra.utils.instantiate(self.config.litmodule)

    def create_trainer(self, callbacks: List[pl.Callback]):
        """Create the PyTorch Lightning trainer."""
        log.info("Creating Trainer...")
        return pl.Trainer(
            **self.config.trainer,
            callbacks=callbacks,
        )

    def run_training(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Execute the complete training workflow.
        
        Returns:
            Tuple of (best_model_path, best_score)
        """
        log.info(f"Starting {self.module_name} model training...")
        
        try:
            # Set random seed for reproducibility
            pl.seed_everything(self.config.get("seed", 42))
            
            # Validate configuration
            self.validate_configuration()
            
            # Log configuration
            log.info(f"Configuration: \n{self.config}")
            
            # Check for checkpoint resumption
            checkpoint_dir = self.config.get("checkpoint_dir", f"checkpoints/{self.module_name}")
            resume_ckpt = self.find_latest_checkpoint(checkpoint_dir)
            
            if resume_ckpt:
                log.info(f"Resuming from checkpoint: {resume_ckpt}")
            else:
                log.info("Starting training from scratch")
            
            # Create components
            datamodule = self.create_datamodule()
            lit_module = self.create_litmodule()
            callbacks = self.setup_callbacks()
            trainer = self.create_trainer(callbacks)
            
            # Execute training
            log.info("Starting training...")
            trainer.fit(lit_module, datamodule=datamodule, ckpt_path=resume_ckpt)
            
            # Get best model information
            best_model_path, best_score = self.get_best_model_info(callbacks)
            
            if best_model_path:
                log.info(f"Best model checkpoint: {best_model_path}")
                if best_score:
                    log.info(f"Best model score: {best_score:.4f}")
            
            # Run testing if test dataloader is available
            if hasattr(datamodule, "test_dataloader") and datamodule.test_dataloader is not None:
                log.info("Running testing...")
                trainer.test(lit_module, datamodule=datamodule)
            
            log.info("Training completed successfully")
            
            return best_model_path, best_score
            
        except Exception as e:
            log.error(f"Training failed: {e}")
            log.error("Stack trace:", exc_info=True)
            raise

    @staticmethod
    def create_training_script(module_name: str, config_name: str):
        """
        Create a standardized training script function.
        This can be used as a decorator or called directly.
        
        Args:
            module_name: Name of the module (ball, court, event, player, pose)
            config_name: Default configuration name
            
        Returns:
            Training function that can be used with Hydra
        """
        def training_main(cfg: DictConfig) -> None:
            """Main training function."""
            manager = TrainingManager(cfg, module_name)
            manager.run_training()
        
        # Set up Hydra decorator
        config_path = f"../../configs/train/{module_name}"
        return hydra.main(
            config_path=config_path,
            config_name=config_name,
            version_base=None
        )(training_main)