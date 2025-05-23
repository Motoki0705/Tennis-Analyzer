import logging
import os
from typing import List, Optional

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch


def train(cfg: DictConfig) -> None:
    """
    プレイヤー検出モデルのトレーニングを実行する関数
    
    Args:
        cfg: Hydra設定
    """
    # シードを設定
    pl.seed_everything(cfg.get("seed", 42))
    
    # ロギングの設定
    logging.info(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # モデルをインスタンス化
    model = instantiate(cfg.model)
    logging.info(f"Model: {model.__class__.__name__}")
    
    # LightningModuleをインスタンス化
    lit_module = instantiate(cfg.litmodule, model=model)
    logging.info(f"LightningModule: {lit_module.__class__.__name__}")
    
    # DataModuleをインスタンス化
    datamodule = instantiate(cfg.litdatamodule)
    logging.info(f"DataModule: {datamodule.__class__.__name__}")
    
    # コールバックをインスタンス化
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate(cb_conf))
    
    # トレーナーをインスタンス化
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
    )
    
    # トレーニングを実行
    trainer.fit(lit_module, datamodule=datamodule)
    
    # テストを実行
    if hasattr(datamodule, "test_dataloader"):
        trainer.test(lit_module, datamodule=datamodule)


@hydra.main(config_path="../../configs/train/player", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    メイン関数
    
    Args:
        cfg: Hydra設定
    """
    try:
        train(cfg)
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 