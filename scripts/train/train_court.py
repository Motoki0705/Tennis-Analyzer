#!/usr/bin/env python
"""
コート検出モデルのトレーニング用スクリプト

使用例:
    python scripts/train/court/train_court.py
"""

import logging
import os
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch


def validate_model_litmodule_compatibility(model_config, litmodule_config):
    """
    モデルとLitModuleの出力タイプの互換性を検証する関数
    
    Args:
        model_config: モデルの設定
        litmodule_config: LitModuleの設定
        
    Raises:
        ValueError: 互換性がない場合
    """
    model_output_type = model_config.meta.output_type
    litmodule_output_type = litmodule_config.meta.output_type
    
    if model_output_type != litmodule_output_type:
        raise ValueError(
            f"モデルの出力タイプ '{model_output_type}' と "
            f"LitModuleの出力タイプ '{litmodule_output_type}' が一致しません。"
            f"モデル: {model_config.meta.name}, LitModule: {litmodule_config.meta.name}"
        )
    
    logging.info(f"モデルとLitModuleの出力タイプ '{model_output_type}' が一致しています")


def train(cfg: DictConfig) -> None:
    """
    コート検出モデルのトレーニングを実行する関数
    
    Args:
        cfg: Hydra設定
    """
    # シードを設定
    pl.seed_everything(cfg.get("seed", 42))
    
    # ロギングの設定
    logging.info(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # モデルとLitModuleの互換性をチェック
    validate_model_litmodule_compatibility(cfg.model, cfg.litmodule)
    
    # モデルをインスタンス化
    model = instantiate(cfg.model.net)
    logging.info(f"Model: {model.__class__.__name__}")
    
    # LightningModuleをインスタンス化
    lit_module = instantiate(cfg.litmodule.module, model=model)
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
    


@hydra.main(config_path="../../configs/train/court", config_name="config", version_base="1.3")
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