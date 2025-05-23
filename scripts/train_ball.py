#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
テニスボール検出モデルの学習スクリプト
"""

import logging
import os
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

log = logging.getLogger(__name__)


def setup_callbacks(cfg: DictConfig) -> List[Callback]:
    """
    設定からコールバックを初期化します。

    Args:
        cfg: Hydra設定

    Returns:
        List[Callback]: 初期化されたコールバックのリスト
    """
    callbacks = []
    
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))
    
    return callbacks


@hydra.main(version_base="1.3", config_path="../configs/train/ball", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    ボール検出モデルの学習メイン関数

    Args:
        cfg: Hydra設定
    """
    # 再現性のために乱数シードを設定
    pl.seed_everything(cfg.seed)
    
    # バージョン名（ディレクトリ名）の設定
    version = cfg.get("version", "ball_detection")
    
    # ロガーの設定
    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.trainer.default_root_dir, "tb_logs"),
        name=version
    )
    
    # コールバックの設定
    callbacks = setup_callbacks(cfg)
    
    # データモジュールの初期化
    log.info(f"Instantiating datamodule <{cfg.litdatamodule._target_}>")
    datamodule = instantiate(cfg.litdatamodule)
    
    # 基本モデルの初期化
    log.info(f"Instantiating base model <{cfg.model._target_}>")
    base_model = instantiate(cfg.model)
    
    # LightningModuleの初期化（基本モデルを渡す）
    log.info(f"Instantiating LightningModule <{cfg.litmodule._target_}>")
    lit_model = instantiate(cfg.litmodule, model=base_model)
    
    # トレーナーの初期化
    log.info(f"Instantiating trainer <{cfg.trainer}>")
    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    
    # 学習の実行
    log.info("Starting training!")
    trainer.fit(model=lit_model, datamodule=datamodule)
    
    # 学習後のテスト（オプション）
    if cfg.get("test_after_training", False):
        trainer.test(model=lit_model, datamodule=datamodule)
    
    # ベストモデルのパスをログ
    best_model_path = None
    for callback in callbacks:
        if hasattr(callback, "best_model_path"):
            best_model_path = callback.best_model_path
            break
            
    if best_model_path:
        log.info(f"Best model path: {best_model_path}")
            
    log.info("Training completed!")


if __name__ == "__main__":
    main() 