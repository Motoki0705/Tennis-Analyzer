#!/usr/bin/env python
"""
イベント検出モデルのトレーニングスクリプト
"""
import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# ロガーの設定
log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs/train/event", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    イベント検出モデルのトレーニングを実行する

    Args:
        cfg: Hydra設定
    """
    log.info("イベント検出モデルのトレーニングを開始します...")
    
    # 乱数シードを設定
    pl.seed_everything(cfg.get("seed", 42))
    
    # 設定の出力
    log.info(f"設定: \n{cfg}")
    
    # DataModuleの作成
    log.info("DataModuleを作成中...")
    
    # バランス調整したデータセットのパスが指定されていれば、litdatamoduleの設定を更新
    balanced_dataset_path = cfg.get("balanced_dataset_path", None)
    if balanced_dataset_path and os.path.exists(balanced_dataset_path):
        log.info(f"バランス調整済みデータセットを使用します: {balanced_dataset_path}")
        cfg.litdatamodule.annotation_file = balanced_dataset_path
    
    datamodule = hydra.utils.instantiate(cfg.litdatamodule)
    
    # モデルの作成
    log.info("モデルを作成中...")
    model = hydra.utils.instantiate(cfg.model.net)
    
    # LightningModuleの作成
    log.info("LightningModuleを作成中...")
    lit_module = hydra.utils.instantiate(
        cfg.litmodule.module,
        model=model,
    )
    
    # Callbacksの設定
    callbacks = []
    
    # モデルチェックポイント
    if "checkpoint" in cfg.callbacks:
        log.info("ModelCheckpointを設定中...")
        checkpoint_callback = ModelCheckpoint(**cfg.callbacks.checkpoint)
        callbacks.append(checkpoint_callback)
    
    # 早期停止
    if "early_stopping" in cfg.callbacks:
        log.info("EarlyStoppingを設定中...")
        early_stopping = EarlyStopping(**cfg.callbacks.early_stopping)
        callbacks.append(early_stopping)
    
    # 学習率モニタリング
    if "lr_monitor" in cfg.callbacks:
        log.info("LearningRateMonitorを設定中...")
        lr_monitor = LearningRateMonitor(**cfg.callbacks.lr_monitor)
        callbacks.append(lr_monitor)
    
    # Trainerの作成
    log.info("Trainerを作成中...")
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
    )
    
    # トレーニングの実行
    log.info("トレーニング開始...")
    trainer.fit(lit_module, datamodule=datamodule)
    
    # 最高性能のモデルチェックポイントのパスを取得
    best_model_path = None
    best_score = None
    
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and hasattr(callback, "best_model_path"):
            best_model_path = callback.best_model_path
            best_score = callback.best_model_score.item() if callback.best_model_score else None
            break
    
    if best_model_path:
        log.info(f"最高性能のモデルチェックポイント: {best_model_path}")
        if best_score:
            log.info(f"最高性能のスコア: {best_score:.4f}")


if __name__ == "__main__":
    main() 