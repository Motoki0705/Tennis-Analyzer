#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ボール検出タスクのSelf-Trainingスクリプト

使用例:
    # デフォルト設定で実行
    python scripts/train/train_ball_self_training.py initial_checkpoint=path/to/checkpoint.ckpt
    
    # モデルアーキテクチャを変更
    python scripts/train/train_ball_self_training.py initial_checkpoint=path/to/checkpoint.ckpt model=_sequence
    
    # 擬似ラベルのパラメータを調整
    python scripts/train/train_ball_self_training.py initial_checkpoint=path/to/checkpoint.ckpt self_training.confidence_threshold=0.8
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from src.ball.self_training.self_training_cycle import BallSelfTrainingCycle

# ロガー設定
log = logging.getLogger(__name__)


def setup_callbacks(cfg: DictConfig) -> List[Callback]:
    """
    設定からコールバックを初期化します。
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra設定
    
    Returns
    -------
    callbacks : List[Callback]
        コールバックのリスト
    """
    callbacks = []
    
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))
    
    return callbacks


def load_initial_model(cfg: DictConfig) -> nn.Module:
    """
    初期モデルを読み込む
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra設定
    
    Returns
    -------
    model : nn.Module
        読み込まれたモデル
    """
    if cfg.initial_checkpoint is None:
        raise ValueError("initial_checkpoint must be specified for self-training")
    
    checkpoint_path = Path(cfg.initial_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    log.info(f"Loading initial model from {checkpoint_path}")
    
    # チェックポイントから LightningModule を読み込む
    try:
        # LightningModuleとして読み込み
        lit_module_class = cfg.litmodule._target_
        lit_module = hydra.utils.get_class(lit_module_class).load_from_checkpoint(
            checkpoint_path,
            map_location="cpu"
        )
        # 内部のモデルを取得
        model = lit_module.model
        log.info("Successfully loaded model from LightningModule checkpoint")
    except Exception as e:
        log.warning(f"Failed to load as LightningModule: {e}")
        # 通常のPyTorchモデルとして読み込み
        try:
            model = instantiate(cfg.model)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            log.info("Successfully loaded model as PyTorch model")
        except Exception as e2:
            log.error(f"Failed to load model: {e2}")
            raise
    
    return model


def run_self_training_cycle(
    cfg: DictConfig,
    model: nn.Module,
    datamodule: pl.LightningDataModule,
) -> Tuple[nn.Module, float, Dict]:
    """
    Self-trainingサイクルを実行
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra設定
    model : nn.Module
        初期モデル
    datamodule : pl.LightningDataModule
        データモジュール
    
    Returns
    -------
    best_model : nn.Module
        最良のモデル
    best_score : float
        最良スコア
    metrics : Dict
        メトリクス
    """
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # データセットの準備
    datamodule.setup("fit")
    datasets = datamodule.get_dataset_for_self_training()
    
    # Self-trainingサイクルの初期化
    self_training = BallSelfTrainingCycle(
        model=model.to(device),
        labeled_dataset=datasets["labeled"],
        unlabeled_dataset=datasets["unlabeled"],
        val_dataset=datasets["val"],
        save_dir=cfg.self_training.save_dir,
        device=device,
        confidence_threshold=cfg.self_training.confidence_threshold,
        max_cycles=cfg.self_training.max_cycles,
        pseudo_label_weight=cfg.self_training.pseudo_label_weight,
        trajectory_params=cfg.self_training.trajectory_params,
        use_trajectory_tracking=cfg.self_training.use_trajectory_tracking,
    )
    
    # Self-trainingの実行
    best_model, best_score, metrics = self_training.run_self_training()
    
    return best_model, best_score, metrics


def train_final_model(
    cfg: DictConfig,
    model: nn.Module,
    datamodule: pl.LightningDataModule,
) -> None:
    """
    最終的なモデルをトレーニング（オプション）
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra設定
    model : nn.Module
        Self-trainingで得られたモデル
    datamodule : pl.LightningDataModule
        データモジュール
    """
    # LightningModuleの初期化
    log.info(f"Instantiating LightningModule <{cfg.litmodule._target_}>")
    lit_model = instantiate(cfg.litmodule, model=model)
    
    # バージョン名（ディレクトリ名）の設定
    version = cfg.get("version", "default_version")
    
    # ロガーの設定
    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.trainer.default_root_dir, "tb_logs"),
        name=version
    )
    
    # コールバックの設定
    callbacks = setup_callbacks(cfg)
    
    # トレーナーの初期化
    log.info(f"Instantiating trainer")
    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    
    # 最終トレーニング
    log.info("Starting final training with all pseudo labels!")
    trainer.fit(model=lit_model, datamodule=datamodule)
    
    # テスト（オプション）
    if cfg.get("test_after_training", False):
        trainer.test(model=lit_model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../../configs/train/ball/self_training", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    メインのSelf-training関数
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra設定
    """
    # 再現性のために乱数シードを設定
    pl.seed_everything(cfg.seed)
    
    try:
        # 初期モデルの読み込み
        initial_model = load_initial_model(cfg)
        
        # データモジュールの初期化
        log.info(f"Instantiating datamodule <{cfg.litdatamodule._target_}>")
        datamodule = instantiate(cfg.litdatamodule)
        
        # Self-trainingサイクルの実行
        log.info("Starting self-training cycle")
        best_model, best_score, metrics = run_self_training_cycle(
            cfg, initial_model, datamodule
        )
        
        log.info(f"Self-training completed with best score: {best_score:.4f}")
        
        # メトリクスの保存
        metrics_path = Path(cfg.self_training.save_dir) / "self_training_metrics.json"
        import json
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Saved metrics to {metrics_path}")
        
        # オプション：最終的なファインチューニング
        if cfg.get("final_finetuning", False):
            log.info("Starting final fine-tuning with all data")
            # 最新の擬似ラベルでデータモジュールを更新
            latest_pseudo_labels = sorted(
                Path(cfg.self_training.save_dir).glob("pseudo_labels/pseudo_labels_cycle_*.json")
            )[-1]
            datamodule.update_pseudo_labels(latest_pseudo_labels)
            
            # 最終トレーニング
            train_final_model(cfg, best_model, datamodule)
        
        log.info("Self-training pipeline completed successfully!")
        
    except Exception as e:
        log.error(f"Error in self-training: {e}")
        raise


if __name__ == "__main__":
    main() 