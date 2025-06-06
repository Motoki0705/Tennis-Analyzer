#!/usr/bin/env python
"""
イベント検出モデルのトレーニングスクリプト（Hydra設定対応版）
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# ルートディレクトリへのパスを確保
from hydra.utils import get_original_cwd, to_absolute_path

# 再現性のための設定
import random
import numpy as np
import torch

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """再現性のためのシードを設定する"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnnが有効な場合は再現性のために次の設定を追加
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(
    config_path="../../configs/train/event", 
    config_name="config", 
    version_base=None
)
def main(cfg: DictConfig) -> None:
    """
    Hydra設定に基づいてイベント検出モデルを訓練するメイン関数

    Args:
        cfg: Hydra設定
    """
    # 再現性のためのシード設定
    if "seed" in cfg:
        set_seed(cfg.seed)
        log.info(f"シード固定: {cfg.seed}")

    # 設定内容を表示
    log.info(f"設定内容:\n{OmegaConf.to_yaml(cfg)}")
    
    # 絶対パスに変換（相対パスで設定されている場合）
    if "annotation_file" in cfg.litdatamodule:
        cfg.litdatamodule.annotation_file = to_absolute_path(cfg.litdatamodule.annotation_file)

    # ディレクトリの準備
    os.makedirs(cfg.trainer.default_root_dir, exist_ok=True)
    log.info(f"出力ディレクトリ: {cfg.trainer.default_root_dir}")

    # LitDataModuleの初期化
    log.info("DataModuleを初期化中...")
    datamodule = hydra.utils.instantiate(cfg.litdatamodule)
    
    # データモジュールのセットアップ
    datamodule.setup(stage="fit")
    feature_dims = datamodule.get_feature_dims()
    max_players = datamodule.get_max_players()
    
    log.info(f"特徴次元: {feature_dims}")
    log.info(f"最大プレイヤー数: {max_players}")

    # モデル構築の準備
    log.info("モデルを構築中...")
    # モデルパラメータ設定
    model_params = OmegaConf.to_container(cfg.model.net, resolve=True)
    
    # バックボーン関連のパラメータを追加
    model_params.update({
        "ball_dim": feature_dims["ball"],
        "player_bbox_dim": feature_dims["player_bbox"],
        "player_pose_dim": feature_dims["player_pose"],
        "court_dim": feature_dims["court"],
        "max_players": max_players,
    })
    
    # モデルの構築
    model = hydra.utils.call(cfg.model.net._target_, **model_params)
    
    # LitModuleの初期化
    log.info("LightningModuleを初期化中...")
    lit_module_params = OmegaConf.to_container(cfg.litmodule.module, resolve=True)
    lit_module_params["model"] = model
    lit_module = hydra.utils.call(cfg.litmodule.module._target_, **lit_module_params)

    # コールバックの準備
    callbacks = []
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"コールバックを追加: {cb_name}")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # ロガーの準備
    logger = TensorBoardLogger(
        save_dir=cfg.trainer.default_root_dir,
        name="",
    )

    # トレーナーの準備
    log.info("トレーナーを初期化中...")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **{k: v for k, v in cfg.trainer.items() if k != "default_root_dir"}
    )

    # トレーニングの実行
    log.info("トレーニング開始...")
    trainer.fit(lit_module, datamodule=datamodule)
    
    # ベストモデルのパスを表示
    best_model_path = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
            best_score = callback.best_model_score
            break
    
    if best_model_path:
        log.info(f"最高性能のモデルパス: {best_model_path}")
        log.info(f"最高性能のスコア: {best_score:.4f}")

if __name__ == "__main__":
    main() 