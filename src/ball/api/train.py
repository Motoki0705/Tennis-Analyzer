#!/usr/bin/env python
"""
ボール検出モデルのトレーニングスクリプト
"""
import glob
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


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    指定されたディレクトリから最新のチェックポイントファイルを検索する
    
    Args:
        checkpoint_dir: チェックポイントディレクトリのパス
        
    Returns:
        最新のチェックポイントファイルのパス、見つからない場合はNone
    """
    try:
        checkpoint_path = Path(checkpoint_dir)
        
        # ディレクトリが存在しない場合
        if not checkpoint_path.exists():
            log.info(f"チェックポイントディレクトリが存在しません: {checkpoint_dir}")
            return None
            
        # .ckptファイルを検索
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
        
        if not ckpt_files:
            log.info(f"チェックポイントファイルが見つかりません: {checkpoint_dir}")
            return None
            
        # 最新のファイルを取得（更新日時順）
        latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        log.info(f"最新のチェックポイントを発見: {latest_ckpt}")
        
        return str(latest_ckpt)
        
    except Exception as e:
        log.error(f"チェックポイント検索中にエラーが発生: {e}")
        return None


def setup_callbacks(cfg: DictConfig) -> list:
    """
    トレーニング用のコールバックを設定する
    
    Args:
        cfg: Hydra設定
        
    Returns:
        設定されたコールバックのリスト
    """
    callbacks = []
    
    try:
        # モデルチェックポイント
        if "checkpoint" in cfg.callbacks:
            log.info("ModelCheckpointを設定中...")
            checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint)
            callbacks.append(checkpoint_callback)
        
        # 早期停止
        if "early_stopping" in cfg.callbacks:
            log.info("EarlyStoppingを設定中...")
            early_stopping = hydra.utils.instantiate(cfg.callbacks.early_stopping)
            callbacks.append(early_stopping)
        
        # 学習率モニタリング
        if "lr_monitor" in cfg.callbacks:
            log.info("LearningRateMonitorを設定中...")
            lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
            callbacks.append(lr_monitor)
            
    except Exception as e:
        log.error(f"コールバック設定中にエラーが発生: {e}")
        raise
        
    return callbacks


def get_best_model_info(callbacks: list) -> tuple[Optional[str], Optional[float]]:
    """
    トレーニング完了後に最良モデルの情報を取得する
    
    Args:
        callbacks: コールバックのリスト
        
    Returns:
        最良モデルのパスとスコアのタプル
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
        log.error(f"最良モデル情報取得中にエラーが発生: {e}")
        
    return best_model_path, best_score


@hydra.main(config_path="../../../configs/train/ball", config_name="lite_tracknet_focal", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    ボール検出モデルのトレーニングを実行する

    Args:
        cfg: Hydra設定
    """
    log.info("ボール検出モデルのトレーニングを開始します...")
    
    try:
        # 乱数シードを設定
        pl.seed_everything(cfg.get("seed", 42))

        # 設定の出力
        log.info(f"設定: \n{cfg}")
        
        # チェックポイントからの再開を確認
        checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints/ball")
        resume_ckpt = find_latest_checkpoint(checkpoint_dir)
        
        if resume_ckpt:
            log.info(f"チェックポイントから再開します: {resume_ckpt}")
        else:
            log.info("初めからトレーニングを開始します")
        
        # DataModuleの作成
        log.info("DataModuleを作成中...")
        datamodule = hydra.utils.instantiate(cfg.litdatamodule)
        
        # LightningModuleの作成
        log.info("LightningModuleを作成中...")
        # Hydraがネストされた設定（model, criterionなど）を自動的にインスタンス化し、
        # LitGenericBallModelに注入します。
        lit_module = hydra.utils.instantiate(cfg.lit_module)
        
        # Callbacksの設定
        callbacks = setup_callbacks(cfg)
        
        # Trainerの作成
        log.info("Trainerを作成中...")
        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=callbacks,
        )
        
        # トレーニングの実行（チェックポイントがあれば再開）
        log.info("トレーニング開始...")
        trainer.fit(lit_module, datamodule=datamodule, ckpt_path=resume_ckpt)
        
        # 最高性能のモデルチェックポイントの情報を取得
        best_model_path, best_score = get_best_model_info(callbacks)
        
        if best_model_path:
            log.info(f"最高性能のモデルチェックポイント: {best_model_path}")
            if best_score:
                log.info(f"最高性能のスコア: {best_score:.4f}")
        
        log.info("トレーニングが正常に完了しました")
        
    except Exception as e:
        log.error(f"トレーニング中にエラーが発生: {e}")
        log.error(f"スタックトレース: ", exc_info=True)
        raise


if __name__ == "__main__":
    main() 