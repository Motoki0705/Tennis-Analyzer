#!/usr/bin/env python
"""
ボール検出モデルのトレーニング用スクリプト

使用例:
    python scripts/train/train_ball.py
"""

import logging
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def setup_logging(cfg: DictConfig) -> None:
    """
    ロギング設定を初期化する関数
    
    Args:
        cfg: Hydra設定
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def train(cfg: DictConfig) -> None:
    """
    ボール検出モデルのトレーニングを実行する関数
    
    Args:
        cfg: Hydra設定
    """
    try:
        # ロギング設定
        setup_logging(cfg)
        
        # シードを設定
        pl.seed_everything(cfg.get("seed", 42))
        
        # 設定内容をログ出力
        logging.info(f"Training configuration:\n{OmegaConf.to_yaml(cfg)}")
        
        # LightningModuleをインスタンス化
        lit_module = instantiate(cfg.litmodule.module)
        logging.info(f"LightningModule: {lit_module.__class__.__name__}")
        
        # DataModuleをインスタンス化
        datamodule = instantiate(cfg.litdatamodule)
        logging.info(f"DataModule: {datamodule.__class__.__name__}")
        
        # コールバックをインスタンス化
        callbacks = []
        if "callbacks" in cfg:
            for callback_name, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    callbacks.append(instantiate(cb_conf))
                    logging.info(f"Callback added: {callback_name}")
        
        # トレーナーをインスタンス化
        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=callbacks,
        )
        logging.info(f"Trainer configured with {len(callbacks)} callbacks")
        
        # トレーニングを実行
        logging.info("Starting training...")
        trainer.fit(lit_module, datamodule=datamodule)
        
        # テストを実行（テストデータローダーが存在する場合）
        if hasattr(datamodule, "test_dataloader") and datamodule.test_dataloader() is not None:
            logging.info("Starting testing...")
            trainer.test(lit_module, datamodule=datamodule)
        else:
            logging.info("No test dataloader found, skipping testing")
            
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


@hydra.main(config_path="../../configs/train/ball", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    メイン関数
    
    Args:
        cfg: Hydra設定
    """
    train(cfg)


if __name__ == "__main__":
    main() 