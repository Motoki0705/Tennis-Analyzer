"""
ballモデルの推論設定ファイルのインスタンス化テスト
"""
import os
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from src.utils.model_utils import extract_hparams_from_ckpt, extract_model_config_from_ckpt

# テスト対象のモデル設定ファイル
BALL_MODEL_CONFIGS = [
    "lite_tracknet",
    "lite_tracknet_xy",
    "resnet_regression",
    "swin_448",
]


@pytest.mark.parametrize("model_config", BALL_MODEL_CONFIGS)
def test_ball_model_instantiate(model_config):
    """
    ball推論モデルが正しくインスタンス化できることをテストします
    """
    # hydraの初期化
    with initialize(version_base=None, config_path="../../configs/infer/ball"):
        # モデル設定の読み込み
        cfg = compose(config_name=model_config)
        
        print(f"\nTesting model: {model_config}")
        print(f"Model config: {OmegaConf.to_yaml(cfg)}")
        
        try:    
            # ckpt_pathを除外してモデルをインスタンス化
            model_cfg = OmegaConf.create({k: v for k, v in cfg.items() if k != 'ckpt_path'})
            model = instantiate(model_cfg)
            assert model is not None
            print(f"Successfully instantiated model: {model.__class__.__name__}")
            
        except Exception as e:
            print(f"Error instantiating model {model_config}: {e}")
            raise


def test_extract_hparams_from_ckpt():
    """チェックポイントファイルからハイパーパラメータを抽出するテスト"""
    # テスト用のモデル設定
    model_config = "lite_tracknet"
    
    # hydraの初期化
    with initialize(version_base=None, config_path="../../configs/infer/ball"):
        # モデル設定の読み込み
        cfg = compose(config_name=model_config)
        
        try:
            # チェックポイントパスの取得
            ckpt_path = cfg.ckpt_path
            if not os.path.exists(ckpt_path):
                pytest.skip(f"Checkpoint file not found: {ckpt_path}")
            
            # ハイパーパラメータの抽出
            hparams = extract_hparams_from_ckpt(ckpt_path)
            assert hparams is not None
            print(f"Extracted hparams: {hparams}")
            
            # モデル設定の抽出
            model_config = extract_model_config_from_ckpt(ckpt_path)
            assert model_config is not None
            print(f"Extracted model config: {model_config}")
            
        except Exception as e:
            print(f"Error extracting hparams: {e}")
            raise 