"""
courtモデルのインスタンス化テスト
"""
import os
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

# テスト対象のモデル設定ファイル
COURT_MODEL_CONFIGS = [
    "fpn",
    "lite_tracknet",
    "swin_384_court",
    "swin_v2_256_court",
    "vit_court",
]


@pytest.mark.parametrize("model_config", COURT_MODEL_CONFIGS)
def test_court_model_instantiate(model_config):
    """
    courtモデルが正しくインスタンス化できることをテストします
    """
    # hydraの初期化
    with initialize(version_base=None, config_path="../../configs/train/court/model"):
        # litdatamodule.Tの値を設定するための構成を作成
        T = 9  # デフォルト値
        defaults = [{f"litdatamodule": {"T": T}}]
        
        # モデル設定の読み込み
        cfg = compose(config_name=model_config, overrides=[f"+litdatamodule.T={T}"])
        
        print(f"\nTesting model: {model_config}")
        print(f"Model config: {OmegaConf.to_yaml(cfg)}")
        
        try:
            # モデルのインスタンス化
            model = instantiate(cfg.model.net)
            assert True  # テスト成功
       
        except Exception as e:
            print(f"Error instantiating model {model_config}: {e}")
            raise 