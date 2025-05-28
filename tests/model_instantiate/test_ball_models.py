"""
ballモデルのインスタンス化テスト
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
BALL_MODEL_CONFIGS = [
    # cat_frames
    "cat_frames/lite_tracknet",
    "cat_frames/lite_tracknet_xy",
    "cat_frames/tracknet",
    "cat_frames/xception",
    "cat_frames/swin_448",
    "cat_frames/resnet_regression",
    # sequence
    "sequence/lstm_unet",
    "sequence/mobile_gru_unet",
    "sequence/mobile_trans",
    "sequence/uniformer",
    "sequence/tsformer_ball",
    "sequence/unet_3d",
    # single_frame
    "single_frame/mobilenet",
]


@pytest.mark.parametrize("model_config", BALL_MODEL_CONFIGS)
def test_ball_model_instantiate(model_config):
    """
    ballモデルが正しくインスタンス化できることをテストします
    """
    # モデル設定ファイルのパスを分割
    parts = model_config.split("/")
    model_type = parts[0]  # cat_frames, sequence, single_frame
    model_name = parts[1]  # 具体的なモデル名
    
    # hydraの初期化
    with initialize(version_base=None, config_path=f"../../configs/train/ball/model/{model_type}"):
        # litdatamodule.Tの値を設定するための構成を作成
        T = 9  # デフォルト値
        defaults = [{f"litdatamodule": {"T": T}}]
        
        # モデル設定の読み込み
        cfg = compose(config_name=model_name, overrides=[f"+litdatamodule.T={T}"])
        
        print(f"\nTesting model: {model_config}")
        print(f"Model config: {OmegaConf.to_yaml(cfg)}")
        
        try:    
            # モデルのインスタンス化
            model = instantiate(cfg.model.net)
            assert True  # テスト成功
            
        except Exception as e:
            print(f"Error instantiating model {model_config}: {e}")
            raise 