#!/usr/bin/env python
"""
トレーニングスクリプトの設定読み込みテスト

使用例:
    python tests/train_script_test.py
"""

import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pytest
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from hydra.utils import instantiate


def test_ball_config_loading():
    """
    ボール検出用設定ファイルの読み込みテスト
    """
    config_dir = project_root / "configs" / "train" / "ball"
    
    try:
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.3"):
            cfg = compose(config_name="config")
            
            # 基本的な設定項目の存在確認
            assert "litmodule" in cfg
            assert "litdatamodule" in cfg
            assert "trainer" in cfg
            assert "callbacks" in cfg
            assert "seed" in cfg
            
            # LightningModuleのインスタンス化テスト
            lit_module = instantiate(cfg.litmodule.module)
            assert lit_module is not None
            
            # DataModuleのインスタンス化テスト
            datamodule = instantiate(cfg.litdatamodule)
            assert datamodule is not None
            
            print("✓ Ball config loading test passed")
            
    except Exception as e:
        pytest.fail(f"Ball config loading failed: {e}")


def test_court_config_loading():
    """
    コート検出用設定ファイルの読み込みテスト
    """
    config_dir = project_root / "configs" / "train" / "court"
    
    try:
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.3"):
            cfg = compose(config_name="config")
            
            # 基本的な設定項目の存在確認
            assert "litmodule" in cfg
            assert "litdatamodule" in cfg
            assert "trainer" in cfg
            assert "callbacks" in cfg
            assert "seed" in cfg
            
            # LightningModuleのインスタンス化テスト
            lit_module = instantiate(cfg.litmodule.module)
            assert lit_module is not None
            
            # DataModuleのインスタンス化テスト
            datamodule = instantiate(cfg.litdatamodule)
            assert datamodule is not None
            
            print("✓ Court config loading test passed")
            
    except Exception as e:
        pytest.fail(f"Court config loading failed: {e}")


def test_callbacks_instantiation():
    """
    コールバックのインスタンス化テスト
    """
    config_dir = project_root / "configs" / "train" / "ball"
    
    try:
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.3"):
            cfg = compose(config_name="config")
            
            # コールバックのインスタンス化テスト
            callbacks = []
            if "callbacks" in cfg:
                for callback_name, cb_conf in cfg.callbacks.items():
                    if "_target_" in cb_conf:
                        callback = instantiate(cb_conf)
                        callbacks.append(callback)
                        print(f"✓ Callback instantiated: {callback_name}")
            
            assert len(callbacks) > 0, "No callbacks were instantiated"
            print(f"✓ Total {len(callbacks)} callbacks instantiated successfully")
            
    except Exception as e:
        pytest.fail(f"Callbacks instantiation failed: {e}")


if __name__ == "__main__":
    print("Running training script configuration tests...")
    
    try:
        test_ball_config_loading()
        test_court_config_loading()
        test_callbacks_instantiation()
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1) 