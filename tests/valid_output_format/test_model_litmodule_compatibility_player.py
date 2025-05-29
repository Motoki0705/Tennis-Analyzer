"""
プレイヤーモデルとLitModuleの互換性をテストするモジュール
"""

import pytest
from omegaconf import OmegaConf
from scripts.train.train_player import validate_model_litmodule_compatibility


def test_compatible_output_types():
    """互換性のある出力タイプの場合、エラーが発生しないことをテスト"""
    # 同じ出力タイプ（detection）の設定
    model_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_player_model',
            'output_type': 'detection'
        }
    })
    
    litmodule_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_player_litmodule',
            'output_type': 'detection'
        }
    })
    
    # エラーが発生しないことを確認
    validate_model_litmodule_compatibility(model_cfg, litmodule_cfg)


def test_incompatible_output_types():
    """互換性のない出力タイプの場合、ValueErrorが発生することをテスト"""
    # 異なる出力タイプの設定
    model_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_player_model',
            'output_type': 'detection'
        }
    })
    
    litmodule_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_player_litmodule',
            'output_type': 'heatmap'
        }
    })
    
    # ValueErrorが発生することを確認
    with pytest.raises(ValueError) as excinfo:
        validate_model_litmodule_compatibility(model_cfg, litmodule_cfg)
    
    # エラーメッセージに期待する情報が含まれていることを確認
    assert "モデルの出力タイプ 'detection' と LitModuleの出力タイプ 'heatmap' が一致しません" in str(excinfo.value)
    assert "test_player_model" in str(excinfo.value)
    assert "test_player_litmodule" in str(excinfo.value) 