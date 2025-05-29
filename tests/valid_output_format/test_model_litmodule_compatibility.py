"""
モデルとLitModuleの互換性をテストするモジュール
"""

import pytest
from omegaconf import OmegaConf
from scripts.train.train_ball import validate_model_litmodule_compatibility


def test_compatible_output_types():
    """互換性のある出力タイプの場合、エラーが発生しないことをテスト"""
    # 同じ出力タイプ（heatmap）の設定
    model_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_heatmap_model',
            'output_type': 'heatmap'
        }
    })
    
    litmodule_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_heatmap_litmodule',
            'output_type': 'heatmap'
        }
    })
    
    # エラーが発生しないことを確認
    validate_model_litmodule_compatibility(model_cfg, litmodule_cfg)
    
    # 同じ出力タイプ（coord）の設定
    model_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_coord_model',
            'output_type': 'coord'
        }
    })
    
    litmodule_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_coord_litmodule',
            'output_type': 'coord'
        }
    })
    
    # エラーが発生しないことを確認
    validate_model_litmodule_compatibility(model_cfg, litmodule_cfg)


def test_incompatible_output_types():
    """互換性のない出力タイプの場合、ValueErrorが発生することをテスト"""
    # 異なる出力タイプの設定
    model_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_heatmap_model',
            'output_type': 'heatmap'
        }
    })
    
    litmodule_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_coord_litmodule',
            'output_type': 'coord'
        }
    })
    
    # ValueErrorが発生することを確認
    with pytest.raises(ValueError) as excinfo:
        validate_model_litmodule_compatibility(model_cfg, litmodule_cfg)
    
    # エラーメッセージに期待する情報が含まれていることを確認
    assert "モデルの出力タイプ 'heatmap' と LitModuleの出力タイプ 'coord' が一致しません" in str(excinfo.value)
    assert "test_heatmap_model" in str(excinfo.value)
    assert "test_coord_litmodule" in str(excinfo.value)
    
    # 逆のパターンでもテスト
    model_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_coord_model',
            'output_type': 'coord'
        }
    })
    
    litmodule_cfg = OmegaConf.create({
        'meta': {
            'name': 'test_heatmap_litmodule',
            'output_type': 'heatmap'
        }
    })
    
    with pytest.raises(ValueError) as excinfo:
        validate_model_litmodule_compatibility(model_cfg, litmodule_cfg)
    
    assert "モデルの出力タイプ 'coord' と LitModuleの出力タイプ 'heatmap' が一致しません" in str(excinfo.value) 