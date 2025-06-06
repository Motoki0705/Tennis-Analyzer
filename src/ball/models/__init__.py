import torch
from .swin_448 import SwinBallUNet


def create_model(model_type, pretrained=True, **kwargs):
    """
    ボール検出用のモデルを作成します。

    Args:
        model_type (str): モデルタイプ ('swin_tiny', 'swin_small', 'swin_base')
        pretrained (bool): 事前学習済みの重みを使用するかどうか

    Returns:
        nn.Module: 初期化されたモデル
    """
    model_dict = {
        "swin_tiny": {
            "swin_model": "swin_tiny_patch4_window7_224", 
            "in_channels": 9, 
            "out_channels": 1,
            "final_channels": [64, 32]
        },
        "swin_small": {
            "swin_model": "swin_small_patch4_window7_224", 
            "in_channels": 9, 
            "out_channels": 1,
            "final_channels": [96, 48]
        },
        "swin_base": {
            "swin_model": "swin_base_patch4_window7_224", 
            "in_channels": 9, 
            "out_channels": 1,
            "final_channels": [128, 64]
        },
    }
    
    if model_type not in model_dict:
        raise ValueError(f"不明なモデルタイプ: {model_type}。利用可能なタイプ: {list(model_dict.keys())}")
    
    # デフォルト設定をコピーして、カスタム引数で上書き
    model_params = model_dict[model_type].copy()
    model_params.update(kwargs)
    model_params["pretrained"] = pretrained
    
    # モデルを作成して返す
    model = SwinBallUNet(**model_params)
    return model
