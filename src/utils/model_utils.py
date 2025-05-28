"""
モデル操作に関するユーティリティ関数
"""

import torch
from hydra.utils import get_class


def load_model_weights(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    """
    PyTorch Lightningで保存されたcheckpointファイルから
    'model.'プレフィックスを除去して、通常のPyTorchモデルにロードします。

    Args:
        model (torch.nn.Module): ロード対象のモデル
        ckpt_path (str): checkpointファイルのパス

    Returns:
        torch.nn.Module: ロード後のモデル
    """
    # ckptをロード
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # state_dictを取得
    state_dict = ckpt["state_dict"]

    # "model."というprefixを除去
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[len("model.") :]  # "model."を取り除く
        else:
            new_key = k  # そのまま
        new_state_dict[new_key] = v

    # モデルにロード
    model.load_state_dict(new_state_dict, strict=True)
    return model


def create_model_with_hf_load(
    model_class_name: str,
    pretrained_model_name_or_path: str,
    ckpt_path: str = None,
    **model_args,
):
    """
    Hugging Faceモデルをfrom_pretrainedを使って初期化し、オプションでカスタムの重みをロードします。

    Args:
        model_class_name (str): モデルクラスのパス（例: "transformers.ConditionalDetrForObjectDetection"）
        pretrained_model_name_or_path (str): 事前学習済みモデルの名前またはパス
        ckpt_path (str, optional): カスタムチェックポイントのパス
        **model_args: モデル初期化のための追加引数

    Returns:
        モデルインスタンス
    """
    model_class = get_class(model_class_name)

    # ほとんどのHFモデルはfrom_pretrainedでロードする
    if hasattr(model_class, "from_pretrained"):
        model = model_class.from_pretrained(pretrained_model_name_or_path, **model_args)
    else:  # 一般的なPyTorchモデル用のフォールバック
        model = model_class(**model_args)

    if ckpt_path:
        # これはload_model_weightsがPyTorch/Lightningチェックポイント用であることを前提としています
        model = load_model_weights(model, ckpt_path)
    return model 