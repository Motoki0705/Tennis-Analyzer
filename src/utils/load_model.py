import torch


def load_model_weights(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    """
    PyTorch Lightningで保存されたcheckpointファイルから
    'model.'プレフィックスを除去して、通常のPyTorchモデルにロードする関数。

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
