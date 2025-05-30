"""
モデル操作に関するユーティリティ関数
"""

import logging
from typing import Dict, Any, Optional

import torch
from hydra.utils import get_class

logger = logging.getLogger(__name__)


def extract_hparams_from_ckpt(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """
    チェックポイントファイルからハイパーパラメータを抽出します。
    Lightning形式のcheckpointからhyper_parametersを取得します。

    Args:
        ckpt_path (str): チェックポイントファイルのパス

    Returns:
        Optional[Dict[str, Any]]: ハイパーパラメータの辞書、または取得できない場合はNone
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # Lightning形式のcheckpointからhyper_parametersを取得
        if "hyper_parameters" in ckpt:
            return ckpt["hyper_parameters"]
        
        # 古い形式のLightningはhparams
        if "hparams" in ckpt:
            return ckpt["hparams"]
            
        logger.warning(f"No hyperparameters found in checkpoint: {ckpt_path}")
        return None
    except Exception as e:
        logger.warning(f"Failed to extract hyperparameters from checkpoint: {e}")
        return None


def extract_model_config_from_ckpt(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """
    チェックポイントファイルからモデル設定パラメータを抽出します。
    特に、in_channels、out_channelsなどのモデル初期化に必要なパラメータを取得します。

    Args:
        ckpt_path (str): チェックポイントファイルのパス

    Returns:
        Optional[Dict[str, Any]]: モデル設定パラメータの辞書、または取得できない場合はNone
    """
    try:
        hparams = extract_hparams_from_ckpt(ckpt_path)
        if not hparams:
            return None
            
        # モデル関連のパラメータのみを抽出
        model_params = {}
        
        # 'model'キーがある場合（LightningModuleでself.model = ...としている場合）
        if isinstance(hparams.get('model'), dict):
            model_params.update(hparams['model'])
        
        # トップレベルのパラメータ（in_channels, out_channelsなど）
        for key in ['in_channels', 'out_channels', 'num_keypoints', 'heatmap_channels']:
            if key in hparams:
                model_params[key] = hparams[key]
                
        # パラメータ名の変換（古いモデルとの互換性のため）
        if 'num_keypoints' in model_params and 'out_channels' not in model_params:
            model_params['out_channels'] = model_params['num_keypoints']
            
        if 'heatmap_channels' in model_params and 'out_channels' not in model_params:
            model_params['out_channels'] = model_params['heatmap_channels']
            
        return model_params
    except Exception as e:
        logger.warning(f"Failed to extract model config from checkpoint: {e}")
        return None


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