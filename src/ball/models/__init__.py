# __init__.py

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Type, Any

# この関数を外部から import して利用します
__all__ = ['load_model_from_checkpoint']


def load_model_from_checkpoint(
    model_class: Type[nn.Module], 
    checkpoint_path: str,
    map_location: str = 'cpu',
    prefix: str = 'model.',
    **kwargs: Any
) -> nn.Module:
    """
    PyTorch Lightningのチェックポイントからモデルをロードするファクトリー関数。

    state_dictのキーに特定のプレフィックスが存在するかを動的にチェックし、
    存在すれば除去してからモデルにウェイトを読み込みます。

    Args:
        model_class (Type[nn.Module]): ロードするモデルのクラス定義。
        checkpoint_path (str): .ckptファイルのパス。
        map_location (str, optional): ウェイトをロードするデバイス。'cpu'が安全です。
        prefix (str, optional): 除去を試みるプレフィックス。デフォルトは 'model.'。
        **kwargs (Any): model_classのコンストラクタに渡す追加の引数。

    Returns:
        nn.Module: ウェイトがロードされたモデルのインスタンス。
    """
    # 1. チェックポイントを読み込む
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as e:
        print(f"❌ チェックポイントファイルの読み込みに失敗しました: {checkpoint_path}")
        raise e

    # 2. state_dict を取り出す
    if 'state_dict' not in checkpoint:
        raise KeyError(
            "チェックポイントに 'state_dict' が見つかりません。"
            "PyTorch Lightningのチェックポイントではない可能性があります。"
        )
    state_dict = checkpoint['state_dict']

    # 3. プレフィックスの動的なチェックと除去
    # 全てのキーが指定されたプレフィックスで始まるかを確認
    has_prefix = all(key.startswith(prefix) for key in state_dict.keys())
    
    if has_prefix:
        print(f"✅ プレフィックス '{prefix}' を検出しました。キーから除去します。")
        # プレフィックスを除去した新しいstate_dictを作成
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace(prefix, '', 1)  # プレフィックスを除去
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        print("ℹ️ プレフィックスは検出されませんでした。キーはそのまま使用します。")

    # 4. モデルをインスタンス化し、ウェイトをロード
    model = model_class(**kwargs)
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ モデル '{model_class.__name__}' へのウェイトのロードが完了しました。")
    except RuntimeError as e:
        print(f"❌ ウェイトのロード中にエラーが発生しました。モデルの構造が一致しない可能性があります。")
        print("strict=Falseで試す場合は、コードを修正してください。")
        raise e

    # 推論モードに設定して返すのが一般的
    model.eval()
    
    return model

if __name__ == "__main__":
    # 例として、HRNetモデルをロードする場合
    from src.ball.models.conv3d_tsm_fpn import BallDetNet  # HRNetの定義をインポート

    # チェックポイントのパスとモデルのクラスを指定
    checkpoint_path = 'checkpoints/ball/conv3d_tsm_fpn/best_model.ckpt'
    
    # モデルをロード
    model = load_model_from_checkpoint(BallDetNet, checkpoint_path, prefix='model.')
    
    print("モデルのロードが完了しました。")