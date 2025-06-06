import torch
from .event_model import EventDetectionModel


def create_model(model_type: str = "lstm", **kwargs):
    """
    イベント検出用のモデルを作成します。

    Args:
        model_type (str): モデルタイプ ('lstm', 'gru', 'bilstm', 'bigru')
        **kwargs: モデルの追加パラメータ
            - ball_dim (int): ボール特徴の次元数
            - player_bbox_dim (int): プレイヤーのBBox特徴の次元数
            - player_pose_dim (int): プレイヤーのポーズ特徴の次元数
            - court_dim (int): コート特徴の次元数
            - max_players (int): フレーム内の最大プレイヤー数
            - hidden_dim (int): 隠れ層の次元数
            - num_layers (int): RNNのレイヤー数
            - dropout (float): ドロップアウト率

    Returns:
        nn.Module: 初期化されたモデル
    """
    model_dict = {
        "lstm": {
            "rnn_type": "lstm",
            "bidirectional": False,
            "hidden_dim": 256,
            "num_layers": 2,
            "ball_dim": 3,
            "player_bbox_dim": 5,
            "player_pose_dim": 51,  # 17キーポイント × 3
            "court_dim": 31,
            "max_players": 2,
        },
        "gru": {
            "rnn_type": "gru",
            "bidirectional": False,
            "hidden_dim": 256,
            "num_layers": 2,
            "ball_dim": 3,
            "player_bbox_dim": 5,
            "player_pose_dim": 51,
            "court_dim": 31,
            "max_players": 2,
        },
        "bilstm": {
            "rnn_type": "lstm",
            "bidirectional": True,
            "hidden_dim": 256,
            "num_layers": 2,
            "ball_dim": 3,
            "player_bbox_dim": 5,
            "player_pose_dim": 51,
            "court_dim": 31,
            "max_players": 2,
        },
        "bigru": {
            "rnn_type": "gru",
            "bidirectional": True,
            "hidden_dim": 256,
            "num_layers": 2,
            "ball_dim": 3,
            "player_bbox_dim": 5,
            "player_pose_dim": 51,
            "court_dim": 31,
            "max_players": 2,
        },
    }
    
    if model_type not in model_dict:
        raise ValueError(f"不明なモデルタイプ: {model_type}。利用可能なタイプ: {list(model_dict.keys())}")
    
    # デフォルト設定をコピーして、カスタム引数で上書き
    model_params = model_dict[model_type].copy()
    model_params.update(kwargs)
    
    # モデルを作成して返す
    model = EventDetectionModel(**model_params)
    return model 