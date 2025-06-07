"""
イベント検出用LightningModuleパッケージ
"""
from src.event.lit_module.event_detection_lit_module import EventDetectionLitModule


def create_lit_module(model_type: str = "transformer_v2", **kwargs):
    """
    イベント検出用のLightningModuleを作成します。
    
    Args:
        model_type (str): モデルタイプ ('transformer_v2')
        **kwargs: LitModuleの追加パラメータ
            - d_model (int): モデルの次元数
            - nhead (int): Transformerのヘッド数
            - num_layers (int): Transformerのレイヤー数
            - dropout (float): ドロップアウト率
            - max_seq_len (int): 最大シーケンス長
            - pose_dim (int): ポーズ特徴の次元数
            - lr (float): 学習率
            - weight_decay (float): 重み減衰
            - warmup_epochs (int): ウォームアップエポック数
            - max_epochs (int): 最大エポック数
            - no_hit_weight (float): no_hit(0,0)の重み
            - hit_weight (float): hit(1,0)の重み
            - bounce_weight (float): bounce(0,1)の重み
            - clarity_weight (float): 明確な予測を促進する重み
    
    Returns:
        pl.LightningModule: 初期化されたLightningModule
    """
    model_configs = {
        "transformer_v2": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dropout": 0.1,
            "max_seq_len": 512,
            "pose_dim": 51,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 5,
            "max_epochs": 100,
            "no_hit_weight": 0.01,
            "hit_weight": 1.0,
            "bounce_weight": 1.0,
            "clarity_weight": 0.02,
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"不明なモデルタイプ: {model_type}。利用可能なタイプ: {list(model_configs.keys())}")
    
    # デフォルト設定をコピーして、カスタム引数で上書き
    config = model_configs[model_type].copy()
    config.update(kwargs)
    
    # LitModuleを作成して返す
    lit_module = EventDetectionLitModule(**config)
    return lit_module 