import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from src.event.models import create_model
from src.utils.event_status_dict import event_status_dict


class EventPredictor:
    """
    イベント検出用予測クラス
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            checkpoint_path: モデルのチェックポイントパス
            device: 推論デバイス
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # チェックポイントからモデルをロード
        self.load_model()
        
        # 逆引きマッピング（予測結果を文字列に変換するため）
        self.id_to_status = {v: k for k, v in event_status_dict.items()}

    def load_model(self):
        """モデルをロードします"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # ハイパーパラメータの取得
        hparams = checkpoint.get("hyper_parameters", {})
        
        # モデルの作成
        model_type = hparams.get("model_type", "bilstm")
        input_dim = hparams.get("input_dim", 148)  # デフォルト値
        hidden_dim = hparams.get("hidden_dim", 256)
        num_layers = hparams.get("num_layers", 2)
        num_classes = hparams.get("num_classes", 4)
        
        self.model = create_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )
        
        # 重みの読み込み
        state_dict = checkpoint.get("state_dict", {})
        
        # LightningModule内のモデルの場合、接頭辞を削除
        if all(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        features: torch.Tensor,
    ) -> Tuple[List[str], np.ndarray]:
        """
        時系列特徴量からイベントを予測します
        
        Args:
            features: 特徴量テンソル [batch_size, seq_len, input_dim]
                      または [seq_len, input_dim]
        
        Returns:
            Tuple[List[str], np.ndarray]: 
                - 予測イベントの文字列リスト
                - 予測確率のnumpy配列 [seq_len, num_classes]
        """
        # 入力の次元チェック
        if features.dim() == 2:
            # [seq_len, input_dim] → [1, seq_len, input_dim]
            features = features.unsqueeze(0)
        
        # デバイス変換
        features = features.to(self.device)
        
        # 推論モード
        with torch.no_grad():
            # 予測
            logits = self.model(features)  # [batch_size, seq_len, num_classes]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)  # [batch_size, seq_len]
        
        # バッチサイズ1を想定
        probs = probs[0].cpu().numpy()  # [seq_len, num_classes]
        preds = preds[0].cpu().numpy()  # [seq_len]
        
        # 予測クラスを文字列に変換
        pred_events = [self.id_to_status.get(int(p), "unknown") for p in preds]
        
        return pred_events, probs
