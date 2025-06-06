import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional


class EventDetectionModel(nn.Module):
    """
    イベント検出用のLSTMベースモデル。
    ボール、プレイヤーBBox、プレイヤーポーズ、コートの特徴量を入力として、
    ボールのイベントステータスを時系列で予測します。
    """

    def __init__(
        self,
        ball_dim: int = 3,
        player_bbox_dim: int = 5,
        player_pose_dim: int = 51,  # 17キーポイント × 3
        court_dim: int = 31,
        max_players: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,  # イベントステータスのクラス数
        dropout: float = 0.2,
        bidirectional: bool = True,
        rnn_type: str = "lstm",  # "lstm" or "gru"
    ):
        super().__init__()
        
        self.ball_dim = ball_dim
        self.player_bbox_dim = player_bbox_dim
        self.player_pose_dim = player_pose_dim
        self.court_dim = court_dim
        self.max_players = max_players
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        
        # 各特徴の特徴抽出用ネットワーク
        # 1. ボール特徴
        self.ball_encoder = nn.Sequential(
            nn.Linear(ball_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 2. プレイヤーBBox特徴
        self.player_bbox_encoder = nn.Sequential(
            nn.Linear(player_bbox_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 3. プレイヤーポーズ特徴
        self.player_pose_encoder = nn.Sequential(
            nn.Linear(player_pose_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 4. コート特徴
        self.court_encoder = nn.Sequential(
            nn.Linear(court_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 特徴結合後の全結合層
        # 特徴次元の合計: ball(16) + player_bbox(16*max_players) + player_pose(32*max_players) + court(32)
        combined_dim = 16 + 16 * max_players + 32 * max_players + 32
        
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # RNNレイヤー（LSTM or GRU）
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # 出力層
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, 
        ball_features: torch.Tensor, 
        player_bbox_features: torch.Tensor, 
        player_pose_features: torch.Tensor, 
        court_features: torch.Tensor
    ) -> torch.Tensor:
        """
        順伝播処理
        
        Args:
            ball_features: ボール特徴 [batch_size, seq_len, ball_dim]
            player_bbox_features: プレイヤーBBox特徴 [batch_size, seq_len, max_players, player_bbox_dim]
            player_pose_features: プレイヤーポーズ特徴 [batch_size, seq_len, max_players, player_pose_dim]
            court_features: コート特徴 [batch_size, seq_len, court_dim]
            
        Returns:
            torch.Tensor: 各時点でのイベントステータス予測 [batch_size, seq_len, num_classes]
        """
        batch_size, seq_len, _ = ball_features.shape
        
        # 1. ボール特徴のエンコード
        ball_encoded = self.ball_encoder(ball_features)  # [batch_size, seq_len, 16]
        
        # 2. プレイヤーBBox特徴のエンコード
        # [batch_size, seq_len, max_players, player_bbox_dim] → [batch_size, seq_len, max_players, 16]
        batch_size, seq_len, num_players, _ = player_bbox_features.shape
        player_bbox_reshaped = player_bbox_features.reshape(batch_size * seq_len * num_players, -1)
        player_bbox_encoded = self.player_bbox_encoder(player_bbox_reshaped)
        player_bbox_encoded = player_bbox_encoded.reshape(batch_size, seq_len, num_players, -1)
        
        # 3. プレイヤーポーズ特徴のエンコード
        # [batch_size, seq_len, max_players, player_pose_dim] → [batch_size, seq_len, max_players, 32]
        player_pose_reshaped = player_pose_features.reshape(batch_size * seq_len * num_players, -1)
        player_pose_encoded = self.player_pose_encoder(player_pose_reshaped)
        player_pose_encoded = player_pose_encoded.reshape(batch_size, seq_len, num_players, -1)
        
        # 4. コート特徴のエンコード
        # [batch_size, seq_len, court_dim] → [batch_size, seq_len, 32]
        court_encoded = self.court_encoder(court_features)
        
        # 5. プレイヤー特徴を平坦化
        # [batch_size, seq_len, max_players, feat_dim] → [batch_size, seq_len, max_players * feat_dim]
        player_bbox_flat = player_bbox_encoded.reshape(batch_size, seq_len, -1)
        player_pose_flat = player_pose_encoded.reshape(batch_size, seq_len, -1)
        
        # 6. すべての特徴を結合
        # [batch_size, seq_len, combined_dim]
        combined_features = torch.cat([
            ball_encoded,
            player_bbox_flat,
            player_pose_flat,
            court_encoded
        ], dim=2)
        
        # 7. 結合された特徴の処理
        features = self.feature_combiner(combined_features)  # [batch_size, seq_len, hidden_dim]
        
        # 8. RNNで時系列処理
        rnn_out, _ = self.rnn(features)  # [batch_size, seq_len, hidden_dim*2] (双方向の場合)
        
        # 9. 分類
        logits = self.classifier(rnn_out)  # [batch_size, seq_len, num_classes]
        
        return logits

    def predict(
        self, 
        ball_features: torch.Tensor, 
        player_bbox_features: torch.Tensor, 
        player_pose_features: torch.Tensor, 
        court_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        予測処理（確率とクラス）
        
        Args:
            ball_features: ボール特徴 [batch_size, seq_len, ball_dim]
            player_bbox_features: プレイヤーBBox特徴 [batch_size, seq_len, max_players, player_bbox_dim]
            player_pose_features: プレイヤーポーズ特徴 [batch_size, seq_len, max_players, player_pose_dim]
            court_features: コート特徴 [batch_size, seq_len, court_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 予測確率 [batch_size, seq_len, num_classes]
                - 予測クラス [batch_size, seq_len]
        """
        logits = self(ball_features, player_bbox_features, player_pose_features, court_features)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return probs, preds 