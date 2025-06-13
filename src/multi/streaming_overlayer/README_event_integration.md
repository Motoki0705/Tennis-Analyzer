# Event Model Integration Guide

このガイドでは、Tennis AnalyzerプロジェクトにEventモデルを統合する方法について説明します。

## 概要

Eventモデル統合により、ball、court、poseワーカーの結果を利用して、テニスの試合におけるヒット・バウンドイベントをリアルタイムで検知できます。

## アーキテクチャ

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Ball Worker │  │Court Worker │  │ Pose Worker │
└─────┬───────┘  └─────┬───────┘  └─────┬───────┘
      │                │                │
      │ ball result    │ court result   │ pose result
      │                │                │
      └────────────────┼────────────────┘
                       │
              ┌────────▼────────┐
              │  Event Worker   │
              │                 │
              │ - 結果統合      │
              │ - 時系列管理    │
              │ - 特徴量変換    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Event Predictor │
              │ (transformer_v2)│
              └────────┬────────┘
                       │
                 event result
```

## 実装済みコンポーネント

### 1. EventWorker (`workers/event_worker.py`)

他のワーカーの結果を統合してevent推論を行うワーカー。

**主な機能:**
- 他のワーカー結果の非同期統合
- 時系列シーケンス管理（デフォルト16フレーム）
- 特徴量変換とテンソル作成
- メモリ効率的なバッファ管理

### 2. VideoPredictorWithEvent (`video_predictor_with_event.py`)

Eventワーカーを統合した動画処理クラス。

**主な機能:**
- 4つのワーカー（ball, court, pose, event）の管理
- 基本ワーカーの結果をeventワーカーに自動転送
- 統合オーバーレイ描画
- パフォーマンス監視

### 3. EventUtils (`event_utils.py`)

統合処理のためのユーティリティ関数群。

**主な機能:**
- 座標正規化・逆正規化
- 特徴量抽出と変換
- 結果フォーマッティング

## 使用方法

### 基本的な使用例

```python
from src.multi.streaming_overlayer.video_predictor_with_event import create_video_predictor_with_event
from src.predictors.ball_predictor import BallPredictor
from src.predictors.court_predictor import CourtPredictor  
from src.predictors.pose_predictor import PosePredictor
from src.predictors.event_predictor import EventPredictor

# 各予測器を初期化
ball_predictor = BallPredictor(model_path="path/to/ball/model")
court_predictor = CourtPredictor(model_path="path/to/court/model")
pose_predictor = PosePredictor(model_path="path/to/pose/model")
event_predictor = EventPredictor(litmodule="path/to/event/model")

# VideoPredictorWithEventを作成
video_predictor = create_video_predictor_with_event(
    ball_predictor=ball_predictor,
    court_predictor=court_predictor,
    pose_predictor=pose_predictor,
    event_predictor=event_predictor,
    debug=True,
    event_sequence_length=16  # イベント推論用のシーケンス長
)

# 動画処理実行
video_predictor.run(
    input_path="input_video.mp4",
    output_path="output_with_events.mp4"
)
```

### 設定オプション

```python
# 詳細設定での初期化
video_predictor = VideoPredictorWithEvent(
    ball_predictor=ball_predictor,
    court_predictor=court_predictor,
    pose_predictor=pose_predictor,
    event_predictor=event_predictor,
    
    # 処理間隔（フレーム単位）
    intervals={
        "ball": 1,    # 毎フレーム
        "court": 5,   # 5フレームおき
        "pose": 2,    # 2フレームおき
        "event": 1    # 毎フレーム
    },
    
    # バッチサイズ
    batch_sizes={
        "ball": 1,
        "court": 1, 
        "pose": 1,
        "event": 1
    },
    
    # その他設定
    debug=True,
    enable_performance_monitoring=True,
    event_sequence_length=16,  # イベント推論用シーケンス長
    max_preload_frames=64
)
```

## データフロー

### 1. 基本ワーカーの結果形状

```python
# Ball Worker結果
ball_result = {
    'x': 1644,           # X座標
    'y': 165,            # Y座標  
    'confidence': 0.502  # 信頼度
}

# Court Worker結果 
court_result = [
    {'x': 516, 'y': 852, 'confidence': 0.709},
    {'x': 1236, 'y': 300, 'confidence': 0.708},
    # ... 最大15個のキーポイント
]

# Pose Worker結果
pose_result = [
    {
        'bbox': [763, 519, 98, 158],     # [x, y, w, h]
        'det_score': 0.948,              # 検出信頼度
        'keypoints': [(776, 537), ...],  # 17個のキーポイント
        'scores': [0.448, 0.536, ...]   # キーポイント信頼度
    },
    # ... 最大4名のプレイヤー
]
```

### 2. Event推論用特徴量

EventWorkerで以下の形状に変換されます：

```python
features = {
    'ball_features': torch.Tensor,      # [1, T, 3]
    'court_features': torch.Tensor,     # [1, T, 45] (15点×3)
    'player_bbox_features': torch.Tensor,  # [1, T, P, 5]
    'player_pose_features': torch.Tensor   # [1, T, P, 51] (17点×3)
}
```

### 3. Event推論結果

```python
event_result = {
    'hit_detected': False,              # ヒット検知
    'bounce_detected': True,            # バウンス検知
    'hit_probability': 0.123,           # ヒット確率
    'bounce_probability': 0.867,        # バウンス確率
    'smoothed_hit_signal': 0.098,       # 平滑化ヒット信号
    'smoothed_bounce_signal': 0.834,    # 平滑化バウンス信号
    'signal_history': [...],            # 信号履歴
    'timestamp': 12345                  # タイムスタンプ
}
```

## パフォーマンス監視

```python
# パフォーマンス統計取得
stats = video_predictor.get_performance_metrics()
print(f"処理済みフレーム数: {stats['total_frames_processed']}")
print(f"平均FPS: {stats['frames_per_second']:.2f}")

# ワーカー別統計
for name, worker in video_predictor.workers.items():
    worker_stats = worker.get_performance_stats()
    print(f"{name}: {worker_stats}")
```

## デバッグ

```python
# デバッグモードで実行
video_predictor = create_video_predictor_with_event(
    # ... 予測器設定
    debug=True  # 詳細ログ出力
)

# 特徴量のデバッグ出力
from src.multi.streaming_overlayer.event_utils import debug_print_features

debug_print_features(features, frame_idx=100)
```

## 注意事項

### 1. メモリ使用量

- EventWorkerは時系列データを保持するため、`event_sequence_length`に応じてメモリ使用量が増加します
- 長時間の動画処理では定期的なガベージコレクションを推奨

### 2. レイテンシ

- Event推論には時系列データが必要なため、初期の数フレームは結果が出力されません
- リアルタイム処理では`event_sequence_length`を短くすることを検討

### 3. GPU使用量

- 4つのワーカーが並列実行されるため、GPU メモリ不足に注意
- 必要に応じて`batch_sizes`や`intervals`を調整

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```python
   # バッチサイズを小さくする
   batch_sizes={"ball": 1, "court": 1, "pose": 1, "event": 1}
   ```

2. **処理が遅い**
   ```python
   # 処理間隔を増やす
   intervals={"ball": 2, "court": 5, "pose": 3, "event": 1}
   ```

3. **Event結果が出ない**
   ```python
   # シーケンス長を短くする
   event_sequence_length=8
   ```

## まとめ

この統合により、Tennis Analyzerプロジェクトでリアルタイムなイベント検知が可能になります。各ワーカーの結果を効率的に統合し、テニスの試合における重要なイベントを正確に検知できます。 