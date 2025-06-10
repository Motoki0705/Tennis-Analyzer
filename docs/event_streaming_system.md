# イベント検知ストリーミングシステム

## 概要

このシステムは、テニス動画からリアルタイムでバウンドとショットイベントを検知し、動画左上に電気信号のような波形を表示するストリーミング処理システムです。

## システム構成

### 主要コンポーネント

1. **EventWorker** (`src/multi/streaming_overlayer/workers/event_worker.py`)
   - 各タスクの特徴量を統合してイベント検知を実行
   - 時系列データのバッファリングと管理

2. **BallWorker** (`src/multi/streaming_overlayer/workers/ball_worker.py`)
   - ボール検知の並列処理
   - ボール位置と信頼度の検知

3. **PoseWorker** (`src/multi/streaming_overlayer/workers/pose_worker.py`)
   - プレイヤーポーズ検知の並列処理
   - バウンディングボックスとキーポイントの検知

4. **EventPredictor** (`src/event/api/event_predictor.py`)
   - LitTransformerV2を使用したイベント検知API
   - 信号波の描画とオーバーレイ機能

5. **MultiEventPredictor** (`src/multi/streaming_overlayer/multi_event_predictor.py`)
   - 複数タスクの並列実行と統合管理
   - ストリーミング処理のメインコントローラー

### アーキテクチャ

```
[動画入力] → [フレーム分散] → [Ball/Court/Pose Workers] → [特徴量統合] → [Event Worker] → [信号波描画] → [動画出力]
```

## 特徴

- **リアルタイム処理**: 複数タスクの並列実行によるストリーミング処理
- **信号波表示**: バウンドとショットの確率を電気信号風の波形で可視化
- **適応的平滑化**: 移動平均による信号ノイズの除去
- **モジュラー設計**: 各コンポーネントの独立性と拡張性

## 使用方法

### 1. 設定ファイルの準備

`configs/infer/event/config.yaml` を編集：

```yaml
# 入力動画設定
input_video: "samples/input_video.mp4"
output_video: "outputs/event_detection_output.mp4"

# イベント検知モデル設定
event_predictor:
  checkpoint_path: "checkpoints/event/best_transformer_v2.ckpt"
  device: "cpu"  # "cuda" if GPU available
  confidence_threshold: 0.5
  smoothing_window: 5

# 処理間隔とバッチサイズ
intervals:
  ball: 1      # 全フレーム処理
  court: 30    # 30フレームごと
  pose: 1      # 全フレーム処理

batch_sizes:
  ball: 4
  court: 1
  pose: 2
  event: 1

# イベント検知用シーケンス長
event_sequence_length: 16
```

### 2. 推論の実行

```bash
python scripts/infer/infer_event_streaming.py
```

### 3. 出力

- 動画左上に2つの信号波が表示されます：
  - **黄色の波**: ショット（Hit）の確率
  - **マゼンタの波**: バウンド（Bounce）の確率
- イベント検知時には赤い点で強調表示

## 技術仕様

### 入力特徴量

1. **Ball Features** (3次元)
   - x, y座標（正規化済み）
   - 信頼度スコア

2. **Player BBox Features** (5次元 × プレイヤー数)
   - バウンディングボックス座標 (x1, y1, x2, y2)
   - 信頼度スコア

3. **Player Pose Features** (51次元 × プレイヤー数)
   - 17キーポイント × 3 (x, y, visibility)

4. **Court Features** (45次元)
   - 15キーポイント × 3 (x, y, visibility)

### 出力

- **Hit Probability**: ショットイベントの確率 (0-1)
- **Bounce Probability**: バウンドイベントの確率 (0-1)
- **Event Detection**: 閾値を超えた場合のバイナリ検知結果

### パフォーマンス

- **処理速度**: リアルタイム処理対応（30fps）
- **メモリ使用量**: シーケンス長とバッチサイズに依存
- **GPU対応**: CUDA利用可能

## 設定パラメータ

### EventPredictor

- `confidence_threshold`: イベント検知の閾値 (デフォルト: 0.5)
- `smoothing_window`: 信号平滑化のウィンドウサイズ (デフォルト: 5)
- `max_history_length`: 信号履歴の最大保持数 (デフォルト: 60)

### EventWorker

- `sequence_length`: 時系列入力の長さ (デフォルト: 16)
- バッファサイズ: `sequence_length * 2`

### 描画設定

- 信号波表示領域: 300×100ピクセル
- 更新頻度: フレームレート同期
- 色設定:
  - Hit: 黄色 (0, 255, 255)
  - Bounce: マゼンタ (255, 0, 255)
  - 検知: 赤色 (0, 0, 255)

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   - バッチサイズを削減
   - シーケンス長を短縮

2. **処理速度の低下**
   - GPU使用を検討
   - 処理間隔を調整

3. **信号波が表示されない**
   - チェックポイントファイルの確認
   - 入力特徴量の形式確認

### デバッグモード

```yaml
debug: true
```

詳細なログとエラー情報が出力されます。

## 拡張性

### 新しいイベントタイプの追加

1. `LitTransformerV2`の出力次元を変更
2. `EventPredictor.postprocess`を拡張
3. 描画ロジックに新しい信号波を追加

### 新しい特徴量の追加

1. `EventWorker._create_sequence_tensor`を拡張
2. `EventTransformerV2`の入力層を調整
3. データローダーの対応

## テスト

```bash
# EventPredictorのテスト
python -m pytest tests/infer_model_instantiate/test_event_predictor.py -v

# Worker クラスのテスト
python -m pytest tests/infer_model_instantiate/test_workers.py -v

# 統合テスト
python scripts/infer/infer_event_streaming.py --config-name=test_config
```

## 依存関係

- PyTorch Lightning
- OpenCV
- NumPy
- Hydra
- tqdm

## ライセンス

プロジェクトのライセンスに従います。 