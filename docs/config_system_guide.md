# Tennis Analyzer キュー設定システム ガイド

## 概要

Tennis Analyzerは柔軟で拡張可能なキューシステムを採用し、Hydra設定フレームワークと統合された設定管理を提供します。このガイドでは、キュー設定システムの使用方法と設定オプションについて説明します。

## 設定ファイル構造

### 基本ディレクトリ構造

```
configs/
├── infer/
│   ├── infer.yaml          # メイン推論設定
│   └── queue/              # キューシステム設定
│       ├── default.yaml        # デフォルト設定
│       ├── high_performance.yaml  # 高性能設定
│       ├── low_memory.yaml     # 軽量設定
│       └── custom_example.yaml # カスタム設定例
```

## 利用可能な設定

### 1. デフォルト設定 (`default.yaml`)

標準的なストリーミング処理用の基本設定です。

```yaml
# 基本キューサイズ設定
base_queue_sizes:
  preprocess: 16
  inference: 16
  postprocess: 16
  results: 100

# ワーカー別拡張キュー設定
worker_extended_queues:
  ball:
    ball_inference: 32
  court:
    court_inference: 32
  pose:
    detection_inference: 32
    detection_postprocess: 32
    pose_inference: 32
    pose_postprocess: 32
```

**用途**: 標準的な使用、開発・テスト環境

### 2. 高性能設定 (`high_performance.yaml`)

GPU使用率最大化とスループット重視の設定です。

```yaml
# 高性能用キューサイズ設定（大容量）
base_queue_sizes:
  preprocess: 32
  inference: 64
  postprocess: 32
  results: 200

# 優先度キュー活用
queue_types:
  inference: "PriorityQueue"
  detection_inference: "PriorityQueue"
  pose_inference: "PriorityQueue"
```

**用途**: 本番環境、高性能GPU搭載システム

### 3. 軽量設定 (`low_memory.yaml`)

メモリ使用量を最小限に抑えた設定です。

```yaml
# 軽量キューサイズ設定（小容量）
base_queue_sizes:
  preprocess: 8
  inference: 8
  postprocess: 8
  results: 32

# 軽量設定
performance:
  enable_monitoring: false
  memory_optimization: true
```

**用途**: 限られたメモリ環境、エッジデバイス

### 4. カスタム設定 (`custom_example.yaml`)

プロジェクト固有のニーズに合わせた設定例です。

```yaml
# 特殊用途キューの追加
custom_queues:
  emergency_processing:
    maxsize: 8
    queue_type: "LifoQueue"
    description: "緊急処理用LIFO キュー"
  
  priority_inference:
    maxsize: 128
    queue_type: "PriorityQueue"
    description: "高優先度推論専用キュー"
```

**用途**: カスタマイズされた処理パイプライン

## 設定の使用方法

### 1. 基本的な使用

`configs/infer/infer.yaml` で設定を指定：

```yaml
defaults:
  - queue: default  # または high_performance, low_memory, custom_example
```

### 2. プログラムでの使用

```python
from omegaconf import OmegaConf
from src.multi.streaming_overlayer.video_predictor import VideoPredictor

# 設定読み込み
queue_config = OmegaConf.load("configs/infer/queue/high_performance.yaml")

# VideoPredictor初期化
video_predictor = VideoPredictor(
    ball_predictor=ball_pred,
    court_predictor=court_pred,
    pose_predictor=pose_pred,
    intervals={"ball": 1, "court": 30, "pose": 5},
    batch_sizes={"ball": 16, "court": 16, "pose": 16},
    hydra_queue_config=queue_config  # Hydra設定を渡す
)
```

### 3. 実行時設定切り替え

```bash
# デフォルト設定で実行
python main.py mode=streaming_overlayer

# 高性能設定で実行
python main.py mode=streaming_overlayer queue=high_performance

# 軽量設定で実行
python main.py mode=streaming_overlayer queue=low_memory
```

## 設定オプション詳細

### キューサイズ設定

| 設定項目 | 説明 | デフォルト値 |
|---------|------|------------|
| `preprocess` | 前処理キューサイズ | 16 |
| `inference` | 推論キューサイズ | 16 |
| `postprocess` | 後処理キューサイズ | 16 |
| `results` | 結果集約キューサイズ | 100 |

### キュータイプ

| タイプ | 説明 | 用途 |
|--------|------|------|
| `Queue` | 標準FIFO キュー | 通常の処理 |
| `PriorityQueue` | 優先度付きキュー | 重要度による処理順制御 |
| `LifoQueue` | LIFO キュー | 最新データ優先処理 |

### パフォーマンス設定

| 設定項目 | 説明 | 影響 |
|---------|------|------|
| `enable_monitoring` | 監視機能有効化 | リアルタイム状態確認 |
| `log_queue_status` | キュー状態ログ出力 | デバッグ・分析 |
| `auto_clear_on_shutdown` | 終了時自動クリア | メモリ効率 |
| `gpu_optimization` | GPU最適化モード | 高性能処理 |

## カスタム設定の作成

### 1. 新しい設定ファイルの作成

```yaml
# configs/infer/queue/my_custom.yaml
defaults:
  - default

# カスタムキューサイズ
base_queue_sizes:
  preprocess: 24
  inference: 48
  postprocess: 24
  results: 150

# 特殊キューの定義
custom_queues:
  my_special_queue:
    maxsize: 64
    queue_type: "Queue"
    description: "特殊処理用キュー"

# パフォーマンス調整
performance:
  enable_monitoring: true
  custom_optimization: true
```

### 2. ワーカー拡張キューの設定

```yaml
worker_extended_queues:
  pose:
    detection_inference: 64      # Detection 推論キュー
    detection_postprocess: 32    # Detection 後処理キュー
    pose_inference: 64           # Pose 推論キュー
    pose_postprocess: 32         # Pose 後処理キュー
    my_special_queue: 64         # カスタムキュー
```

## デバッグ・監視

### 1. キュー状態の確認

```python
# キュー状態取得
status = video_predictor.get_queue_status_with_settings()
print(f"Results queue: {status['results_queue_size']} items")

for worker, info in status['workers'].items():
    base_total = sum(info['base_queues'].values())
    extended_total = sum(info['extended_queues'].values())
    print(f"{worker}: {base_total} base + {extended_total} extended")
```

### 2. 設定検証

```python
from src.multi.streaming_overlayer.config_utils import validate_queue_config

if validate_queue_config(queue_config):
    print("✅ 設定は有効です")
else:
    print("❌ 設定に問題があります")
```

## パフォーマンス チューニング

### 1. 高スループット向け設定

```yaml
base_queue_sizes:
  preprocess: 32    # 大きなバッファ
  inference: 128    # GPU並列処理対応
  postprocess: 32   # 後処理バッファ
  results: 200      # 結果蓄積

queue_types:
  inference: "PriorityQueue"  # 優先度制御
```

### 2. 低レイテンシ向け設定

```yaml
base_queue_sizes:
  preprocess: 8     # 小さなバッファ
  inference: 16     # 即座に処理
  postprocess: 8    # 高速転送
  results: 32       # 迅速な出力

performance:
  minimal_threads: true
```

### 3. メモリ効率向け設定

```yaml
base_queue_sizes:
  preprocess: 4
  inference: 8
  postprocess: 4
  results: 16

performance:
  memory_optimization: true
  enable_monitoring: false
```

## トラブルシューティング

### よくある問題と解決方法

1. **キュー溢れエラー**
   - キューサイズを増加
   - 処理速度の改善
   - バッチサイズの調整

2. **メモリ不足**
   - `low_memory` 設定を使用
   - キューサイズの削減
   - 監視機能の無効化

3. **パフォーマンス不足**
   - `high_performance` 設定を使用
   - PriorityQueue の活用
   - GPU最適化の有効化

4. **設定エラー**
   - `validate_queue_config()` で検証
   - ログ出力で詳細確認
   - デフォルト設定へのフォールバック

## まとめ

Tennis Analyzer のキュー設定システムは：

- **柔軟性**: 用途に応じた設定切り替え
- **拡張性**: カスタムキューの追加対応
- **監視性**: リアルタイム状態確認
- **堅牢性**: エラー時の自動フォールバック

これらの機能により、様々な環境やユースケースに対応した最適なパフォーマンスを実現できます。 