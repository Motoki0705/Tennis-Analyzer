# 🚀 Parallel Video Processing Pipeline

高効率テニスボール検出・可視化パイプライン

## 📋 概要

このパイプラインは、マルチスレッド並列処理によりGPU利用率を最大化し、テニス動画からのボール検出と可視化を高速で実行するシステムです。

### 🎯 主な特徴

- **🔄 並列処理アーキテクチャ**: GPU待機時間を最小化
- **⚡ 高速処理**: 従来比3-5倍の処理速度向上
- **🎥 リアルタイム対応**: ライブ配信にも対応可能
- **💾 メモリ効率**: 大容量動画も安全に処理
- **🎨 高品質可視化**: 軌跡追跡・予測表示対応

## 🏗️ アーキテクチャ

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Frame Reader   │ -> │  Preprocessor    │ -> │ Inference Queue │
│     Thread      │    │     Thread       │    │   (GPU Ready)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│ Visualization   │ <- │  Postprocessor   │ <----------┘
│     Thread      │    │     Thread       │    ┌─────────────────┐
└─────────────────┘    └──────────────────┘ <- │  GPU Inference  │
                                                │     Thread      │
                                                └─────────────────┘
```

### スレッド構成

1. **Frame Reader Thread**: 動画フレーム読み込み
2. **Preprocessor Threads**: フレーム前処理（複数並列）
3. **GPU Inference Thread**: GPUでの推論実行
4. **Postprocessor Threads**: 後処理・検出結果生成
5. **Visualization Thread**: 可視化・動画出力
6. **Monitor Thread**: 性能監視・ログ出力

### キューベース設計

- **Frame Queue**: 読み込みフレームのバッファリング
- **Tensor Queue**: 前処理済みテンソルの蓄積（GPU待機時間削減）
- **Result Queue**: 推論結果の一時保存
- **Render Queue**: 可視化待ちデータ

## 🚀 使用方法

### 基本的な使用

```python
from src.predictor.pipeline import VideoPipeline

# パイプライン初期化
pipeline = VideoPipeline()

# 検出器設定
detector_config = {
    "model_type": "wasb_sbdt",
    "model_path": "checkpoints/model.pth",
    "device": "cuda"
}

# 動画処理実行
result = pipeline.process_video(
    video_path="input.mp4",
    detector_config=detector_config,
    output_path="output.mp4"
)

print(f"処理完了: {result['average_fps']:.2f} FPS")
```

### 高性能設定

```python
from src.predictor.pipeline import VideoPipeline, HIGH_PERFORMANCE_CONFIG

# 高性能設定でパイプライン作成
pipeline = VideoPipeline(HIGH_PERFORMANCE_CONFIG)

# カスタム可視化設定
from src.predictor.visualization import VisualizationConfig
vis_config = VisualizationConfig(
    ball_radius=12,
    trajectory_length=25,
    enable_smoothing=True,
    enable_prediction=True
)

result = pipeline.process_video(
    video_path="input.mp4",
    detector_config=detector_config,
    output_path="output.mp4",
    visualization_config=vis_config
)
```

### 非同期処理

```python
# 非同期処理開始
async_result = pipeline.process_video_async(
    video_path="input.mp4",
    detector_config=detector_config,
    output_path="output.mp4"
)

# 進捗監視
while not async_result.is_completed():
    progress = async_result.get_progress()
    print(f"進捗: {progress*100:.1f}%")
    time.sleep(0.5)

# 結果取得
result = async_result.get_result()
```

## ⚙️ 設定オプション

### PipelineConfig パラメータ

```python
from src.predictor.pipeline import PipelineConfig

config = PipelineConfig(
    # バッファサイズ
    frame_buffer_size=30,        # フレームバッファ
    tensor_buffer_size=15,       # テンソルバッファ
    
    # スレッド数
    num_preprocessing_threads=2, # 前処理スレッド数
    gpu_batch_size=4,           # GPUバッチサイズ
    
    # 最適化設定
    enable_memory_optimization=True,
    enable_cpu_offload=False,
    
    # デバッグ
    enable_profiling=True,
    log_queue_sizes=True
)
```

### プリセット設定

```python
from src.predictor.pipeline import (
    HIGH_PERFORMANCE_CONFIG,    # 最高性能
    MEMORY_EFFICIENT_CONFIG,    # メモリ効率
    REALTIME_CONFIG,           # リアルタイム
    DEBUG_CONFIG               # デバッグ用
)
```

## 📊 性能ベンチマーク

```python
# 性能テスト実行
results = pipeline.benchmark_performance(
    video_path="test_video.mp4",
    detector_config=detector_config
)

for config_name, result in results.items():
    print(f"{config_name}: {result['average_fps']:.2f} FPS")
```

## 🎨 可視化オプション

```python
from src.predictor.visualization import VisualizationConfig

vis_config = VisualizationConfig(
    # 球体描画
    ball_radius=10,
    ball_color=(0, 0, 255),     # BGR色指定
    
    # 軌跡表示
    show_trajectory=True,
    trajectory_length=20,
    trajectory_color=(0, 255, 255),
    
    # 高度機能
    enable_smoothing=True,      # 位置スムージング
    enable_prediction=True,     # 予測表示
    
    # フィルタリング
    confidence_threshold=0.3
)
```

## 📈 性能最適化のヒント

### GPU利用率最大化
- `tensor_buffer_size`を大きくしてGPU待機時間を削減
- `gpu_batch_size`を調整してGPUメモリを有効活用
- `num_preprocessing_threads`を増やして前処理を並列化

### メモリ効率化
- `enable_memory_optimization=True`で自動最適化
- `enable_cpu_offload=True`でGPUメモリを節約
- `frame_buffer_size`を調整してメモリ使用量制御

### リアルタイム処理
- `enable_frame_skipping=True`でフレームスキップ
- `frame_skip_interval`で処理間隔調整
- 小さな`buffer_size`で遅延を最小化

## 🔧 トラブルシューティング

### よくある問題

1. **GPU OutOfMemory**
   - `gpu_batch_size`を小さくする
   - `enable_cpu_offload=True`を設定
   - `max_gpu_memory_fraction`を調整

2. **処理が遅い**
   - `tensor_buffer_size`を増やす
   - `num_preprocessing_threads`を増やす
   - GPUが利用されているか確認

3. **メモリ不足**
   - `MEMORY_EFFICIENT_CONFIG`を使用
   - `frame_buffer_size`を小さくする
   - `enable_memory_optimization=True`を設定

### デバッグモード

```python
from src.predictor.pipeline import DEBUG_CONFIG

pipeline = VideoPipeline(DEBUG_CONFIG)
# 詳細なログとプロファイリング情報が出力される
```

## 📝 例とサンプル

詳細な使用例は`examples.py`を参照してください：

```python
from src.predictor.pipeline import examples

# 基本処理例
examples.example_basic_processing()

# 高性能処理例
examples.example_high_performance_processing()

# 非同期処理例
examples.example_async_processing()
```

## 🔗 関連モジュール

- `src.predictor.ball`: ボール検出器
- `src.predictor.visualization`: 可視化コンポーネント
- `src.predictor.base`: ベースクラス

## 📄 ライセンス

このソフトウェアは研究・開発目的で提供されています。 