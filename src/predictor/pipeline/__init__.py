"""
Parallel Processing Pipeline Module
===================================

🚀 高効率テニスボール検出・可視化パイプライン

このモジュールは、マルチスレッド並列処理による高効率なボール検出パイプラインを提供します。
GPU利用率の最大化と待機時間の最小化を目的とした設計により、
リアルタイム処理と大容量動画処理の両方に対応します。

主な特徴:
- 🔄 マルチスレッド並列処理アーキテクチャ
- ⚡ GPU待機時間最小化設計
- 📊 効率的なキューベース処理
- 🎥 リアルタイム・バッチ処理統合対応

========================================
🏗️ Pipeline Architecture
========================================

Thread Distribution:
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

Queue-Based Data Flow:
- Raw Frame Queue: フレーム読み込み → 前処理
- Tensor Queue: 前処理済み → GPU推論（バッファリング）
- Result Queue: 推論結果 → 後処理
- Render Queue: 検出結果 → 可視化出力

========================================
🚀 Quick Start
========================================

```python
from src.predictor.pipeline import VideoPipeline, PipelineConfig

# 1. 基本的なパイプライン実行
pipeline = VideoPipeline()
output_path = pipeline.process_video(
    video_path="input.mp4",
    detector_config={"model_type": "wasb_sbdt", "model_path": "model.pth"},
    output_path="output.mp4"
)

# 2. カスタム設定による最適化
config = PipelineConfig(
    frame_buffer_size=50,      # フレームバッファサイズ
    tensor_buffer_size=20,     # テンソルバッファサイズ
    num_preprocessing_threads=2,  # 前処理スレッド数
    gpu_batch_size=4,          # GPU バッチサイズ
    enable_visualization=True   # 可視化有効
)

pipeline = VideoPipeline(config)
pipeline.process_video_async(video_path, detector_config, output_path)
```

========================================
📊 Performance Features
========================================

GPU Utilization Optimization:
- Tensor Pre-buffering: 前処理済みテンソルの事前蓄積
- Batch Inference: 複数フレームの同時推論
- Pipeline Parallelism: GPU待機時間の完全排除

Memory Management:
- Circular Buffer: 固定メモリ使用量
- Streaming Processing: メモリ効率的な大容量動画処理
- Resource Cleanup: 自動リソース管理

Thread Safety:
- Lock-free Queues: 高速データ交換
- Exception Handling: スレッド間エラー伝播
- Graceful Shutdown: 安全な処理終了
"""

from .video_pipeline import VideoPipeline
from .config import PipelineConfig
from .async_processor import AsyncVideoProcessor
from . import examples

__all__ = [
    'VideoPipeline',
    'PipelineConfig', 
    'AsyncVideoProcessor',
    'examples'
] 