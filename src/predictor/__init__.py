"""
Unified Tennis Ball Detection & Processing System
===============================================

🎾 テニスボール検出・処理・可視化統合システム

このモジュールは、テニス動画からのボール検出、並列処理パイプライン、
高品質可視化までを統合したワンストップソリューションを提供します。
複数のアーキテクチャ、最適化設定、使用シナリオに対応した
包括的なテニス分析プラットフォームです。

主な特徴:
- 🤖 複数検出アーキテクチャ (LiteTrackNet, WASB-SBDT)
- 🚀 並列処理パイプライン (GPU最適化)
- 🎨 高品質可視化エンジン (軌跡追跡・予測)
- ⚡ リアルタイム〜バッチ処理対応

========================================
🚀 Quick Start
========================================

## 1. 基本的なボール検出

```python
from src.predictor import create_ball_detector

# 自動検出器選択
detector = create_ball_detector("checkpoints/model.ckpt")  # LiteTrackNet
detector = create_ball_detector("checkpoints/model.pth")   # WASB-SBDT

# フレームデータ準備
frame_data = [(frame1, {"frame_id": "frame_000001"}), ...]

# 検出実行
detections = detector.detect_balls(frame_data)
# → {"frame_000001": [[x_norm, y_norm, confidence], ...], ...}
```

## 2. 高速並列パイプライン

```python
from src.predictor import VideoPipeline, HIGH_PERFORMANCE_CONFIG

# 並列パイプライン作成
pipeline = VideoPipeline(HIGH_PERFORMANCE_CONFIG)

# 動画全体処理
result = pipeline.process_video(
    video_path="tennis_match.mp4",
    detector_config={"model_type": "wasb_sbdt", "model_path": "model.pth"},
    output_path="annotated_match.mp4"
)

print(f"処理完了: {result['average_fps']:.2f} FPS")
```

## 3. 高品質可視化

```python
from src.predictor import VideoOverlay, VisualizationConfig

# カスタム可視化設定
vis_config = VisualizationConfig(
    ball_radius=12,
    trajectory_length=25,
    enable_smoothing=True,
    enable_prediction=True
)

# 可視化実行
overlay = VideoOverlay(vis_config)
output_path = overlay.create_overlay_video(
    video_path="input.mp4",
    detections=detections,
    output_path="output.mp4"
)
```

========================================
🏗️ System Architecture
========================================

Detection Layer:
- BaseBallDetector: 統一API
- LiteTrackNetDetector: 高速軽量検出
- WASBSBDTDetector: 高精度トラッキング統合

Processing Layer:
- VideoPipeline: 並列処理エンジン
- AsyncVideoProcessor: マルチスレッド制御
- PipelineConfig: 設定管理

Visualization Layer:
- VideoOverlay: 動画オーバーレイ
- DetectionRenderer: フレーム描画
- VisualizationConfig: 可視化設定

========================================
⚡ Performance Configurations
========================================

```python
from src.predictor import (
    HIGH_PERFORMANCE_CONFIG,    # 最高速度
    MEMORY_EFFICIENT_CONFIG,    # メモリ最適化
    REALTIME_CONFIG,           # リアルタイム
    DEBUG_CONFIG               # デバッグ用
)

# 用途別最適化
pipeline = VideoPipeline(HIGH_PERFORMANCE_CONFIG)
```

========================================
🎯 Use Cases & Examples
========================================

### リアルタイム処理
```python
from src.predictor import VideoPipeline, REALTIME_CONFIG

pipeline = VideoPipeline(REALTIME_CONFIG)
async_result = pipeline.process_video_async(...)

# 進捗監視
while not async_result.is_completed():
    progress = async_result.get_progress()
    print(f"進捗: {progress*100:.1f}%")
```

### バッチ処理
```python
# 大量動画の一括処理
videos = ["match1.mp4", "match2.mp4", "match3.mp4"]
for video in videos:
    result = pipeline.process_video(
        video_path=video,
        detector_config=detector_config,
        output_path=video.replace('.mp4', '_annotated.mp4')
    )
```

### カスタム検出器
```python
# 検出器の詳細制御
detector = create_ball_detector(
    model_path="custom_model.pth",
    model_type="wasb_sbdt",
    config_path="custom_config.yaml",
    device="cuda"
)

# 段階的処理
frame_data = load_frames(video_path)
preprocessed = detector.preprocess(frame_data)
inference_results = detector.infer(preprocessed)
detections = detector.postprocess(inference_results)
```

========================================
📊 Performance Optimization
========================================

### GPU利用率最大化
- Tensor pre-buffering: 前処理済みテンソルの事前蓄積
- Batch inference: 複数フレームの同時推論
- Pipeline parallelism: GPU待機時間の完全排除

### メモリ効率化
- Streaming processing: 大容量動画の安全処理
- Memory optimization: 自動メモリ管理
- CPU offloading: GPUメモリ節約

### 処理速度向上
- Multi-threading: 最大8スレッドでの並列処理
- Queue optimization: 効率的なデータフロー
- Configuration presets: 用途別最適化

========================================
🔧 Advanced Features
========================================

### 検出精度向上
- Connected components analysis: 高精度位置推定
- Temporal tracking: フレーム間一貫性
- Multi-model ensemble: 複数モデルの統合

### 可視化機能
- Trajectory visualization: 動的軌跡表示
- Smoothing & prediction: 位置補正と予測
- Custom rendering: 拡張可能な描画システム

### モニタリング
- Performance profiling: 詳細性能分析
- Progress tracking: リアルタイム進捗
- Error handling: 堅牢なエラー回復

========================================
ℹ️ Package Information
========================================

Components:
- Ball Detection: 複数アーキテクチャ対応検出エンジン
- Pipeline Processing: 並列処理・最適化エンジン  
- Visualization: 高品質オーバーレイ・軌跡システム
- Configuration: 柔軟な設定管理・プリセット

Performance: リアルタイム対応、GPU最適化、メモリ効率化
Compatibility: PyTorch、OpenCV、WASB-SBDT統合
Use Cases: 研究分析、リアルタイム配信、大規模バッチ処理
"""

# Core Detection API
from .base.detector import BaseBallDetector
from .ball.factory import create_ball_detector

# Ball Detectors
from .ball import LiteTrackNetDetector, WASBSBDTDetector

# Parallel Processing Pipeline  
from .pipeline import VideoPipeline, PipelineConfig, AsyncVideoProcessor
from .pipeline.config import (
    HIGH_PERFORMANCE_CONFIG,
    MEMORY_EFFICIENT_CONFIG, 
    REALTIME_CONFIG,
    DEBUG_CONFIG
)

# Visualization Components
from .visualization import VideoOverlay, DetectionRenderer, VisualizationConfig

# Example and utilities
from .pipeline import examples

__all__ = [
    # Core Detection API
    'BaseBallDetector',
    'create_ball_detector',
    
    # Specific Detectors
    'LiteTrackNetDetector', 
    'WASBSBDTDetector',
    
    # Pipeline Processing
    'VideoPipeline',
    'PipelineConfig',
    'AsyncVideoProcessor',
    
    # Pipeline Configurations
    'HIGH_PERFORMANCE_CONFIG',
    'MEMORY_EFFICIENT_CONFIG',
    'REALTIME_CONFIG', 
    'DEBUG_CONFIG',
    
    # Visualization
    'VideoOverlay',
    'DetectionRenderer',
    'VisualizationConfig',
    
    # Examples and utilities
    'examples',
]