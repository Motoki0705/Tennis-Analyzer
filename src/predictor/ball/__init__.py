"""
Ball Detection Module
=====================

🎾 テニスボール検出システム統合パッケージ

このモジュールは、複数のアーキテクチャによる高精度なテニスボール検出システムを提供します。
自動モデル選択、統一されたAPI、高度なトラッキング機能を組み合わせ、
実用的なボール検出ソリューションを実現します。

主な特徴:
- 🤖 自動モデルタイプ検出とファクトリーパターン
- 🎯 複数アーキテクチャの統合サポート
- 📊 統一されたAPI インターフェース
- ⚡ 高性能な時間的一貫性追跡

========================================
🚀 Quick Start (Factory Pattern)
========================================

```python
from src.predictor.ball import create_ball_detector

# 1. 自動検出による簡単な開始
detector = create_ball_detector("path/to/model.ckpt")  # LiteTrackNet自動選択
detector = create_ball_detector("path/to/model.pth")   # WASB-SBDT自動選択

# 2. 明示的なモデルタイプ指定
detector = create_ball_detector("path/to/model.ckpt", model_type="lite_tracknet")
detector = create_ball_detector("path/to/model.pth", model_type="wasb_sbdt", 
                               config_path="config.yaml")

# 3. 統一された推論パイプライン
frame_data = [(frame1, {"frame_id": 0}), (frame2, {"frame_id": 1}), ...]
processed = detector.preprocess(frame_data)
results = detector.infer(processed)
detections = detector.postprocess(results)
```

========================================
🏗️ Model Architectures
========================================

LiteTrackNet Detector
---------------------
- **アーキテクチャ**: 軽量な時系列解析ネットワーク
- **入力形式**: 3連続フレーム [3×H×W×C] → [9×H×W]
- **出力形式**: ヒートマップ [H×W]
- **モデルファイル**: .ckpt (PyTorch Lightning checkpoint)
- **特徴**: 高速推論、少ないメモリ使用量

```python
from src.predictor.ball import LiteTrackNetDetector

detector = LiteTrackNetDetector(
    model_path="checkpoints/lite_tracknet.ckpt",
    device="cuda",
    input_size=(360, 640)  # (height, width)
)
```

WASB-SBDT Detector (HRNet)
--------------------------
- **アーキテクチャ**: HRNet + TrackNetV2 + Online Tracker
- **入力形式**: 3フレーム [B, 3×C, 288, 512]
- **出力形式**: 3フレーム分ヒートマップ [B, 3, 288, 512]
- **モデルファイル**: .pth / .pth.tar
- **特徴**: 最高精度、統合トラッキング、3フレーム同時予測

```python
from src.predictor.ball import WASBSBDTDetector

detector = WASBSBDTDetector(
    model_path="checkpoints/wasb_model.pth",
    config_path="configs/wasb_config.yaml",  # optional
    device="cuda"
)
```

========================================
🔧 Advanced Usage
========================================

Model Discovery & Validation
-----------------------------
```python
from src.predictor.ball.factory import get_available_models, validate_model_compatibility

# 利用可能なモデルを検索
models = get_available_models("checkpoints/ball")
print(f"LiteTrackNet models: {models['lite_tracknet']}")
print(f"WASB-SBDT models: {models['wasb_sbdt']}")

# モデル互換性チェック
is_valid = validate_model_compatibility("model.ckpt", "lite_tracknet")
```

Batch Processing Pipeline
-------------------------
```python
# 大量フレーム処理の例
def process_video_frames(detector, video_frames):
    frame_data = [(frame, {"frame_id": i}) for i, frame in enumerate(video_frames)]
    
    # バッチ処理
    processed = detector.preprocess(frame_data)
    inference_results = detector.infer(processed)
    detections = detector.postprocess(inference_results)
    
    return detections

# 使用例
detections = process_video_frames(detector, video_frames)
for frame_id, detection_list in detections.items():
    for x_norm, y_norm, confidence in detection_list:
        print(f"Frame {frame_id}: Ball at ({x_norm:.3f}, {y_norm:.3f}), conf={confidence:.3f}")
```

Custom Configuration
--------------------
```python
# WASB-SBDT カスタム設定
detector = WASBSBDTDetector(
    model_path="model.pth",
    device="cuda"
)

# 後処理パラメータの調整
detector.score_threshold = 0.3  # 検出閾値
detector.use_hm_weight = True   # 重み付き重心計算

# LiteTrackNet カスタム設定
detector = LiteTrackNetDetector(
    model_path="model.ckpt",
    input_size=(480, 720),  # 高解像度処理
    device="cuda"
)
```

========================================
📊 Data Format Specifications
========================================

Input Format (共通)
-------------------
```python
frame_data: List[Tuple[np.ndarray, Dict[str, Any]]] = [
    (frame_array,  # shape: [H, W, C] (BGR/RGB)
     {"frame_id": int, "timestamp": float, ...}),
    ...
]
```

Output Format (共通)
--------------------
```python
detections: Dict[str, List[List[float]]] = {
    "frame_0": [[x_norm, y_norm, confidence], ...],  # 正規化座標 [0, 1]
    "frame_1": [[x_norm, y_norm, confidence], ...],
    ...
}
```

Model Information
-----------------
```python
info = detector.model_info
# 共通フィールド:
# - model_type: str
# - model_path: str
# - frames_required: int
# - device: str
# - architecture: str
```

========================================
⚡ Performance Optimization
========================================

GPU Memory Management
---------------------
```python
import torch

# メモリ効率的な処理
with torch.no_grad():
    results = detector.infer(processed_data)

# メモリ解放
torch.cuda.empty_cache()
```

Batch Size Optimization
-----------------------
```python
# フレーム数に応じたバッチサイズ調整
frames_required = detector.frames_required
optimal_batch_size = min(len(frame_data) - frames_required + 1, 32)
```

========================================
ℹ️ Package Information
========================================

Available Detectors:
- LiteTrackNetDetector: 高速・軽量検出器
- WASBSBDTDetector: 高精度・統合トラッキング検出器

Factory Functions:
- create_ball_detector(): 自動選択ファクトリー
- detect_model_type(): モデルタイプ自動判定
- get_available_models(): モデル検索機能

Performance: リアルタイム処理対応、GPU最適化、メモリ効率化
Compatibility: PyTorch, PyTorch Lightning, WASB-SBDT統合
"""

from .lite_tracknet_detector import LiteTrackNetDetector
from .wasb_sbdt_detector import WASBSBDTDetector
from .factory import create_ball_detector

__all__ = [
    'LiteTrackNetDetector',
    'WASBSBDTDetector', 
    'create_ball_detector',
]