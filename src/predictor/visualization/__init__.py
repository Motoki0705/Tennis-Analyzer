"""
Visualization Module
====================

🎨 テニスボール検出結果可視化システム

このモジュールは、ボール検出結果を高品質な動画オーバーレイとして可視化するための
包括的なソリューションを提供します。軌跡追跡、信頼度表示、予測可視化など、
分析と評価に必要な全ての視覚化機能を統合しています。

主な特徴:
- 🎥 高品質なオーバーレイ動画生成
- 🎯 リアルタイム描画とフレーム処理
- 📊 カスタマイズ可能な視覚化設定
- ⚡ 効率的なバッチ処理とプログレス管理

========================================
🚀 Quick Start
========================================

```python
from src.predictor.visualization import VideoOverlay, VisualizationConfig

# 1. デフォルト設定で簡単開始
overlay = VideoOverlay()
output_path = overlay.create_overlay_video(
    video_path="input.mp4",
    detections=detection_results,
    output_path="output_with_overlay.mp4"
)

# 2. カスタム設定による高度な制御
config = VisualizationConfig(
    ball_radius=12,
    show_trajectory=True,
    trajectory_length=20,
    enable_smoothing=True
)
overlay = VideoOverlay(config)
```

========================================
🏗️ Core Components
========================================

VideoOverlay - 動画オーバーレイエンジン
-------------------------------------
動画全体に対するオーバーレイ処理とファイル管理を担当

**主要機能:**
- 完全な動画オーバーレイ生成
- 進捗管理とプログレスコールバック
- サマリー動画作成（検出フレームのみ）
- フレーム抽出と統計情報生成

```python
from src.predictor.visualization import VideoOverlay

overlay = VideoOverlay()

# 完全なオーバーレイ動画作成
output_path = overlay.create_overlay_video(
    video_path="tennis_match.mp4",
    detections=ball_detections,
    output_path="annotated_match.mp4",
    progress_callback=lambda p: print(f"Progress: {p*100:.1f}%")
)

# 検出フレームのみのサマリー動画
summary_path = overlay.create_detection_summary_video(
    video_path="tennis_match.mp4",
    detections=ball_detections,
    output_path="best_detections.mp4",
    summary_frames=50
)

# 統計情報取得
stats = overlay.get_processing_stats(ball_detections)
print(f"Total detections: {stats['total_detections']}")
print(f"Frames with balls: {stats['frames_with_detections']}")
```

DetectionRenderer - リアルタイム描画エンジン
-----------------------------------------
個別フレームへの検出結果描画と視覚効果を担当

**主要機能:**
- 高精度な球体位置描画
- 動的軌跡可視化（厚み変化）
- 位置スムージングとノイズ除去
- 未来予測位置の表示

```python
from src.predictor.visualization import DetectionRenderer, VisualizationConfig

# カスタム設定でレンダラー作成
config = VisualizationConfig(
    ball_color=(0, 255, 0),      # 緑色の球体
    trajectory_length=15,         # 15フレーム分の軌跡
    enable_smoothing=True,        # 位置スムージング有効
    enable_prediction=True        # 予測表示有効
)
renderer = DetectionRenderer(config)

# フレーム単位での描画
for frame_idx, frame in enumerate(video_frames):
    frame_detections = detections.get(f"frame_{frame_idx}", [])
    
    rendered_frame = renderer.render_frame(
        frame=frame,
        detections=frame_detections,
        frame_info={
            'frame_number': frame_idx,
            'timestamp': frame_idx / fps,
            'detection_count': len(frame_detections)
        }
    )
    
    cv2.imshow('Ball Detection', rendered_frame)
```

VisualizationConfig - 設定管理システム
------------------------------------
全ての視覚化パラメータの統一管理

**設定カテゴリ:**
- **球体描画**: 色、サイズ、中心点表示
- **軌跡表示**: 長さ、色、厚み変化
- **テキスト**: フォント、色、位置オフセット
- **動画出力**: コーデック、品質、プログレス
- **高度機能**: スムージング、予測、フィルタリング

```python
from src.predictor.visualization import VisualizationConfig
from src.predictor.visualization.config import (
    HIGH_QUALITY_CONFIG,
    MINIMAL_CONFIG,
    TRAJECTORY_FOCUSED_CONFIG
)

# プリセット設定の使用
config = HIGH_QUALITY_CONFIG  # 高品質描画設定

# カスタム設定作成
custom_config = VisualizationConfig(
    # 球体描画設定
    ball_radius=10,
    ball_color=(255, 0, 0),     # 青色 (BGR)
    center_radius=4,
    center_color=(255, 255, 255),
    
    # 軌跡設定
    show_trajectory=True,
    trajectory_length=25,
    trajectory_color=(0, 255, 255),  # 黄色
    trajectory_max_thickness=5,
    
    # フィルタリング
    confidence_threshold=0.4,
    
    # 高度機能
    enable_smoothing=True,
    smoothing_window=5,
    enable_prediction=True,
    prediction_frames=3
)

# 設定の動的更新
config.update(
    ball_radius=15,
    trajectory_length=30
)
```

========================================
🔧 Advanced Usage Patterns
========================================

実時間処理パイプライン
--------------------
```python
import cv2
from src.predictor.visualization import DetectionRenderer, VisualizationConfig

def real_time_visualization(detector, video_source=0):
    # リアルタイム処理用設定
    config = VisualizationConfig(
        ball_radius=8,
        show_trajectory=True,
        trajectory_length=10,
        enable_smoothing=True,
        smoothing_window=3
    )
    
    renderer = DetectionRenderer(config)
    cap = cv2.VideoCapture(video_source)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 検出実行
            detections = detector.detect_frame(frame)
            
            # 可視化
            rendered_frame = renderer.render_frame(frame, detections)
            
            cv2.imshow('Real-time Ball Detection', rendered_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
```

バッチ処理とプログレス管理
------------------------
```python
from src.predictor.visualization import VideoOverlay
import time

def batch_process_videos(video_list, detection_results):
    overlay = VideoOverlay()
    
    def progress_callback(progress):
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProgress: |{bar}| {progress*100:.1f}%', end='', flush=True)
    
    for video_path, detections in zip(video_list, detection_results):
        output_path = video_path.replace('.mp4', '_annotated.mp4')
        
        print(f"\nProcessing: {video_path}")
        overlay.create_overlay_video(
            video_path=video_path,
            detections=detections,
            output_path=output_path,
            progress_callback=progress_callback
        )
        print(f"\nCompleted: {output_path}")
```

カスタム描画ハンドラー
--------------------
```python
from src.predictor.visualization import DetectionRenderer
import cv2

class CustomRenderer(DetectionRenderer):
    def render_frame(self, frame, detections, frame_info=None):
        # 基本描画を実行
        rendered_frame = super().render_frame(frame, detections, frame_info)
        
        # カスタム要素追加
        if detections:
            # 検出数に応じた背景色変更
            detection_count = len(detections)
            if detection_count > 3:
                # 多数検出時は背景を薄緑に
                overlay = rendered_frame.copy()
                overlay[:] = (0, 255, 0)
                rendered_frame = cv2.addWeighted(rendered_frame, 0.9, overlay, 0.1, 0)
            
            # カスタム信頼度ヒストグラム
            self._draw_confidence_histogram(rendered_frame, detections)
        
        return rendered_frame
    
    def _draw_confidence_histogram(self, frame, detections):
        # 信頼度分布のヒストグラム描画
        confidences = [det[2] for det in detections if len(det) >= 3]
        # ヒストグラム実装...
```

========================================
📊 Data Format Specifications
========================================

Detection Data Structure
-------------------------
```python
# 入力検出データ形式
detections: Dict[str, List[List[float]]] = {
    "frame_000000": [
        [x_norm, y_norm, confidence],  # 正規化座標 [0, 1]
        [0.45, 0.32, 0.89],           # 例: x=45%, y=32%, conf=89%
        ...
    ],
    "frame_000001": [...],
    ...
}

# フレーム情報データ
frame_info: Dict[str, Any] = {
    'frame_number': int,      # フレーム番号
    'timestamp': float,       # タイムスタンプ（秒）
    'detection_count': int,   # 検出数
    'fps': float,            # フレームレート（オプション）
    'resolution': tuple      # (width, height)（オプション）
}
```

Configuration Schema
--------------------
```python
# 基本描画設定
ball_radius: int = 8                    # 球体半径
center_radius: int = 3                  # 中心点半径
ball_color: Tuple[int, int, int]        # BGR色指定
center_color: Tuple[int, int, int]      # 中心点色

# 軌跡設定
show_trajectory: bool = True            # 軌跡表示有効/無効
trajectory_length: int = 15             # 軌跡点数
trajectory_color: Tuple[int, int, int]  # 軌跡色
trajectory_max_thickness: int = 3       # 最大線幅
trajectory_min_thickness: int = 1       # 最小線幅

# フィルタリング
confidence_threshold: float = 0.3       # 信頼度閾値

# 高度機能
enable_smoothing: bool = False          # スムージング有効
smoothing_window: int = 3               # スムージングウィンドウ
enable_prediction: bool = False         # 予測表示有効
prediction_frames: int = 2              # 予測フレーム数
```

========================================
⚡ Performance Optimization
========================================

メモリ効率化
-----------
```python
# 大きなビデオ処理時のメモリ管理
import gc

def memory_efficient_processing(video_path, detections):
    overlay = VideoOverlay()
    
    # チャンク単位での処理
    chunk_size = 1000  # フレーム数
    total_frames = get_video_frame_count(video_path)
    
    for start_frame in range(0, total_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, total_frames)
        
        # チャンクごとの検出データ
        chunk_detections = {
            k: v for k, v in detections.items()
            if start_frame <= int(k.replace('frame_', '')) < end_frame
        }
        
        # 処理実行
        process_chunk(video_path, chunk_detections, start_frame, end_frame)
        
        # メモリ解放
        del chunk_detections
        gc.collect()
```

並列処理対応
-----------
```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_frame_rendering(frames, detections, config):
    renderer = DetectionRenderer(config)
    
    def render_single_frame(args):
        frame_idx, frame = args
        frame_detections = detections.get(f"frame_{frame_idx:06d}", [])
        return renderer.render_frame(frame, frame_detections)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        rendered_frames = list(executor.map(
            render_single_frame,
            enumerate(frames)
        ))
    
    return rendered_frames
```

========================================
ℹ️ Package Information
========================================

Core Classes:
- VideoOverlay: 動画レベルのオーバーレイ処理とファイル管理
- DetectionRenderer: フレームレベルの描画エンジンと視覚効果
- VisualizationConfig: 統一設定管理とプリセット提供

Advanced Features:
- Real-time visualization: リアルタイム描画対応
- Trajectory tracking: 動的軌跡追跡と表示
- Predictive visualization: 未来位置予測表示
- Custom rendering: 拡張可能な描画フレームワーク

Performance: 高効率メモリ管理、並列処理対応、大容量動画対応
Output Quality: 高品質コーデック、カスタマイズ可能な解像度とFPS
Compatibility: OpenCV統合、NumPy最適化、クロスプラットフォーム対応
"""

from .overlay import VideoOverlay
from .renderer import DetectionRenderer
from .config import VisualizationConfig

__all__ = [
    'VideoOverlay',
    'DetectionRenderer',
    'VisualizationConfig',
]