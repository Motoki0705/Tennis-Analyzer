# Ball Tracker Module

WASB-SBDT からコピーされたテニスボール追跡システム

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使い方

```python
from ball_tracker import BallTracker

# トラッカー初期化
tracker = BallTracker(model_path="path/to/weights.pth.tar")

# 動画処理
results = tracker.track_video("input.mp4", "output.mp4")

# 結果確認
for result in results:
    if result['visible']:
        print(f"Frame {result['frame']}: Ball at ({result['x']:.1f}, {result['y']:.1f})")
```

### フレーム単位処理

```python
import cv2

cap = cv2.VideoCapture("video.mp4")
tracker = BallTracker(model_path="weights.pth.tar")

buffer = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    buffer.append(frame)
    if len(buffer) >= tracker.detector.frames_in:
        result = tracker.track_frames(buffer)
        if result['visible']:
            print(f"Ball detected at ({result['x']}, {result['y']})")
        buffer.pop(0)  # スライディングウィンドウ
```

## ファイル構成

- `models/hrnet.py` - HRNet アーキテクチャ
- `online.py` - オンライントラッカー
- `postprocessor.py` - ヒートマップ後処理
- `video_demo.py` - SimpleDetector クラス
- `utils/image.py` - 画像変換ユーティリティ

## 元実装

このモジュールは以下の論文実装から抽出されました:
- 論文: "Widely Applicable Strong Baseline for Sports Ball Detection and Tracking"
- GitHub: https://github.com/starashima/WASB-SBDT_sandbox

## ライセンス

元リポジトリのライセンスに従います。
