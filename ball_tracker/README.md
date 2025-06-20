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

### 分析ツール (NEW!)

動画の検出性能を分析し、蒸留学習の戦略を決定:

```bash
# 基本的な分析
python run_analysis.py \
  --video tennis_video.mp4 \
  --model_path weights.pth.tar

# バッチ分析
python batch_analysis.py \
  --video_dir ./tennis_videos/ \
  --model_path weights.pth.tar
```

詳細は `README_ANALYSIS.md` を参照。

### 強化分析ツール (NEW! 🚀)

3段階フィルタリングによる高精度分析：

```bash
# ローカル分類器の学習
python -m ball_tracker.local_classifier.train \
  --annotation_file coco_annotations.json \
  --images_dir ./images/

# 強化分析実行
python enhanced_analysis_tool.py \
  --video tennis_video.mp4 \
  --ball_tracker_model ball_tracker.pth.tar \
  --local_classifier_model local_classifier_checkpoints/best_model.pth
```

詳細は `README_ENHANCED.md` を参照。

## ファイル構成

- `models/hrnet.py` - HRNet アーキテクチャ
- `online.py` - オンライントラッカー
- `postprocessor.py` - ヒートマップ後処理
- `video_demo.py` - SimpleDetector クラス
- `utils/image.py` - 画像変換ユーティリティ
- `analysis_tool.py` - 性能分析・可視化ツール
- `batch_analysis.py` - バッチ処理分析ツール  
- `run_analysis.py` - 簡易実行スクリプト
- `enhanced_analysis_tool.py` - 3段階フィルタリング強化分析
- `local_classifier/` - ローカル分類器モジュール（16x16パッチ2値分類）

## 元実装

このモジュールは以下の論文実装から抽出されました:
- 論文: "Widely Applicable Strong Baseline for Sports Ball Detection and Tracking"
- GitHub: https://github.com/starashima/WASB-SBDT_sandbox

## ライセンス

元リポジトリのライセンスに従います。
