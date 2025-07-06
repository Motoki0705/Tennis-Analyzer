# Scripts Usage Guide

このディレクトリにはプロジェクトで使用する実行スクリプトが含まれています。

## 利用可能なスクリプト

### 1. ローカル分類器学習: `train_local_classifier.sh`

16x16パッチでのボール2値分類器を学習します。

#### 基本使用法

```bash
# デフォルト設定で学習
./scripts/train_local_classifier.sh

# カスタム設定で学習
./scripts/train_local_classifier.sh \
    --model_type efficient \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.0005
```

#### 主要オプション

| オプション | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--annotation_file` | `datasets/ball/coco_annotations_ball_pose_court.json` | アノテーションファイル |
| `--images_dir` | `datasets/ball/images` | 画像ディレクトリ |
| `--output_dir` | `./local_classifier_checkpoints` | 出力ディレクトリ |
| `--model_type` | `standard` | モデルタイプ (`standard` or `efficient`) |
| `--epochs` | `50` | エポック数 |
| `--batch_size` | `64` | バッチサイズ |
| `--learning_rate` | `0.001` | 学習率 |
| `--device` | `cuda` | デバイス (`cuda` or `cpu`) |

#### 出力ファイル

```
local_classifier_checkpoints/
├── best_model.pth              # 最高性能モデル
├── final_model.pth             # 最終モデル
├── training_history.json       # 学習履歴
├── final_training_curves.png   # 学習曲線
├── confusion_matrix.png        # 混同行列
└── logs/                       # TensorBoardログ
```

### 2. 統合分析: `run_enhanced_analysis.sh`

3段階フィルタリング（ball_tracker + ローカル分類器 + 軌跡バリデーション）による動画分析を実行します。

#### 基本使用法

```bash
# ローカル分類器のみ使用
./scripts/run_enhanced_analysis.sh \
    --video path/to/video.mp4 \
    --local_classifier checkpoints/best_model.pth

# 完全な3段階分析
./scripts/run_enhanced_analysis.sh \
    --video path/to/video.mp4 \
    --ball_tracker_weights path/to/ball_tracker.pth \
    --local_classifier path/to/local_classifier.pth \
    --output_dir ./analysis_results
```

#### 主要オプション

| オプション | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--video` | **必須** | 入力動画ファイル |
| `--local_classifier` | **必須** | 学習済みローカル分類器 |
| `--ball_tracker_config` | `third_party/WASB-SBDT/configs/model/tracknetv2.yaml` | ball_tracker設定 |
| `--ball_tracker_weights` | なし | ball_tracker重みファイル |
| `--output_dir` | 自動生成 | 出力ディレクトリ |
| `--stage1_threshold` | `0.5` | Stage1信頼度閾値 |
| `--stage2_threshold` | `0.5` | Stage2信頼度閾値 |
| `--stage3_max_distance` | `50.0` | Stage3最大移動距離 |
| `--device` | `cuda` | デバイス |
| `--no_visualize` | false | 可視化を無効化 |

#### 出力ファイル

```
analysis_results/video_name_enhanced/
├── analysis_results.json        # 詳細分析結果
├── summary_report.md            # サマリーレポート
└── video_name_enhanced_analysis.mp4  # 可視化動画
```

## Windows使用時の注意

Windowsでbashスクリプトを実行する場合：

### 1. Git Bash使用
```bash
# Git Bashを開いて実行
./scripts/train_local_classifier.sh --help
```

### 2. WSL使用
```bash
# WSL (Windows Subsystem for Linux) から実行
./scripts/train_local_classifier.sh --help
```

### 3. PowerShell直接実行
```powershell
# PowerShellから直接Pythonコマンドを実行
python -m src.ball.local_classifier.train --help
python -m src.ball.enhanced_analysis_tool --help
```

## 実行例

### 1. 学習からanalysisまでの完全フロー

```bash
# Step 1: ローカル分類器学習
./scripts/train_local_classifier.sh \
    --model_type standard \
    --epochs 50 \
    --batch_size 64

# Step 2: 学習完了後、分析実行
./scripts/run_enhanced_analysis.sh \
    --video samples/lite_tracknet.mp4 \
    --local_classifier ./local_classifier_checkpoints/best_model.pth
```

### 2. 高速テスト実行

```bash
# 少ないエポック数でクイックテスト
./scripts/train_local_classifier.sh \
    --model_type efficient \
    --epochs 5 \
    --batch_size 32

# テスト動画で分析
./scripts/run_enhanced_analysis.sh \
    --video samples/lite_tracknet.mp4 \
    --local_classifier ./local_classifier_checkpoints/best_model.pth \
    --no_visualize
```

## トラブルシューティング

### 1. インポートエラー
```bash
# Pythonパスの確認
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 2. CUDA out of memory
```bash
# バッチサイズを削減
./scripts/train_local_classifier.sh --batch_size 16
```

### 3. データセットが見つからない
```bash
# データセットパスの確認
ls -la datasets/ball/
ls -la datasets/ball/images/
```

## ヘルプ

各スクリプトの詳細なヘルプは `--help` オプションで確認できます：

```bash
./scripts/train_local_classifier.sh --help
./scripts/run_enhanced_analysis.sh --help
``` 