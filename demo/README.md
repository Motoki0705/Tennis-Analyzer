# VideoPredictor Demo

## 概要

`video_predictor_demo.py` は、Tennis-Analyzerの `VideoPredictor` クラスを使用してテニス動画に対してボール・コート・ポーズの並列推論を実行するデモスクリプトです。

## 主な機能

- **マルチタスク並列推論**: ボール検出、コート検出、ポーズ推定を同時実行
- **パフォーマンス最適化**: キューベースのパイプライン処理による高速化
- **柔軟な設定**: 処理間隔、バッチサイズ、閾値などを設定可能
- **パフォーマンス監視**: 処理速度やリソース使用状況の可視化

## セットアップ

### 1. 前提条件

```bash
# 必要なパッケージのインストール
pip install torch torchvision transformers hydra-core omegaconf opencv-python tqdm
```

### 2. チェックポイントファイルの準備

各モデルのチェックポイントファイルを `checkpoints/` ディレクトリに配置します：

```
checkpoints/
├── ball/
│   └── lite_tracknet.ckpt
├── court/
│   └── lite_tracknet_1heat.ckpt
└── player/
    └── rt_detr.ckpt
```

### 3. 設定ファイルの編集

`demo/config_template.yaml` をコピーして、実際のチェックポイントパスを設定します：

```bash
cp demo/config_template.yaml demo/my_config.yaml
# my_config.yaml を編集してチェックポイントパスを正しく設定
```

## 使用方法

### 基本的な使用方法

```bash
python demo/video_predictor_demo.py \
    --input_path datasets/test/input.mp4 \
    --output_path outputs/demo_output.mp4
```

### カスタム設定ファイルを使用

```bash
python demo/video_predictor_demo.py \
    --config_path demo/my_config.yaml \
    --input_path datasets/test/input.mp4 \
    --output_path outputs/demo_output.mp4
```

### デバッグモードで実行

```bash
python demo/video_predictor_demo.py \
    --input_path datasets/test/input.mp4 \
    --output_path outputs/demo_output.mp4 \
    --debug
```

### CPUで実行

```bash
python demo/video_predictor_demo.py \
    --input_path datasets/test/input.mp4 \
    --output_path outputs/demo_output.mp4 \
    --device cpu
```

## コマンドライン引数

| 引数 | 必須 | 説明 | デフォルト |
|------|------|------|-----------|
| `--input_path` | ✅ | 入力動画ファイルのパス | - |
| `--output_path` | ✅ | 出力動画ファイルのパス | - |
| `--config_path` | ❌ | 設定ファイルのパス | `configs/infer/infer.yaml` |
| `--device` | ❌ | 使用するデバイス (cuda/cpu) | 設定ファイル依存 |
| `--debug` | ❌ | デバッグモードで実行 | False |

## 設定パラメータ

### 処理間隔 (intervals)

各タスクの処理間隔をフレーム単位で指定：

```yaml
intervals:
  ball: 1     # 毎フレーム処理
  court: 30   # 30フレームごとに処理
  pose: 5     # 5フレームごとに処理
```

### バッチサイズ (batch_sizes)

各タスクのバッチサイズを指定：

```yaml
batch_sizes:
  ball: 16    # 16クリップをバッチ処理
  court: 16   # 16フレームをバッチ処理
  pose: 16    # 16フレームをバッチ処理
```

### キュー設定 (queue)

パフォーマンス最適化のためのキュー設定：

```yaml
queue:
  enable_monitoring: true     # パフォーマンス監視
  log_queue_status: false     # キュー状態ログ出力
  gpu_optimization: true      # GPU最適化
  
  base_queue_sizes:
    preprocess: 32            # 前処理キューサイズ
    inference: 16             # 推論キューサイズ
    postprocess: 32           # 後処理キューサイズ
```

## 出力例

### 実行時のログ出力

```
[INFO] 🚀 VideoPredictor デモを開始します...
[INFO] 📋 設定ファイルを読み込み中: configs/infer/infer.yaml
[INFO] 🖥️ デバイス: cuda
[INFO] 📊 Half precision: True
[INFO] 🎾 ボール予測器を初期化中...
[INFO] 📥 ball モデルをロード中: src.ball.lit_module.ball_litmodule.BallLitModule
[INFO] 💾 チェックポイントからロード: checkpoints/ball/lite_tracknet.ckpt
[INFO] 🏟️ コート予測器を初期化中...
[INFO] 📥 court モデルをロード中: src.court.lit_module.court_litmodule.CourtLitModule
[INFO] 💾 チェックポイントからロード: checkpoints/court/lite_tracknet_1heat.ckpt
[INFO] 🤸 ポーズ予測器を初期化中...
[INFO] 📥 player モデルをロード中: src.player.lit_module.player_litmodule.PlayerLitModule
[INFO] 💾 チェックポイントからロード: checkpoints/player/rt_detr.ckpt
[INFO] 🤗 transformers.from_pretrained を使用: pose
[INFO] ⏱️ 処理間隔: {'ball': 1, 'court': 30, 'pose': 5}
[INFO] 📦 バッチサイズ: {'ball': 16, 'court': 16, 'pose': 16}
[INFO] 📹 動画処理を開始: datasets/test/input.mp4 → outputs/demo_output.mp4
```

### パフォーマンス結果

```
[INFO] 📊 処理完了！パフォーマンス結果:
[INFO]   • 総処理フレーム数: 1500
[INFO]   • 総処理時間: 45.23 秒
[INFO]   • 平均FPS: 33.17
[INFO] ✅ 処理が完了しました！出力ファイル: outputs/demo_output.mp4
```

## トラブルシューティング

### よくある問題

1. **チェックポイントファイルが見つからない**
   ```
   FileNotFoundError: Checkpoint not found: checkpoints/ball/lite_tracknet.ckpt
   ```
   → 設定ファイルのパスが正しいか確認してください

2. **GPU メモリ不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   → バッチサイズを小さくするか、`--device cpu` で実行してください

3. **入力動画が見つからない**
   ```
   ❌ 入力ファイルが見つかりません: datasets/test/input.mp4
   ```
   → 入力パスが正しいか確認してください

### デバッグのヒント

- `--debug` フラグを使用して詳細なログを出力
- 小さなテスト動画で動作確認
- GPU使用量を `nvidia-smi` で監視

## パフォーマンス最適化

### 推奨設定

**高性能GPU環境**:
```yaml
batch_sizes:
  ball: 32
  court: 32
  pose: 32
intervals:
  ball: 1
  court: 15
  pose: 3
```

**メモリ制約環境**:
```yaml
batch_sizes:
  ball: 8
  court: 8
  pose: 8
intervals:
  ball: 2
  court: 60
  pose: 10
```

## 関連ファイル

- `src/multi/streaming_overlayer/video_predictor.py`: VideoPredictor本体
- `configs/infer/infer.yaml`: デフォルト設定ファイル
- `demo/config_template.yaml`: 設定テンプレート 