# ボール検出 Self-Training 実装

このディレクトリには、テニスボール検出タスクのためのSelf-Training（自己学習）実装が含まれています。

## 概要

Self-Trainingは、ラベル付きデータと大量のラベルなしデータを活用して、モデルの性能を向上させる半教師あり学習手法です。本実装では以下の特徴を持ちます：

- **擬似ラベル生成**: 高信頼度の予測から擬似ラベルを自動生成
- **軌跡追跡**: テニスボールの物理的な軌跡を考慮した擬似ラベルの洗練
- **反復的な改善**: 複数サイクルによる段階的な性能向上
- **PyTorch Lightning統合**: 効率的な学習とスケーラビリティ

## アーキテクチャ

### 主要コンポーネント

1. **BallSelfTrainingCycle** (`self_training_cycle.py`)
   - Self-Trainingの全体的なフローを管理
   - 擬似ラベルの生成、保存、モデルの再学習を制御
   - Average Precisionなどの評価メトリクスを使用

2. **BallTrajectoryTracker** (`trajectory_tracker.py`)
   - ボールの軌跡を追跡し、時間的に一貫した擬似ラベルを生成
   - 欠落フレームの補間、外れ値の除去、軌跡の平滑化を実行
   - 物理的に妥当なボールの動きを保証

3. **PseudoLabeledSequenceDataset** (`../dataset/pseudo_labeled_seq_dataset.py`)
   - 擬似ラベルをサポートするデータセット実装
   - オリジナルラベルと擬似ラベルを統合して学習

4. **SelfTrainingLitModule** (`../lit_module/self_training_lit_module.py`)
   - 擬似ラベルの重み付けをサポートするLightningModule
   - ヒートマップ回帰と座標回帰の両方に対応

## 使用方法

### 1. 基本的な実行

```bash
# デフォルト設定でSelf-Trainingを実行
python scripts/train/train_ball_self_training.py \
    initial_checkpoint=path/to/checkpoint.ckpt
```

### 2. パラメータの調整

```bash
# 信頼度閾値を変更
python scripts/train/train_ball_self_training.py \
    initial_checkpoint=path/to/checkpoint.ckpt \
    self_training.confidence_threshold=0.8

# 最大サイクル数を変更
python scripts/train/train_ball_self_training.py \
    initial_checkpoint=path/to/checkpoint.ckpt \
    self_training.max_cycles=5

# 擬似ラベルの重みを調整
python scripts/train/train_ball_self_training.py \
    initial_checkpoint=path/to/checkpoint.ckpt \
    self_training.pseudo_label_weight=0.3
```

### 3. 軌跡追跡の設定

```bash
# 軌跡追跡を無効化
python scripts/train/train_ball_self_training.py \
    initial_checkpoint=path/to/checkpoint.ckpt \
    self_training.use_trajectory_tracking=false

# 軌跡追跡パラメータを調整
python scripts/train/train_ball_self_training.py \
    initial_checkpoint=path/to/checkpoint.ckpt \
    self_training.trajectory_params.max_ball_speed=150.0 \
    self_training.trajectory_params.min_trajectory_length=10
```

## 設定ファイル

### メイン設定 (`configs/train/ball/self_training/config.yaml`)

```yaml
defaults:
  - /train/ball/model: _cat_frames
  - litmodule: self_training_heatmap
  - litdatamodule: ball_self_training_data
  - trainer: self_training
  - callbacks: self_training
  - self_training: default

version: ${model.name}_self_training_${self_training.confidence_threshold}
seed: 42
initial_checkpoint: null  # 必須：コマンドラインから指定
```

### Self-Training設定 (`configs/train/ball/self_training/self_training/default.yaml`)

```yaml
confidence_threshold: 0.7
max_cycles: 3
pseudo_label_weight: 0.5

trajectory_params:
  temporal_window: 9
  max_trajectory_gap: 5
  min_trajectory_length: 7
  interpolation_method: "quadratic"
  smoothing_window: 5
  max_ball_speed: 100.0

use_trajectory_tracking: true
min_pseudo_labels: 10
save_dir: "outputs/ball/self_training/${version}"
```

## アルゴリズムの詳細

### Self-Trainingのフロー

1. **初期化**
   - 事前学習済みモデルを読み込み
   - ラベル付き/ラベルなしデータセットを準備

2. **擬似ラベル生成**
   - ラベルなしデータに対して予測を実行
   - 信頼度閾値を超える予測を擬似ラベルとして採用

3. **軌跡追跡による洗練**
   - 時系列でボールの軌跡を構築
   - 物理的に妥当でない検出を除去
   - 欠落フレームを補間

4. **モデルの再学習**
   - オリジナルラベルと擬似ラベルを組み合わせて学習
   - 擬似ラベルには重み付けを適用

5. **評価と反復**
   - 検証セットで性能を評価
   - 改善が見られる限りサイクルを継続

### 軌跡追跡アルゴリズム

1. **高信頼度ポイントの抽出**
   - 信頼度閾値を超える検出を収集

2. **外れ値の除去**
   - 物理的に不可能な速度の検出を除去

3. **軌跡の構築**
   - 時間的に近いポイントを接続
   - 最小長を満たす軌跡のみを保持

4. **補間と平滑化**
   - Savitzky-Golayフィルタで軌跡を平滑化
   - 二次/三次補間で欠落フレームを補完

## テスト

```bash
# ユニットテストの実行
pytest tests/test_ball_self_training.py -v

# 特定のテストのみ実行
pytest tests/test_ball_self_training.py::TestBallSelfTraining::test_trajectory_tracking -v
```

## トラブルシューティング

### 擬似ラベルが生成されない
- 信頼度閾値を下げる: `self_training.confidence_threshold=0.5`
- モデルの品質を確認（初期チェックポイント）

### メモリ不足
- バッチサイズを削減: `litdatamodule.batch_size=8`
- ワーカー数を削減: `litdatamodule.num_workers=2`

### 軌跡追跡が機能しない
- 最大ボール速度を調整: `self_training.trajectory_params.max_ball_speed=200.0`
- 最小軌跡長を短くする: `self_training.trajectory_params.min_trajectory_length=5`

## 今後の改善点

1. **動的な信頼度閾値**
   - サイクルごとに閾値を自動調整

2. **マルチスケール軌跡追跡**
   - 異なる時間スケールで軌跡を分析

3. **アクティブラーニング統合**
   - 最も有益なサンプルを優先的にラベル付け

4. **分散Self-Training**
   - 複数GPUでの並列処理サポート 