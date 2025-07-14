# Tennis Systems - Docker Development Environment

CPU専用のPython 3.11開発環境です。GPU不要でローカル開発・テスト・デモ実行が可能です。

## 🚀 クイックスタート

### 1. イメージのビルド
```bash
cd /mnt/c/Users/kamim/code/tennis_systems
docker-compose -f docker/docker-compose.yml build
```

### 2. 開発環境の起動
```bash
# 対話的開発環境
docker-compose -f docker/docker-compose.yml run --rm tennis-dev bash

# バックグラウンド実行
docker-compose -f docker/docker-compose.yml up -d tennis-dev
```

### 3. コマンド実行
```bash
# デモ実行
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python demo/ball.py

# テスト実行
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python -m pytest tests/

# モデル訓練
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python -m src.ball.api.train --config-name lite_tracknet_focal
```

## 📁 マウント構成

| ホストパス | コンテナパス | 用途 |
|------------|--------------|------|
| `./` | `/workspace` | プロジェクト全体 |
| `./datasets` | `/workspace/datasets` | 訓練・テストデータ |
| `./checkpoints` | `/workspace/checkpoints` | モデルウェイト |
| `./outputs` | `/workspace/outputs` | 実行結果 |

## 🛠️ 主な使用方法

### デモ実行
```bash
# ボール検出デモ
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python demo/ball.py

# コート検出デモ
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python demo/court.py

# 統合イベント検出デモ
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python demo/event.py
```

### テスト実行
```bash
# 全テスト
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python -m pytest tests/

# 特定テスト
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python -m pytest tests/infer_model_instantiate/
```

### 開発作業
```bash
# コンテナに入る
docker-compose -f docker/docker-compose.yml run --rm tennis-dev bash

# Pythonインタープリタ
docker-compose -f docker/docker-compose.yml run --rm tennis-dev python
```

## 🔧 環境仕様

- **Base**: Ubuntu 22.04
- **Python**: 3.11
- **PyTorch**: 2.7.1+cpu (CPU専用)
- **主要ライブラリ**: 
  - PyTorch Lightning 2.5.2
  - OpenCV 4.12.0
  - Transformers 4.52.4
  - Gradio 5.35.0

## 📝 注意事項

- GPU機能は含まれていません
- WindowsのWSL2環境での動作を想定
- 大容量データセットは事前にホストに配置してください
- 初回ビルド時は依存関係のダウンロードに時間がかかります

## 🔄 メンテナンス

### イメージの再ビルド
```bash
docker-compose -f docker/docker-compose.yml build --no-cache
```

### コンテナのクリーンアップ
```bash
docker-compose -f docker/docker-compose.yml down
docker system prune -f
```