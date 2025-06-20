# Tennis Systems - 包括的テニス分析プラットフォーム

本プロジェクトは、AI・機械学習技術を活用したテニス分析システムです。ボール追跡、コート検出、プレイヤー検出、ポーズ推定、イベント検出の5つの主要機能を統合し、テニス動画の包括的な分析を実現します。

## 🎾 主要機能

### 1. ボール追跡システム
- **高精度ボール検出**: 優秀な外部ボールトラッカー（[WASB-SBDT](https://github.com/starashima/WASB-SBDT_sandbox)）を統合
- **軌跡平滑化**: 異常値除去・補間機能による滑らかな軌跡生成
- **リアルタイム追跡**: オンライン・オフライン両対応

### 2. コート検出システム
- **キーポイント検出**: テニスコートの15個の主要ポイントを自動検出
- **ヒートマップ可視化**: 検出確率をカラーマップで直感的に表示
- **高精度モデル**: 独自のLite-TrackNetアーキテクチャを採用

### 3. プレイヤー検出システム
- **物体検出**: RT-DETRベースのファインチューニング済みモデル
- **高速推論**: リアルタイム処理可能な軽量アーキテクチャ
- **高精度認識**: テニス特化のトレーニングデータによる最適化

### 4. ポーズ推定システム
- **2段階パイプライン**: プレイヤー検出 → ポーズ推定の効率的な処理
- **17キーポイント**: COCOフォーマット準拠の詳細な姿勢情報
- **リアルタイム描画**: 骨格線とキーポイントの直感的な可視化

### 5. イベント検出システム
- **統合分析**: ボール・コート・プレイヤー・ポーズ情報を統合
- **イベント分類**: ヒット・バウンスの自動検出
- **心電図風可視化**: 確率推移をリアルタイムで表示
- **バッチ処理**: 大規模動画に対応した効率的な処理

## 🚀 デモシステム

`demo/` ディレクトリに各機能の独立したデモアプリケーションを用意しています：

- **`demo/ball.py`**: ボール追跡デモ（Gradio UI）
- **`demo/court.py`**: コート検出デモ（Gradio UI）
- **`demo/player.py`**: プレイヤー検出デモ（Gradio UI）
- **`demo/pose.py`**: ポーズ推定デモ（Gradio UI）
- **`demo/event.py`**: 統合イベント検出デモ（Gradio UI）

### デモの実行方法
```bash
# 例：ボール追跡デモの実行
python demo/ball.py

# 例：統合イベント検出デモの実行
python demo/event.py
```

## 📁 プロジェクト構成

```
tennis_systems/
├── demo/                    # 各機能のデモアプリケーション
├── src/                     # メインソースコード
│   ├── ball/               # ボール検出・追跡モジュール
│   ├── court/              # コート検出モジュール
│   ├── event/              # イベント検出モジュール
│   ├── player/             # プレイヤー検出モジュール
│   ├── pose/               # ポーズ推定モジュール
│   └── utils/              # 共通ユーティリティ
├── ball_tracker/           # 外部ボールトラッカー統合
├── configs/                # モデル・データセット設定
├── tools/                  # 開発支援ツール群
│   ├── annotation/         # Webベースアノテーションシステム
│   ├── check/              # データ検証ツール
│   ├── collect/            # データ収集ツール
│   └── video_clipper/      # 動画編集ツール
├── tests/                  # テストスイート
├── checkpoints/            # 学習済みモデル
├── datasets/               # データセット
└── samples/                # サンプル動画
```

## 🔧 インストール

### システム要件
- **Python**: 3.8以上
- **CUDA**: 11.0以上（GPU使用時）
- **FFmpeg**: 動画処理に必要
- **Node.js**: アノテーションツール使用時

### 基本インストール
```bash
# リポジトリのクローン
git clone <repository-url>
cd tennis_systems

# Python依存関係のインストール
pip install -r requirements.txt

# GPU使用時は公式サイトからPyTorchを先にインストール推奨
# https://pytorch.org/get-started/locally/
```

### アノテーションツールのセットアップ
```bash
# フロントエンド依存関係のインストール
cd tools/annotation/web_app/frontend
npm install
cd ../../../..

# アノテーションシステムの起動
cd tools/annotation
./run_annotation_system.sh setup
```

## 🎯 使用方法

### 1. デモアプリケーションでの試用
```bash
# 統合イベント検出（推奨）
python demo/event.py

# 個別機能のテスト
python demo/ball.py     # ボール追跡
python demo/court.py    # コート検出
python demo/player.py   # プレイヤー検出
python demo/pose.py     # ポーズ推定
```

### 2. モデルのトレーニング
```bash
# ボール検出モデルのトレーニング
bash scripts/train/ball/lite_tracknet_focal.sh

# プレイヤー検出モデルのトレーニング
bash scripts/train/player/rt_detr.sh

# コート検出モデルのトレーニング
bash scripts/train/court/lite_tracknet_focal.sh

# イベント検出モデルのトレーニング
bash scripts/train/event/event_transformer.sh
```

### 3. アノテーションツールの使用
```bash
cd tools/annotation

# Webアノテーションシステムの起動
./run_annotation_system.sh start

# ブラウザで http://localhost:8000 にアクセス
```

## 🏗️ システムアーキテクチャ

### ML パイプライン
1. **データ前処理**: OpenCV、Albumentation による画像・動画処理
2. **モデル**: PyTorch Lightning ベースの統一アーキテクチャ
3. **推論**: バッチ処理・リアルタイム処理両対応
4. **後処理**: 平滑化・補間・可視化

### モデル詳細
- **ボール検出**: Lite-TrackNet + Focal Loss
- **コート検出**: Lite-TrackNet (15キーポイント)
- **プレイヤー検出**: RT-DETR v2 (ファインチューニング)
- **ポーズ推定**: ViT-Pose (HuggingFace)
- **イベント検出**: Transformer v2 (統合特徴量)

## 🛠️ 開発ツール

### アノテーションシステム
- **Webベースインターフェース**: React + FastAPI
- **リアルタイム動画プレビュー**: フレーム単位の精密制御
- **協調作業**: 複数アノテータ対応
- **自動品質チェック**: データ整合性検証

### 評価・検証ツール
- **モデル性能評価**: `tests/` ディレクトリ
- **データ整合性チェック**: `tools/check/`
- **統計情報生成**: `tools/collect/`

### 動画処理ツール
- **クリップ抽出**: `tools/video_clipper/`
- **フレーム抽出**: バッチ処理対応
- **メタデータ管理**: 自動ファイル整理

## 📊 パフォーマンス

### 処理速度（RTX 4090基準）
- **ボール追跡**: ~60 FPS
- **プレイヤー検出**: ~45 FPS
- **ポーズ推定**: ~30 FPS
- **統合処理**: ~25 FPS

### 精度指標
- **ボール検出**: mAP@0.5 > 0.85
- **プレイヤー検出**: mAP@0.5 > 0.90
- **イベント検出**: F1-Score > 0.80

## 🔍 トラブルシューティング

### よくある問題

**1. CUDA メモリ不足**
```bash
# バッチサイズを調整
export CUDA_VISIBLE_DEVICES=0
# または config ファイルでバッチサイズを減らす
```

**2. モデルファイルが見つからない**
```bash
# checkpoints ディレクトリの確認
ls checkpoints/*/
# 必要に応じてモデルを再ダウンロード
```

**3. FFmpeg エラー**
```bash
# FFmpeg のインストール確認
ffmpeg -version
# Windowsの場合は公式サイトからダウンロード
```

### ログ確認
```bash
# 詳細ログの有効化
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python demo/event.py --log-level DEBUG
```

## 📚 ドキュメント

詳細なドキュメントは `docs/` ディレクトリを参照してください：
- **`docs/models_overview.md`**: モデルアーキテクチャの詳細
- **`docs/model.md`**: 技術仕様書
- **`tools/annotation/README.md`**: アノテーションシステム詳細ガイド

## 🤝 貢献

### 開発ガイドライン
1. **コードスタイル**: PEP 8 準拠
2. **ドキュメント**: Google スタイル Docstring
3. **テスト**: pytest による単体テスト
4. **設定管理**: Hydra による階層化設定

### 貢献方法
1. Issue の作成（バグ報告・機能要望）
2. Fork & Pull Request
3. コードレビュー
4. ドキュメント更新

## 📄 ライセンス

本プロジェクトは適切なライセンスの下で公開されています。
外部ライブラリ（ball_tracker）については元リポジトリのライセンスに従います。

## 🏷️ タグ・キーワード

`tennis` `sports-analytics` `computer-vision` `pytorch` `object-detection` `pose-estimation` `event-detection` `machine-learning` `deep-learning` `gradio` `annotation-tool`

---

**更新日**: 2024年
**メンテナー**: Tennis Systems Development Team 