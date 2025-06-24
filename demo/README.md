# 🎾 Tennis Analysis System - Demo Applications

テニス動画・画像解析システムのデモンストレーション用アプリケーション集です。Gradioを使用したインタラクティブなWebインターフェースを提供し、AI-Powered テニス分析の機能を体験できます。

## 📋 目次

- [利用可能デモ](#利用可能デモ)
- [クイックスタート](#クイックスタート)
- [システム要件](#システム要件)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [デモ詳細](#デモ詳細)
- [トラブルシューティング](#トラブルシューティング)

## 🎮 利用可能デモ

### 🚀 新世代デモ (推奨)

| デモ | ファイル | 説明 | 特徴 | ポート |
|------|----------|------|------|--------|
| **Simple Demo** | `simple_demo.py` | 軽量ボール検出デモ | 🚀 ワンクリック解析<br/>📱 モバイル対応<br/>⚡ 高速処理 | 7861 |
| **Full System** | `tennis_analysis_app.py` | 統合型フル機能システム | 🎯 ボール検出<br/>📁 バッチ処理<br/>📊 統計分析<br/>⚙️ 詳細設定 | 7860 |

### 🔧 レガシーデモ

| デモ | ファイル | 説明 | 特徴 | ポート |
|------|----------|------|------|--------|
| **Ball Detection** | `ball.py` | レガシー版ボール検出 | 🎾 ボール検出<br/>📈 軌跡追跡<br/>🔧 異常値除去 | 7862 |
| **Court Detection** | `court.py` | コート検出・解析 | 🏟️ コート認識<br/>📍 キーポイント検出<br/>🎨 ヒートマップ | 7863 |
| **Player Detection** | `player.py` | プレーヤー検出・姿勢推定 | 👥 プレーヤー検出<br/>🤸 姿勢推定<br/>📊 動作解析 | 7864 |
| **Event Analysis** | `event.py` | イベント検出・分析 | 🏆 イベント検出<br/>⏰ タイムライン<br/>📊 統計分析 | 7865 |
| **Pose Analysis** | `pose.py` | 姿勢解析・動作分析 | 🤸 姿勢推定<br/>📐 関節角度<br/>🏃 動作解析 | 7866 |

## 🚀 クイックスタート

### 1. デモランチャー使用（推奨）

```bash
# インタラクティブ選択
python demo/run_demo.py

# 直接起動
python demo/run_demo.py simple    # シンプルデモ
python demo/run_demo.py full      # フル機能デモ
```

### 2. 直接起動

```bash
# シンプルデモ（軽量・初心者向け）
python demo/simple_demo.py

# フル機能デモ（上級者向け）
python demo/tennis_analysis_app.py

# レガシーデモ
python demo/ball.py
python demo/court.py
```

### 3. ブラウザでアクセス

デモ起動後、以下のURLにアクセス：
- Simple Demo: http://localhost:7861
- Full System: http://localhost:7860
- 各種レガシーデモ: 対応ポート番号を確認

## 💻 システム要件

### 基本要件

- **Python**: 3.8以上
- **メモリ**: 8GB以上推奨
- **ストレージ**: 5GB以上の空き容量

### 推奨環境

- **GPU**: CUDA対応GPU（RTX 3060以上）
- **メモリ**: 16GB以上
- **CPU**: Intel i7 / AMD Ryzen 7以上

### 対応プラットフォーム

- ✅ Windows 10/11
- ✅ macOS 10.15以上  
- ✅ Ubuntu 18.04以上
- ✅ Docker環境

## 🔧 インストール

### 1. 依存関係インストール

```bash
# 基本パッケージ
pip install -r requirements.txt

# 追加デモ依存関係
pip install gradio plotly pandas
```

### 2. モデルファイル配置

```bash
# checkpointsディレクトリ作成
mkdir -p checkpoints/ball
mkdir -p checkpoints/court  
mkdir -p checkpoints/player

# モデルファイルを配置
# checkpoints/ball/lite_tracknet.ckpt
# checkpoints/ball/wasb_sbdt.pth
# checkpoints/court/lite_tracknet.ckpt
# checkpoints/player/rt_detr.ckpt
```

### 3. システム確認

```bash
# システム状態確認
python demo/run_demo.py --status

# 利用可能デモ確認
python demo/run_demo.py --list
```

## 📚 使用方法

### 基本的な流れ

1. **デモ選択**: 用途に応じてデモを選択
2. **ファイルアップロード**: 動画・画像をアップロード
3. **設定調整**: 必要に応じてパラメータ調整
4. **解析実行**: 解析ボタンをクリック
5. **結果確認**: 処理済み動画・統計情報を確認

### 推奨デモ選択

| 用途 | 推奨デモ | 理由 |
|------|----------|------|
| **初回体験** | Simple Demo | 軽量・簡単・高速 |
| **本格利用** | Full System | 全機能・詳細設定 |
| **研究開発** | レガシーデモ | 個別機能・カスタマイズ |
| **バッチ処理** | Full System | 複数ファイル一括処理 |

## 🎯 デモ詳細

### Simple Demo (`simple_demo.py`)

**特徴**: 軽量で使いやすいボール検出デモ

**主な機能**:
- 🎾 ワンクリックボール検出・軌跡解析
- 📱 モバイル対応レスポンシブUI
- ⚡ 軽量設定による高速処理
- 📊 基本統計情報表示

**使用場面**:
- 初めてシステムを試す場合
- 簡単な動画解析が必要な場合
- 軽量環境での動作確認

**操作手順**:
1. Video Analysisタブを選択
2. 動画ファイルをアップロード
3. ボール半径・軌跡長を調整（オプション）
4. 「解析開始」ボタンをクリック
5. 結果動画・統計情報を確認

### Full System (`tennis_analysis_app.py`)

**特徴**: 統合型フル機能テニス解析システム

**主な機能**:
- 🎯 高精度ボール検出・軌跡解析
- 📁 複数動画の一括バッチ処理
- 📊 詳細統計分析・レポート生成
- ⚙️ 豊富な設定オプション
- 🎨 高品質可視化・カスタマイズ

**使用場面**:
- 本格的なテニス動画解析
- 大量動画の一括処理
- 詳細な統計分析が必要
- カスタム設定での処理

**操作手順**:
1. Ball Detectionタブで単一動画解析
2. Batch Processingタブで複数動画処理
3. Settings & Infoタブでシステム確認
4. 各種パラメータの詳細調整
5. 結果ダウンロード・レポート確認

### Legacy Demos

**Ball Detection (`ball.py`)**:
- オリジナルボール検出実装
- 異常値除去機能
- 軌跡可視化

**Court Detection (`court.py`)**:
- コートキーポイント検出
- ヒートマップ可視化
- コート認識精度評価

**Player Detection (`player.py`)**:
- プレーヤー検出・追跡
- バウンディングボックス表示
- 複数プレーヤー対応

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. モデルファイルが見つからない

**症状**: 「モデルファイルが見つかりません」エラー

**解決方法**:
```bash
# システム状態確認
python demo/run_demo.py --status

# checkpointsディレクトリにモデルファイルを配置
# 正しいパス: checkpoints/ball/model.ckpt
```

#### 2. GPU が認識されない

**症状**: CPU処理になってしまう

**解決方法**:
```bash
# CUDA環境確認
python -c "import torch; print(torch.cuda.is_available())"

# CUDAドライバー・PyTorchの再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. メモリ不足エラー

**症状**: 「CUDA out of memory」エラー

**解決方法**:
- Simple Demoを使用（軽量設定）
- バッチサイズを削減
- CPUモードで実行
- 動画解像度を下げる

#### 4. デモが起動しない

**症状**: デモファイルが見つからない・実行エラー

**解決方法**:
```bash
# 依存関係確認・インストール
pip install gradio plotly pandas

# ファイル存在確認
python demo/run_demo.py --list

# デバッグモードで実行
python demo/run_demo.py simple --debug
```

#### 5. 処理が遅い

**症状**: 動画処理に時間がかかりすぎる

**解決方法**:
- GPU利用可能か確認
- メモリ効率設定を使用
- 動画サイズ・長さを短縮
- バッチサイズを調整

### パフォーマンス最適化

#### GPU使用時
```python
# 推奨設定
config_preset = "high_performance"
batch_size = 16
num_workers = 8
```

#### CPU使用時
```python
# 推奨設定  
config_preset = "memory_efficient"
batch_size = 4
num_workers = 2
```

## 📞 サポート・コミュニティ

### サポートリソース

- **ドキュメント**: [docs/](../docs/)
- **API リファレンス**: [src/predictor/](../src/predictor/)
- **設定例**: [configs/](../configs/)

### デバッグ情報収集

問題報告時は以下の情報を提供してください：

```bash
# システム情報取得
python demo/run_demo.py --status > system_info.txt

# エラーログ取得
python demo/simple_demo.py 2>&1 | tee error_log.txt
```

### 貢献・開発

デモの改善・新機能追加は以下の方法で：

1. **Issue作成**: バグ報告・機能要望
2. **Pull Request**: コード貢献
3. **ドキュメント**: 使用例・チュートリアル

---

<div align="center">

**🎾 Tennis Analysis System Demo**  
*AI-Powered Tennis Video Analysis Platform*

[Home](../README.md) | [Documentation](../docs/) | [API](../src/predictor/) | [Examples](./examples/)

</div> 