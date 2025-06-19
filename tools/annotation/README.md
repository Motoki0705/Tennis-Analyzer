# テニスイベント検出用 手動クリップベースアノテーションシステム

手動で準備されたテニス動画クリップに対して、効率的なアノテーション作業を行うためのシステムです。モデルによる自動生成に依存せず、完全に手動制御されたワークフローで高品質なデータセットを生成できます。

## システム概要

本システムは以下の3つのコンポーネントで構成されています：

1. **空アノテーション生成** (`generate_empty_annotations.py`) - 手動配置されたクリップから空のアノテーションJSONを生成
2. **Webアノテーションツール** - ブラウザ上でアノテーションの作成・修正を行う
3. **マージスクリプト** (`merge_to_coco.py`) - 完成したアノテーションを最終的なCOCO形式データセットに統合

## 事前準備

### 1. 依存関係のインストール

```bash
# Python 依存関係
pip install -r requirements.txt

# フロントエンド依存関係（Node.js が必要）
cd web_app/frontend
npm install
```

### 2. 手動クリップの準備

動画編集ツールまたはFFmpegを使用して、元のテニス動画から目的のイベントを含む短いクリップを抽出してください。

**対応形式**: mp4, avi, mov, mkv, flv, wmv

**FFmpegを使用したクリップ抽出例:**
```bash
# ゲーム開始から30秒〜34秒のクリップを抽出
ffmpeg -i game1.mp4 -ss 00:00:30 -t 00:00:04 -c copy clip_001.mp4

# より多くのクリップを抽出
ffmpeg -i game1.mp4 -ss 00:01:15 -t 00:00:04 -c copy clip_002.mp4
ffmpeg -i game1.mp4 -ss 00:02:45 -t 00:00:04 -c copy clip_003.mp4
```

## 使用方法

### ステップ1: 手動クリップの配置と空アノテーション生成

1. クリップファイルを配置先ディレクトリに移動：
```bash
mkdir -p ./datasets/annotation_workspace/clips/
cp clip_*.mp4 ./datasets/annotation_workspace/clips/
```

2. 空のアノテーションJSONファイルを自動生成：
```bash
python generate_empty_annotations.py \
    --clips_dir ./datasets/annotation_workspace/clips \
    --annotations_dir ./datasets/annotation_workspace/annotations \
    --source_video ./raw_videos/game1.mp4 \
    --validate
```

**パラメータ:**
- `--clips_dir`: クリップファイルが配置されたディレクトリ
- `--annotations_dir`: アノテーションJSONの出力ディレクトリ
- `--source_video`: 元動画のパス（記録用、オプション）
- `--validate`: 生成後に妥当性チェックを実行

**出力:**
- `clips/`: 手動配置された動画クリップファイル
- `annotations/`: 空のアノテーションJSON (`clip_001.json`, `clip_002.json`, ...)

### ステップ2: Webアノテーションツールでのアノテーション作業

#### バックエンドサーバーの起動

```bash
python web_app/app.py --port 8000 --data_dir ./datasets/annotation_workspace
```

#### フロントエンドの起動

```bash
cd web_app/frontend
npm start
```

ブラウザで `http://localhost:3000` にアクセスしてアノテーション作業を開始します。

#### アノテーション操作方法

**キーボードショートカット:**
- `Space`: 再生/一時停止
- `←` / `→`: 1フレーム移動
- `Shift` + `←` / `→`: 10フレーム移動
- `H`: 現在のフレームを「Hit」に設定
- `B`: 現在のフレームを「Bounce」に設定
- `N`: 現在のフレームのイベントを解除
- `V`: ボールの可視性 (visibility) をトグル
- `I`: 範囲選択後のキーポイント補間を実行

**インタラクティブ操作:**
- **キーポイント編集**: キャンバス上をクリックしてボール位置を設定
- **イベント編集**: タイムライン上のマーカーをドラッグして移動
- **区間補間**: `Shift`を押しながら範囲選択後、`I`キーで線形補間

### ステップ3: 最終データセットの生成

アノテーションが完了したファイルをCOCO形式に統合します。

```bash
python merge_to_coco.py \
    --input_dir ./datasets/annotation_workspace/annotations \
    --output_file ./datasets/tennis_events_dataset.json \
    --stats_file ./datasets/dataset_statistics.json \
    --cleanup
```

**パラメータ:**
- `--input_dir`: アノテーションJSONファイルのディレクトリ
- `--output_file`: 出力するCOCO形式JSONファイル
- `--stats_file`: データセット統計情報の出力ファイル（オプション）
- `--cleanup`: 実行後に一時ファイルを削除

## データフォーマット

### 中間アノテーションフォーマット

各クリップに対して以下の形式のJSONファイルが生成されます：

```json
{
  "clip_info": {
    "source_video": "path/to/game1.mp4",
    "clip_name": "clip_001",
    "clip_path": "data/clips/clip_001.mp4",
    "fps": 30.0,
    "width": 1280,
    "height": 720
  },
  "frames": [
    {
      "frame_number": 0,
      "ball": {
        "keypoint": [527.0, 384.0],
        "visibility": 2,
        "is_interpolated": false
      },
      "event_status": 0
    }
  ]
}
```

### 最終COCO形式フォーマット

マージスクリプトによって以下のCOCO形式に変換されます：

```json
{
  "info": { "description": "Tennis Event Detection Dataset", ... },
  "licenses": [...],
  "categories": [{"id": 1, "name": "ball", ...}],
  "images": [
    {
      "id": 1,
      "file_name": "frame_000001.jpg",
      "width": 1280,
      "height": 720,
      "clip_name": "clip_001",
      "frame_number": 0
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [527.0, 384.0, 2],
      "bbox": [522.0, 379.0, 10, 10],
      "event_status": 0,
      "is_interpolated": false
    }
  ]
}
```

## 統合実行スクリプト

より簡単に使用するための統合実行スクリプトも提供されています：

```bash
# 環境セットアップ
./run_annotation_system.sh setup

# 手動配置クリップから空アノテーション生成
./run_annotation_system.sh prepare

# Webアノテーションサーバー起動
./run_annotation_system.sh server

# COCO形式データセット生成
./run_annotation_system.sh merge

# 完全ワークフロー実行（クリップ配置後）
./run_annotation_system.sh full
```

## トラブルシューティング

### よくある問題

1. **クリップファイルが見つからない**
   - `clips/` ディレクトリにクリップファイルが正しく配置されているか確認
   - 対応形式（mp4, avi, mov, mkv, flv, wmv）であることを確認

2. **動画ファイルが読み込めない**
   - OpenCVでサポートされている形式であることを確認
   - ファイルパスに日本語が含まれていないか確認
   - ファイルが破損していないか確認

3. **Webアプリケーションにアクセスできない**
   - バックエンドサーバーが起動しているか確認（ポート8000）
   - フロントエンドサーバーが起動しているか確認（ポート3000）
   - ファイアウォールの設定を確認

4. **アノテーション保存に失敗する**
   - 出力ディレクトリの書き込み権限を確認
   - ディスク容量を確認
   - JSONファイルの形式が正しいか確認

5. **空アノテーション生成に失敗する**
   - 動画ファイルのメタデータが正しく取得できているか確認
   - FFmpegがインストールされているか確認

### ログの確認

各コンポーネントは詳細なログを出力します：

```bash
# 空アノテーション生成のログ
python generate_empty_annotations.py --verbose ...

# Webアプリケーションのログ
python web_app/app.py --log-level DEBUG ...

# マージスクリプトのログ
python merge_to_coco.py --verbose ...
```

## パフォーマンス最適化

### 推奨システム要件

- **CPU**: Intel i5 以上 または AMD Ryzen 5 以上
- **メモリ**: 8GB 以上（16GB推奨）
- **ストレージ**: SSD推奨（動画ファイルの読み込み速度向上）
- **ブラウザ**: Chrome, Firefox, Safari の最新版

### 最適化のヒント

1. **動画ファイルサイズ**: 適切な圧縮率でクリップサイズを調整
2. **同時アクセス数**: 複数ユーザーでの同時アノテーション作業時のリソース配分
3. **ディスク I/O**: 高速SSDの使用でフレーム読み書き速度を向上
4. **ネットワーク**: ローカルネットワークでの使用により動画読み込み速度を最適化

## 今後の展開

- Player や Court のアノテーション機能追加
- 複数アノテータによる協調作業機能
- アノテーション品質管理機能
- クラウド環境での大規模運用
- 自動品質チェック機能の追加

## ライセンス

本プロジェクトは[適切なライセンス]の下で公開されています。

## 貢献方法

バグ報告や機能要望は Issue よりお知らせください。
プルリクエストも歓迎いたします。 