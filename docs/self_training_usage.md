# 自己学習機能の使用方法

このドキュメントでは、Tennis-Analyzerの自己学習機能の使用方法を説明します。

## 概要

自己学習機能は、ラベル付きデータと未ラベルデータを組み合わせてモデルのパフォーマンスを向上させる手法です。この機能は以下のコンポーネントで構成されています：

1. ボール検出の自己学習
2. コート検出の自己学習

## 未ラベルデータの準備

### 未ラベルフレームの管理

未ラベルフレームは以下のディレクトリ構造で管理します：

```
datasets/
  ball/
    unlabeled/
      images/        # 未ラベルのフレーム画像
      coco_annotations_unlabeled.json  # 画像情報のみを含むCOCO形式アノテーション
  court/
    unlabeled/
      images/        # 未ラベルのフレーム画像
      coco_annotations_unlabeled.json  # 画像情報のみを含むCOCO形式アノテーション
```

未ラベルのアノテーションファイルは、画像情報のみを含む簡易なCOCO形式のJSONファイルです。以下はその例です：

```json
{
  "images": [
    {"id": 1, "file_name": "frame_00001.jpg", "width": 1280, "height": 720},
    {"id": 2, "file_name": "frame_00002.jpg", "width": 1280, "height": 720},
    ...
  ],
  "annotations": [],
  "categories": []
}
```

### 新しいフレームを追加する方法

新しいフレームを追加するには以下の手順を実行します：

1. フレーム画像を適切なディレクトリに配置します：
   - ボール検出: `datasets/ball/unlabeled/images/`
   - コート検出: `datasets/court/unlabeled/images/`

2. 対応するCOCO形式のアノテーションファイルを更新または作成します：
   - ボール検出: `datasets/ball/unlabeled/coco_annotations_unlabeled.json`
   - コート検出: `datasets/court/unlabeled/coco_annotations_unlabeled.json`

## 設定ファイル

### ボール検出の自己学習

ボール検出の自己学習の設定は `configs/train/ball/self_training_config.yaml` で行います。

重要なパラメータ：

- `litdatamodule.unlabeled_annotation_file`: 未ラベルデータのアノテーションファイルのパス
- `litdatamodule.unlabeled_image_root`: 未ラベル画像のルートディレクトリ
- `litdatamodule.pseudo_label_dir`: 擬似ラベルの保存ディレクトリ
- `self_training.max_cycles`: 自己学習の最大サイクル数
- `self_training.confidence_threshold`: 擬似ラベルとして採用する信頼度の閾値

### コート検出の自己学習

コート検出の自己学習の設定は `configs/train/court/self_training_config.yaml` で行います。

重要なパラメータ：

- `litdatamodule.unlabeled_annotation_file`: 未ラベルデータのアノテーションファイルのパス
- `litdatamodule.unlabeled_image_root`: 未ラベル画像のルートディレクトリ
- `litdatamodule.pseudo_label_dir`: 擬似ラベルの保存ディレクトリ
- `self_training.max_cycles`: 自己学習の最大サイクル数
- `self_training.confidence_threshold`: 擬似ラベルとして採用する信頼度の閾値

## 実行方法

### ボール検出の自己学習

```bash
python scripts/train/train_ball_self_training.py
```

### コート検出の自己学習

```bash
python scripts/train/train_court_self_training.py
```

## 出力結果

自己学習の結果は以下のディレクトリに保存されます：

- ボール検出: `outputs/ball/self_training/`
- コート検出: `outputs/court/self_training/`

各サイクルの擬似ラベルは以下のファイルに保存されます：

- ボール検出: `outputs/ball/self_training/pseudo_labels/pseudo_labels_cycle_X.json`
- コート検出: `outputs/court/self_training/pseudo_labels/pseudo_labels_cycle_X.json`

（Xはサイクル番号） 