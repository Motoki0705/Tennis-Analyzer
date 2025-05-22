# テニス自動分析システム

このプロジェクトは、テニスの試合動画を包括的に分析するためのAIパワードツールキットです。ディープラーニングモデルを活用して、テニスボール、プレーヤー、コートライン、プレーヤーのポーズなどの重要な要素を自動的に検出・追跡します。このシステムは、動画から貴重な洞察を抽出し、注釈付きの出力を生成することができます。

## 機能

このシステムは、詳細なスポーツ動画分析のための様々な機能を提供します：

* **ボールの検出と追跡：** 動画全体を通してボールの軌道を正確に識別し、追跡します。
* **コートラインの検出：** プレーエリアを理解するために、コートラインをセグメント化し識別します。
* **プレーヤーの検出と追跡：** コート上のプレーヤーを検出し、その動きを追跡することができます。
* **プレーヤーのポーズ推定：** プレーヤーのポーズを推定し、姿勢やアクションに関する洞察を提供します。
* **マルチオブジェクトオーバーレイ：** 検出結果（ボール、コート、プレーヤー、ポーズ）を単一の動画出力にオーバーレイとして組み合わせます。
* **フレームごとの注釈：** 各フレームの詳細な注釈をJSONL形式で出力し、さらなる分析や他のツールとの統合に利用できます。
* **設定可能な分析：** Hydraを利用して、モデル、推論パラメータ、入出力設定を柔軟に構成できます。

## 使用技術

このプロジェクトは、強力なツールとライブラリの組み合わせで構築されています：

* **プログラミング言語：** Python（主にPython 3.11）
* **コアディープラーニングフレームワーク：**
    * PyTorch
    * PyTorch Lightning（トレーニングコードの構造化）
* **設定管理：**
    * Hydra（柔軟で組織化された設定）
* **コンピュータビジョン：**
    * OpenCV（ビデオ処理と画像操作）
    * Albumentations（データ拡張）
* **データ処理＆科学計算：**
    * NumPy
    * Pandas
* **主要なモデルアーキテクチャ/ライブラリ：** （システムは以下のようなさまざまなモデルを統合）
    * TrackNet（ボール/オブジェクト追跡用）
    * Swin Transformer
    * Vision Transformer (ViT)
    * DETR (Detection Transformer)
    * UniFormer/SlowFast（ビデオ理解タスク用）
    * `timm`（PyTorch Image Models）
    * `transformers`（Hugging Face）
* **実験追跡（依存関係から推測）：**
    * TensorBoard / TensorBoardX
* **ビデオ処理ユーティリティ：**
    * `yt-dlp`（必要に応じてビデオクリップをダウンロード）

## プロジェクト構造

プロジェクトは以下のように構成されています：

```
.
├── configs/              # Hydra設定ファイル（トレーニング、推論など）
│   ├── infer/            # 推論固有の設定
│   └── train/            # トレーニング固有の設定
├── docs/                 # ドキュメントと補足資料
│   └── images/           # ドキュメントで使用される画像
├── hydra_outputs/        # Hydraのデフォルト出力ディレクトリ（ログ、モデルなど）
├── samples/              # テストとデモ用のサンプルビデオファイル
├── src/                  # メインソースコード
│   ├── annotation/       # データ注釈用スクリプト
│   ├── ball/             # ボール検出と追跡モジュール
│   ├── court/            # コートライン検出モジュール
│   ├── event/            # イベント認識モジュール
│   ├── multi/            # 複合マルチオブジェクト分析モジュール
│   ├── player/           # プレーヤー検出と追跡モジュール
│   ├── pose/             # ポーズ推定モジュール
│   ├── utils/            # 共通ユーティリティ関数とクラス
│   ├── infer.py          # 推論実行用メインスクリプト
│   ├── train_ball.py     # ボール検出トレーニング用スクリプト
│   ├── train_court.py    # コート検出トレーニング用スクリプト
│   └── train_player.py   # プレーヤー検出トレーニング用スクリプト
├── tests/                # テストスクリプト（存在する場合）
├── tools/                # 様々なタスク用ユーティリティスクリプト（データセット変換など）
├── .gitignore            # Gitが無視すべき意図的に追跡しないファイルを指定
├── pyproject.toml        # プロジェクトメタデータとビルドシステム設定（リンターを含む）
├── requirements.txt      # プロジェクト依存関係
└── README.md             # 英語版READMEファイル
```

* **`src/`**: コアロジックを含み、各主要コンポーネント（ボール、コート、プレーヤー、ポーズ、イベント検出）と共有ユーティリティのサブディレクトリがあります。各コンポーネントには通常、データセット、モデル、予測器、トレーナーのサブモジュールが含まれています。
* **`configs/`**: すべてのHydra設定ファイルを保持し、異なるタスク（推論、トレーニング）とコンポーネントの設定を分離します。
* **`samples/`**: クイックテストとデモのためのサンプルビデオデータを提供します。
* **`tools/`**: データコンバータなどの様々なヘルパースクリプトを含みます。
* **`docs/`**: 追加のドキュメント（元の詳細なプロジェクト構造を含む）を含みます。
* **`hydra_outputs/`**: ログ、モデルチェックポイント、その他の実行アーティファクトがHydraによって通常保存される場所です。

## セットアップとインストール

1. **リポジトリのクローン：**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Python環境の作成：**
   仮想環境（例：venv、conda）の使用をお勧めします。このプロジェクトはPython 3.11を使用しています（`pyproject.toml`に示されている通り）。
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windowsでは `venv\Scripts\activate`
   ```
   または、condaを使用：
   ```bash
   conda create -n sports_analysis python=3.11
   conda activate sports_analysis
   ```

3. **依存関係のインストール：**
   プロジェクトの依存関係は`requirements.txt`にリストされています。
   ```bash
   pip install -r requirements.txt
   ```
   *注：`requirements.txt`ファイルはUTF-16エンコーディングのようです。問題が発生した場合は、最初にUTF-8に変換するか、pipのバージョンがサポートしている場合はインストール中にエンコーディングを処理する必要があるかもしれません。*
   依存関係の一つはGitHubからの`UniFormer`の特定のコミットです。pipがクローンするためにgitがインストールされていることを確認してください。

4. **事前トレーニング済みモデル（チェックポイント）：**
   設定ファイル（`configs/infer/**/*.yaml`）は様々なモデルの`ckpt_path`を参照しています。あなたは：
   * 事前トレーニング済みモデルのチェックポイントを別途入手可能な場合はダウンロードする。
   * または、独自のモデルをトレーニングし（トレーニングセクションを参照）、これらのパスを適宜更新する。
   * 設定内のパスはプレースホルダーであるか、チェックポイントが特定のディレクトリ（例：`checkpoints/`または`hydra_outputs/`内）に配置されていることを前提とした相対パスである可能性があります。

5. **環境変数（オプション）：**
   コンポーネントによっては特定の環境変数（データセットやAPIへのアクセスなど）が必要かどうかを確認してください。これは明示的に記載されていませんが、複雑なプロジェクトでは一般的です。

## 使用方法

このプロジェクトは設定管理にHydraを使用しています。すべての設定は`configs/`ディレクトリにあり、コマンドラインからオーバーライドできます。

### 1. 設定

* **主な設定ファイル：**
    * トレーニング：`configs/train/`（例：`ball.yaml`、`court.yaml`）
    * 推論：`configs/infer/`（例：`infer.yaml`、および`ball/`、`court/`などの下の特定タスク設定）
* **Hydraオーバーライド：** コマンドラインから任意の設定パラメータをオーバーライドできます。例えば、トレーニングのバッチサイズを変更するには：
    ```bash
    python src/train_ball.py batch_size=16
    ```
* **出力ディレクトリ：** Hydraは自動的に各実行の出力ディレクトリを`hydra_outputs/`の下に作成します。これにはログ、チェックポイント（コールバックを介して保存された場合）、その他のアーティファクトが含まれます。

### 2. モデルのトレーニング

トレーニングスクリプトは`src/`ディレクトリに提供されています（例：`train_ball.py`、`train_court.py`、`train_player.py`）。これらのスクリプトはPyTorch Lightningを使用します。

* **例：ボール検出モデルのトレーニング**
    ```bash
    python src/train_ball.py \
        annotation_file=path/to/your/ball_annotations.json \
        image_root=path/to/your/images/ \
        trainer.max_epochs=100 \
        batch_size=8 \
        num_workers=4
    ```
    * `annotation_file`と`image_root`をあなたのデータセットを指すように調整します。
    * 必要に応じて、それぞれの設定ファイル（例：`configs/train/ball.yaml`）を参照して他のパラメータを変更します。
    * トレーニングされたモデルチェックポイントは、通常`ModelCheckpoint`のようなコールバック（多くの場合、トレーニングスクリプトまたはそのHydra設定で構成されます）によって、その実行のHydra出力ディレクトリに保存されます。

### 3. 推論の実行

推論のメインスクリプトは`src/infer.py`です。様々な分析タスクのためのさまざまなモードをサポートしています。

* **一般的なコマンド構造：**
    ```bash
    python src/infer.py \
        mode=<inference_mode> \
        input_path=<path_to_video_or_image_directory> \
        output_path=<path_to_output_video_or_directory> \
        common.device=cuda  # または cpu
        [other_config_overrides]
    ```

* **推論モード（`mode=`）：**
    * `ball`：ボールの検出と追跡。
        ```bash
        python src/infer.py mode=ball input_path=samples/sample_video.mp4 output_path=outputs/ball_detected.mp4
        ```
    * `court`：コートラインの検出。
        ```bash
        python src/infer.py mode=court input_path=samples/sample_video.mp4 output_path=outputs/court_detected.mp4
        ```
    * `player`：プレーヤーの検出。
        ```bash
        python src/infer.py mode=player input_path=samples/sample_video.mp4 output_path=outputs/player_detected.mp4
        ```
    * `pose`：プレーヤーのポーズ推定（プレーヤー検出とポーズ推定を組み合わせる）。
        ```bash
        python src/infer.py mode=pose input_path=samples/sample_video.mp4 output_path=outputs/pose_estimated.mp4
        ```
    * `multi`：ボール、コート、ポーズの検出を出力ビデオ上のオーバーレイと組み合わせる。
        ```bash
        python src/infer.py mode=multi input_path=samples/sample_video.mp4 output_path=outputs/multi_analysis.mp4
        ```
    * `frames`：ビデオを処理し、フレームごとの注釈をJSONLファイルに出力および/または個々のフレームを保存する。
        ```bash
        python src/infer.py mode=frames input_path=samples/sample_video.mp4 output_path=outputs/frame_data/ output_json_path=outputs/annotations.jsonl
        ```

* **推論のための重要な考慮事項：**
    * **チェックポイントパス（`ckpt_path`）：** 関連する設定ファイル（例：`configs/infer/ball/lite_tracknet.yaml`）内の`ckpt_path`変数が、トレーニング済みまたはダウンロードしたモデルチェックポイントを指していることを確認してください。これらのパスはコマンドラインからオーバーライドできます：
        ```bash
        python src/infer.py mode=ball ball.ckpt_path=path/to/your/ball_model.ckpt ...
        ```
    * **入力/出力：**
        * `input_path`：入力ビデオファイルまたは画像ディレクトリへのパス。
        * `output_path`：出力ビデオまたはディレクトリへのパス。`frames`モードでは、多くの場合これはディレクトリです。
        * `output_json_path`（`frames`モード用）：出力JSONLファイルへのパス。
    * **デバイス：** GPUには`common.device=cuda`、CPUには`common.device=cpu`を使用します。
    * **ハーフ精度：** 互換性のあるGPUでより高速な推論を行うには、ハーフ精度を有効にすることができます：`common.use_half=True`。

### 4. 注釈ツール

データ注釈と準備のためのスクリプトは`src/annotation/`にあります。
* `auto_court_annotator.py`：コートラインの自動注釈を支援する可能性があります。
* `generate_pose.py`：検出されたプレーヤーからポーズ注釈を生成するためのものかもしれません。
    （詳細な使用法については、スクリプト自体を参照してください。）

## 例/デモ

* **サンプルビデオ：** `samples/`ディレクトリには、いくつかのビデオファイル（例：`lite_tracknet.mp4`、`multi_overlay_up.mp4`、`overlay_fpn.mp4`）が含まれており、推論スクリプトのテスト入力として使用できます。
    サンプルビデオを使用した例：
    ```bash
    python src/infer.py mode=multi input_path=samples/multi_overlay_up.mp4 output_path=outputs/demo_multi_output.mp4
    ```

* **デモスクリプト：** ファイル`chat_gpt/demo_cache_batch.py`はデモンストレーションスクリプトのようで、おそらくバッチ処理やキャッシングメカニズムを紹介しています。その特定の使用法と目的については、スクリプトを参照してください。

* **キーポイントの可視化：** `docs/images/`ディレクトリには、可視化されたキーポイントを示すサンプル画像（例：`visualized_keypoints_sample_0_PuXlxKdUIes_2450.png`）が含まれており、ポーズ推定や他のキーポイント検出モジュールの出力を示している可能性があります。

## 貢献

このプロジェクトへの貢献を歓迎します！貢献したい場合は、以下を検討してください：

* **バグレポート：** バグを見つけた場合は、問題、再現手順、お使いの環境を詳細に記載した問題を開いてください。
* **機能リクエスト：** 新機能や拡張については、アイデアを議論するために問題を開くことを検討してください。
* **プルリクエスト：**
    1. リポジトリをフォークします。
    2. 機能またはバグ修正用の新しいブランチを作成します（`git checkout -b feature/your-feature-name`または`bugfix/your-bug-fix`）。
    3. 変更を加えます。
    4. コードがプロジェクトのスタイルに準拠していることを確認します（Black、isort、Ruffなどのリンターが`pyproject.toml`で設定されています）。
    5. 該当する場合は、変更のテストを書きます。
    6. 変更をコミットし、フォークにプッシュします。
    7. メインリポジトリにプルリクエストを開きます。

このプロジェクトには日本語のコメントと命名規則が含まれていることに注意してください。貢献する際は、一貫性を維持するか、コードとドキュメントでの言語使用についてメンテナーと相談してください。
