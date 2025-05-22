# 自動テニス分析システム

このプロジェクトは、テニスの試合映像を包括的に分析するために設計されたAI搭載ツールキットです。ディープラーニングモデルを活用して、テニスボール、選手、コートライン、選手のポーズなどの主要要素を自動的に検出・追跡します。このシステムは、映像を処理して貴重な洞察を抽出し、注釈付きの出力を生成することができます。

## 主な機能

このシステムは、詳細なスポーツ映像分析のためのさまざまな機能を提供します。

*   **ボール検出と追跡:** 映像全体を通してボールの軌道を正確に識別し追跡します。
*   **コートライン検出:** コートラインをセグメント化し識別することで、プレイエリアを理解します。
*   **選手検出と追跡:** コート上の選手を検出し、その動きを追跡することができます。
*   **選手ポーズ推定:** 選手のポーズを推定し、姿勢やアクションに関する洞察を提供します。
*   **マルチオブジェクトオーバーレイ:** 検出結果（ボール、コート、選手、ポーズ）を単一のビデオ出力にオーバーレイ付きで統合します。
*   **フレーム毎アノテーション:** 詳細なアノテーションをフレーム毎にJSONL形式で出力し、さらなる分析や他のツールとの連携を可能にします。
*   **設定可能な分析:** Hydraを利用して、モデル、推論パラメータ、入出力設定などを柔軟に設定できます。

## 使用技術

このプロジェクトは、以下の強力なツールとライブラリを組み合わせて構築されています。

*   **プログラミング言語:** Python (主に Python 3.11)
*   **コアディープラーニングフレームワーク:**
    *   PyTorch
    *   PyTorch Lightning (トレーニングコードの構造化のため)
*   **設定管理:**
    *   Hydra (柔軟で整理された設定のため)
*   **コンピュータビジョン:**
    *   OpenCV (ビデオ処理と画像操作のため)
    *   Albumentations (データ拡張のため)
*   **データ処理・科学計算:**
    *   NumPy
    *   Pandas
*   **主要モデルアーキテクチャ/ライブラリ:** (システムは様々なモデルを統合しており、以下を含むか類似のものが使われています)
    *   TrackNet (ボール/オブジェクト追跡用)
    *   Swin Transformer
    *   Vision Transformer (ViT)
    *   DETR (Detection Transformer)
    *   UniFormer/SlowFast (ビデオ理解タスク用)
    *   `timm` (PyTorch Image Models)
    *   `transformers` (Hugging Face)
*   **実験追跡 (依存関係から推測):**
    *   TensorBoard / TensorBoardX
*   **ビデオ処理ユーティリティ:**
    *   `yt-dlp` (必要に応じてビデオクリップをダウンロードするため)

## プロジェクト構成

プロジェクトの構成は以下の通りです。

```
.
├── configs/              # トレーニング、推論などのためのHydra設定ファイル
│   ├── infer/            # 推論固有の設定
│   └── train/            # トレーニング固有の設定
├── docs/                 # ドキュメントおよび補足資料
│   └── images/           # ドキュメントで使用される画像
├── hydra_outputs/        # Hydraのデフォルト出力ディレクトリ（ログ、モデルなど）
├── samples/              # テストおよびデモ用のサンプルビデオファイル
├── src/                  # メインソースコード
│   ├── annotation/       # データアノテーション用スクリプト
│   ├── ball/             # ボール検出・追跡モジュール
│   ├── court/            # コートライン検出モジュール
│   ├── event/            # イベント認識モジュール
│   ├── multi/            # 複数オブジェクト統合分析モジュール
│   ├── player/           # 選手検出・追跡モジュール
│   ├── pose/             # ポーズ推定モジュール
│   ├── utils/            # 共通ユーティリティ関数・クラス
│   ├── infer.py          # 推論実行用メインスクリプト
│   ├── train_ball.py     # ボール検出用トレーニングスクリプト例
│   ├── train_court.py    # コート検出用トレーニングスクリプト例
│   └── train_player.py   # 選手検出用トレーニングスクリプト例
├── tests/                # テストスクリプト (もしあれば)
├── tools/                # 各種タスク用ユーティリティスクリプト (データセット変換など)
├── .gitignore            # Gitが無視すべき意図的に追跡しないファイル群
├── pyproject.toml        # プロジェクトメタデータとビルドシステム設定 (リンター含む)
├── requirements.txt      # プロジェクトの依存関係
└── README.md             # このファイル (英語版)
└── README.ja.md          # このファイル (日本語版)
```

*   **`src/`**: 主要コンポーネント（ボール、コート、選手、ポーズ、イベント検出）および共有ユーティリティのサブディレクトリを含むコアロジックを格納します。各コンポーネントには通常、データセット、モデル、プレディクタ、トレーナー用のサブモジュールが含まれます。
*   **`configs/`**: すべてのHydra設定ファイルを保持し、さまざまなタスク（推論、トレーニング）やコンポーネントの設定を分離します。
*   **`samples/`**: 簡単なテストやデモンストレーション用のサンプルビデオデータを提供します。
*   **`tools/`**: データコンバータなどのさまざまなヘルパースクリプトが含まれます。
*   **`docs/`**: 元の詳細なプロジェクト構成を含む追加ドキュメントを格納します。
*   **`hydra_outputs/`**: 通常、Hydraによって実行時のログ、モデルチェックポイント、その他のアーティファクトが保存される場所です。

## セットアップとインストール

1.  **リポジトリをクローンします:**
    ```bash
    git clone <リポジトリURL>
    cd <リポジトリディレクトリ>
    ```

2.  **Python環境を作成します:**
    仮想環境（例: venv, conda）の使用を推奨します。このプロジェクトはPython 3.11を使用しています（`pyproject.toml`に記載）。
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windowsでは `venv\Scripts\activate` を使用
    ```
    またはcondaを使用する場合:
    ```bash
    conda create -n sports_analysis python=3.11
    conda activate sports_analysis
    ```

3.  **依存関係をインストールします:**
    プロジェクトの依存関係は`requirements.txt`にリストされています。
    ```bash
    pip install -r requirements.txt
    ```
    *注意: `requirements.txt`ファイルはUTF-16エンコーディングのようです。問題が発生した場合は、まずUTF-8に変換するか、pipのバージョンが対応していればインストール時にエンコーディングを処理する必要があるかもしれません。*
    依存関係の1つはGitHubからの`UniFormer`の特定コミットです。pipがクローンできるようにgitがインストールされていることを確認してください。

4.  **事前学習済みモデル (チェックポイント):**
    設定ファイル（`configs/infer/**/*.yaml`）は、さまざまなモデルの`ckpt_path`を参照しています。以下を行う必要があります:
    *   利用可能な場合は、事前学習済みのモデルチェックポイントを別途ダウンロードします。
    *   または、独自のモデルをトレーニングし（トレーニングセクションを参照）、これらのパスを適宜更新します。
    *   設定ファイル内のパスは、プレースホルダーであるか、チェックポイントが特定のディレクトリ（例: `checkpoints/`または`hydra_outputs/`内）に配置されることを前提とした相対パスである可能性があります。

5.  **環境変数 (オプション):**
    特定のコンポーネントが特定の環境変数（データセットやAPIへのアクセスなど）を必要とするかどうかを確認してください。これは明示的に記載されていませんが、複雑なプロジェクトでは一般的です。

## 使用方法

このプロジェクトは設定管理にHydraを使用しています。すべての設定は`configs/`ディレクトリにあり、コマンドライン経由で上書きできます。

### 1. 設定

*   **主な設定ファイル:**
    *   トレーニング: `configs/train/` (例: `ball.yaml`, `court.yaml`)
    *   推論: `configs/infer/` (例: `infer.yaml`、および`ball/`, `court/`などの下の特定のタスク設定)
*   **Hydraオーバーライド:** コマンドラインから任意の設定パラメータを上書きできます。例えば、トレーニングのバッチサイズを変更するには:
    ```bash
    python src/train_ball.py batch_size=16
    ```
*   **出力ディレクトリ:** Hydraは実行ごとに`hydra_outputs/`の下に出力ディレクトリを自動的に作成します。これにはログ、チェックポイント（コールバック経由で保存された場合）、その他のアーティファクトが含まれます。

### 2. モデルのトレーニング

トレーニングスクリプトは`src/`ディレクトリに提供されています（例: `train_ball.py`, `train_court.py`, `train_player.py`）。これらのスクリプトはPyTorch Lightningを使用しています。

*   **例: ボール検出モデルのトレーニング**
    ```bash
    python src/train_ball.py \
        annotation_file=path/to/your/ball_annotations.json \
        image_root=path/to/your/images/ \
        trainer.max_epochs=100 \
        batch_size=8 \
        num_workers=4
    ```
    *   `annotation_file`と`image_root`をデータセットに合わせて調整してください。
    *   それぞれの設定ファイル（例: `configs/train/ball.yaml`）を参照して、必要に応じて他のパラメータを変更してください。
    *   トレーニング済みのモデルチェックポイントは、通常、`ModelCheckpoint`のようなコールバックによって（多くの場合、トレーニングスクリプトまたはそのHydra設定で設定）、その実行のHydra出力ディレクトリに保存されます。

### 3. 推論の実行

推論のメインスクリプトは`src/infer.py`です。さまざまな分析タスクに対応するさまざまなモードをサポートしています。

*   **一般的なコマンド構造:**
    ```bash
    python src/infer.py \
        mode=<推論モード> \
        input_path=<ビデオまたは画像ディレクトリへのパス> \
        output_path=<出力ビデオまたはディレクトリへのパス> \
        common.device=cuda  # または cpu
        [その他の設定オーバーライド]
    ```

*   **推論モード (`mode=`):**
    *   `ball`: ボール検出と追跡。
        ```bash
        python src/infer.py mode=ball input_path=samples/sample_video.mp4 output_path=outputs/ball_detected.mp4
        ```
    *   `court`: コートライン検出。
        ```bash
        python src/infer.py mode=court input_path=samples/sample_video.mp4 output_path=outputs/court_detected.mp4
        ```
    *   `player`: 選手検出。
        ```bash
        python src/infer.py mode=player input_path=samples/sample_video.mp4 output_path=outputs/player_detected.mp4
        ```
    *   `pose`: 選手ポーズ推定（選手検出とポーズ推定を組み合わせる）。
        ```bash
        python src/infer.py mode=pose input_path=samples/sample_video.mp4 output_path=outputs/pose_estimated.mp4
        ```
    *   `multi`: ボール、コート、ポーズの検出結果をオーバーレイ付きで出力ビデオに統合。
        ```bash
        python src/infer.py mode=multi input_path=samples/sample_video.mp4 output_path=outputs/multi_analysis.mp4
        ```
    *   `frames`: ビデオを処理し、フレーム毎のアノテーションをJSONLファイルに出力、および/または個々のフレームを保存。
        ```bash
        python src/infer.py mode=frames input_path=samples/sample_video.mp4 output_path=outputs/frame_data/ output_json_path=outputs/annotations.jsonl
        ```

*   **推論に関する重要な考慮事項:**
    *   **チェックポイントパス (`ckpt_path`):** 関連する設定ファイル（例: `configs/infer/ball/lite_tracknet.yaml`）内の`ckpt_path`変数が、トレーニング済みまたはダウンロードしたモデルチェックポイントを指していることを確認してください。これらのパスはコマンドラインから上書きできます:
        ```bash
        python src/infer.py mode=ball ball.ckpt_path=path/to/your/ball_model.ckpt ...
        ```
    *   **入力/出力:**
        *   `input_path`: 入力ビデオファイルまたは画像ディレクトリへのパス。
        *   `output_path`: 出力ビデオまたはディレクトリのパス。`frames`モードの場合、これは多くの場合ディレクトリです。
        *   `output_json_path` (`frames`モードの場合): 出力JSONLファイルへのパス。
    *   **デバイス:** GPUの場合は`common.device=cuda`、CPUの場合は`common.device=cpu`を使用します。
    *   **半精度:** 対応GPUでの高速推論のために、半精度を有効にすることを試みることができます: `common.use_half=True`。

### 4. アノテーションツール

データアノテーションと準備のためのスクリプトは`src/annotation/`にあります。
*   `auto_court_annotator.py`: コートラインの自動アノテーションを支援する可能性があります。
*   `generate_pose.py`: 検出された選手からポーズアノテーションを生成するためかもしれません。
    (詳細な使用方法については、スクリプト自体を参照してください。)

## サンプル/デモ

*   **サンプルビデオ:** `samples/`ディレクトリには、推論スクリプトのテスト入力として使用できるいくつかのビデオファイル（例: `lite_tracknet.mp4`, `multi_overlay_up.mp4`, `overlay_fpn.mp4`）が含まれています。
    サンプルビデオを使用した例:
    ```bash
    python src/infer.py mode=multi input_path=samples/multi_overlay_up.mp4 output_path=outputs/demo_multi_output.mp4
    ```

*   **デモスクリプト:** `chat_gpt/demo_cache_batch.py`ファイルはデモンストレーションスクリプトのようで、バッチ処理やキャッシュメカニズムを示している可能性があります。特定の用途と目的については、スクリプトを参照してください。

*   **キーポイントの視覚化:** `docs/images/`ディレクトリには、視覚化されたキーポイントを示すサンプル画像（例: `visualized_keypoints_sample_0_PuXlxKdUIes_2450.png`）が含まれており、これらはポーズ推定や他のキーポイント検出モジュールの出力を示している可能性があります。

## コントリビューション

このプロジェクトへの貢献を歓迎します！貢献したい場合は、以下を検討してください。

*   **バグレポート:** バグを見つけた場合は、問題、再現手順、および環境を詳述したissueを開いてください。
*   **機能リクエスト:** 新機能や機能強化については、気軽にissueを開いてアイデアを議論してください。
*   **プルリクエスト:**
    1.  リポジトリをフォークします。
    2.  機能またはバグ修正用の新しいブランチを作成します（`git checkout -b feature/your-feature-name`または`bugfix/your-bug-fix`）。
    3.  変更を加えます。
    4.  コードがプロジェクトのスタイル（`pyproject.toml`で設定されているBlack、isort、Ruffなどのリンター）に準拠していることを確認します。
    5.  該当する場合は、変更に対するテストを作成します。
    6.  変更をコミットし、フォークにプッシュします。
    7.  メインリポジトリへのプルリクエストを開きます。

このプロジェクトには日本語のコメントや命名規則がいくつか見られることに注意してください。貢献する際には、一貫性を保つか、コードやドキュメントでの言語使用についてメンテナーと話し合ってください。

## ライセンス

このプロジェクトのライセンスはまだ指定されていません。他者がプロジェクトをどのように使用し貢献できるかを明確にするために、リポジトリにLICENSEファイルを追加することをお勧めします。一般的なオープンソースライセンスには、MIT、Apache 2.0、GPLv3などがあります。
