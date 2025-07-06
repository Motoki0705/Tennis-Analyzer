# テニス解析AIモデル群：技術リファレンス

## 1. 概要

このドキュメントは、テニス映像解析のために開発された5つの深層学習モデル群に関する技術的な仕様を定義するものです。各モデルは独立したコンポーネントとして、または組み合わせて使用することが可能であり、新たなテニス解析システムの構築における基礎ブロックとして利用できます。

本文書では、各モデルの読み込み方法、必須となる前処理、入力と出力のデータ形式（テンソル形状）、および出力を実用的な情報に変換するための後処理について詳述します。

**対象モデル一覧:**
1.  **ボール検出モデル**: 映像からボールの位置を特定します。
2.  **コート検出モデル**: コートのキーポイントを検出します。
3.  **選手検出モデル**: 選手の位置をバウンディングボックスとして検出します。
4.  **姿勢推定モデル**: 検出した選手の骨格キーポイントを推定します。
5.  **イベント検出モデル**: 上記4つのモデル情報を統合し、「ヒット」や「バウンド」を時系列で検出します。

---

## 2. モデル別技術仕様

### 2.1. ボール検出モデル

-   **目的**: 連続するフレームからボールの2次元位置をヒートマップとして予測する。
-   **モデルクラス**: `src.ball.lit_module.lit_lite_tracknet_focal.LitLiteTracknetFocalLoss`
-   **チェックポイント例**: `checkpoints/ball/lit_lite_tracknet/best_model.ckpt`

-   **モデルのロード方法**:
    ```python
    from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss

    # PyTorch Lightningのチェックポイントからモデルをロード
    lit_model = LitLiteTracknetFocalLoss.load_from_checkpoint("path/to/your/ball_model.ckpt")
    
    # 推論に使用するモデル本体を取得
    ball_model = lit_model.model.eval() # 評価モードに設定
    ```

-   **入力 (Input)**:
    -   **概要**: 連続する3フレームをチャンネル方向に結合した画像テンソル。
    -   **前処理**:
        1.  入力画像をリサイズする (例: 高さ360, 幅640)。
        2.  ピクセル値を正規化する (例: `albumentations.Normalize`)。
        3.  PyTorchテンソルに変換する。
        4.  連続3フレーム分のテンソル (`C, H, W`) をチャンネル次元で結合する (`torch.cat`)。
    -   **最終的なテンソル形状**: $B \times (3 \times N_{frames}) \times H \times W$
        -   例: $1 \times (3 \times 3) \times 360 \times 640 \rightarrow 1 \times 9 \times 360 \times 640$

-   **出力 (Output)**:
    -   **概要**: ボールの存在確率を表す生のヒートマップ（ロジット）。
    -   **テンソル形状**: $B \times 1 \times H \times W$
        -   例: $1 \times 1 \times 360 \times 640$

-   **推奨される後処理**:
    1.  出力テンソルにシグモイド関数を適用し、確率に変換する: `probs = torch.sigmoid(output)`。
    2.  ヒートマップ上で最も確率の高い点のインデックスを取得する: `torch.argmax()`。
    3.  インデックスを `(y, x)` 座標に変換する。
    4.  座標を画像のH, Wで割ることで `[0, 1]` の範囲に正規化する。
    5.  **最終的な特徴量形式**: `[x_normalized, y_normalized, max_probability]` の3次元ベクトル。

---

### 2.2. コート検出モデル

-   **目的**: 単一フレームからテニスコートのキーポイント（ラインの交点など）をヒートマップとして予測する。
-   **モデルクラス**: `src.court.lit_module.lit_lite_tracknet_focal.LitLiteTracknetFocal`
-   **チェックポイント例**: `checkpoints/court/lit_lite_tracknet/epoch=... .ckpt`

-   **モデルのロード方法**:
    ```python
    from src.court.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocal

    lit_model = LitLiteTracknetFocal.load_from_checkpoint("path/to/your/court_model.ckpt")
    court_model = lit_model.model.eval()
    ```

-   **入力 (Input)**:
    -   **概要**: 標準的な3チャンネル（RGB）の単一画像テンソル。
    -   **前処理**: リサイズ (例: 360x640)、正規化、テンソル変換。
    -   **最終的なテンソル形状**: $B \times C \times H \times W$ (例: $1 \times 3 \times 360 \times 640$)

-   **出力 (Output)**:
    -   **概要**: コートキーポイントの存在確率を表す生のヒートマップ（ロジット）。
    -   **テンソル形状**: $B \times 1 \times H \times W$ (例: $1 \times 1 \times 360 \times 640$)

-   **推奨される後処理**:
    1.  出力テンソルにシグモイド関数を適用し、確率に変換する。
    2.  ヒートマップから確率の高い上位N個（例: 15個）のピーク座標 `(y, x)` を抽出する。
        -   *実装例: 最も高い点を見つけ、その周辺をマスクして次の点を探す処理を繰り返す。*
    3.  各座標を正規化し、仮の可視性フラグ `1.0` を付与して `[x_norm, y_norm, 1.0]` の形式にする。
    4.  **最終的な特徴量形式**: N個のキーポイントベクトルをフラット化した $N \times 3$ 次元のベクトル (例: $15 \times 3 = 45$ 次元)。

---

### 2.3. 選手検出モデル (RT-DETR)

-   **目的**: 単一フレームから選手を検出し、バウンディングボックス（BBox）を取得する。
-   **モデルクラス**: `src.player.lit_module.lit_rtdetr.LitRtdetr`
-   **チェックポイント例**: `checkpoints/player/lit_rt_detr/epoch=... .ckpt`

-   **モデルのロード方法**:
    ```python
    from src.player.lit_module.lit_rtdetr import LitRtdetr
    from transformers import RTDetrImageProcessor

    # モデル本体をロード
    player_model = LitRtdetr.load_from_checkpoint("path/to/your/player_model.ckpt").eval()
    
    # 対応するプロセッサをロード
    player_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    ```

-   **入力 (Input)**:
    -   **概要**: `PIL.Image` 形式の単一フレーム画像。
    -   **前処理**: `RTDetrImageProcessor` がリサイズ、正規化、テンソル化を全て担当する。
        ```python
        inputs = player_processor(images=pil_image, return_tensors="pt")
        ```
    -   **最終的なテンソル形式**: プロセッサが生成した辞書オブジェクト。

-   **出力 (Output)**:
    -   **概要**: モデル固有の生出力オブジェクト。
    -   **形式**: `transformers.models.rtdetr.modeling_rtdetr.RTDetrObjectDetectionOutput`

-   **推奨される後処理**:
    1.  `RTDetrImageProcessor.post_process_object_detection()` を使用して生出力をパースする。
    2.  信頼度スコアで結果をフィルタリングする（例: `threshold=0.5`）。
    3.  結果からBBox座標 `(x_min, y_min, x_max, y_max)` とスコアを取得する。
    4.  座標を画像の幅と高さで正規化する。
    5.  **最終的な特徴量形式**: 検出された選手ごとに、`[x1_norm, y1_norm, x2_norm, y2_norm, score]` の5次元ベクトル。

---

### 2.4. 姿勢推定モデル (ViTPose)

-   **目的**: 指定されたBBox内の人物の骨格キーポイント（17点）を推定する。
-   **モデルクラス**: `transformers.VitPoseForPoseEstimation`
-   **Hugging Face ID**: `usyd-community/vitpose-base-simple`

-   **モデルのロード方法**:
    ```python
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    # Hugging Face Hubからモデルとプロセッサをロード
    hf_id = "usyd-community/vitpose-base-simple"
    pose_model = VitPoseForPoseEstimation.from_pretrained(hf_id).eval()
    pose_processor = AutoProcessor.from_pretrained(hf_id)
    ```

-   **入力 (Input)**:
    -   **概要**: `PIL.Image` 形式の画像と、その画像内の人物BBoxリスト。
    -   **前提条件**: **選手検出モデルの出力（BBox）が必須。**
    -   **前処理**: `AutoProcessor` が画像とBBoxリストを受け取り、必要な前処理を行う。
        ```python
        # player_boxesは (x, y, width, height) 形式のNumPy配列
        inputs = pose_processor(images=pil_image, boxes=[player_boxes], return_tensors="pt")
        ```
    -   **最終的なテンソル形式**: プロセッサが生成した辞書オブジェクト。

-   **出力 (Output)**:
    -   **概要**: モデル固有の生出力オブジェクト。
    -   **形式**: `transformers.models.vit_pose.modeling_vit_pose.PoseEstimationOutput`

-   **推奨される後処理**:
    1.  `AutoProcessor.post_process_pose_estimation()` を使用して生出力をパースする。
    2.  検出された各人物について、17点のキーポイント座標 `(x, y)` とスコアを取得する。
    3.  各キーポイント座標を正規化し、スコアに基づいて可視性フラグ `v` を設定する（例: `v=2` if `score > 0.5` else `v=1`）。
    4.  **最終的な特徴量形式**: 各人物について、17個の `[x_norm, y_norm, v]` をフラット化した $17 \times 3 = 51$ 次元のベクトル。

---

### 2.5. イベント検出モデル (Transformer)

-   **目的**: 複数の時系列特徴量（ボール、コート、選手、姿勢）を統合し、フレームごとのイベント（ヒット、バウンド）発生確率を予測する。
-   **モデルクラス**: `src.event.model.transformer_v2.EventTransformerV2`
-   **チェックポイント例**: `checkpoints/event/transformer_v2.py/epoch=... .ckpt`

-   **モデルのロード方法**:
    ```python
    from src.event.model.transformer_v2 import EventTransformerV2
    
    event_model = EventTransformerV2.load_from_checkpoint("path/to/your/event_model.ckpt").eval()
    ```

-   **入力 (Input)**:
    -   **概要**: 4種類の時系列特徴量テンソル。**全てのテンソルはシーケンス長（フレーム数）を揃える必要がある。**
    -   **前提条件**: **他の4モデル全ての後処理済み出力が必須。**
    -   **入力テンソル仕様**:
        1.  `ball_tensor`: $B \times S \times D_{ball}$ (例: $1 \times S \times 3$)
        2.  `court_tensor`: $B \times S \times D_{court}$ (例: $1 \times S \times 45$)
        3.  `player_bbox_tensor`: $B \times S \times P \times D_{bbox}$ (例: $1 \times S \times P \times 5$)
        4.  `player_pose_tensor`: $B \times S \times P \times D_{pose}$ (例: $1 \times S \times P \times 51$)
    -   **記号**:
        -   $B$: バッチサイズ (通常は1)
        -   $S$: シーケンス長 (動画の総フレーム数)
        -   $P$: 1フレームあたりの最大選手数（シーケンス全体でパディング）
        -   $D$: 各特徴量の次元数
    -   **注意**: 選手数が変動するため、`player_bbox_tensor` と `player_pose_tensor` はシーケンス全体で検出された最大選手数 `P` に合わせてゼロパディングする必要がある。

-   **出力 (Output)**:
    -   **概要**: フレームごとの各イベント（クラス0: ヒット, クラス1: バウンド）のロジット。
    -   **テンソル形状**: $B \times S \times N_{events}$ (例: $1 \times S \times 2$)

-   **推奨される後処理**:
    1.  出力テンソルにシグモイド関数を適用し、各イベントの確率に変換する。
    2.  **最終的な特徴量形式**: `(S, 2)` の形状を持つNumPy配列またはテンソル。各行がフレーム、各列がイベントの確率を表す。

## 3. モデル間の依存関係

これらのモデルを組み合わせて使用する際には、以下の依存関係を考慮する必要があります。

-   **姿勢推定モデル**は、**選手検出モデル**が生成したバウンディングボックスを直接の入力として要求します。したがって、姿勢推定を行うには必ず選手検出を先に行う必要があります。
-   **イベント検出モデル**は、他の**4つ全てのモデル（ボール、コート、選手、姿勢）**から得られる特徴量を入力として要求します。これらの特徴量は、本ドキュメントで詳述した後処理を施し、指定されたテンソル形状に整形（特にパディング）されている必要があります。

個別のモデル（例: ボール検出のみ、選手検出のみ）は、これらの依存関係なしに単独で使用することが可能です。