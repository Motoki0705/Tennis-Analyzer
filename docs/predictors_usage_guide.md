# Tennis-Analyzerの各Predictorの使用ガイド

## 1. 概要

Tennis-Analyzerは、テニス映像を分析するための機械学習フレームワークです。主に以下の7つの予測モジュール（Predictor）を提供しています：

1. **BallPredictor**: テニスボールの検出・追跡
2. **CourtPredictor**: テニスコートのキーポイント検出 
3. **PlayerPredictor**: プレイヤーの検出
4. **PosePredictor**: プレイヤーの姿勢推定
5. **MultiPredictor**: 上記の複数のPredictorを統合して同時に処理
6. **FramesPredictor**: 動画をフレームに分解し、各フレームに対する総合的な分析結果をJSONL形式で出力
7. **ImageAnnotator**: 画像ディレクトリに対して各種予測を実行し、JSON形式で出力

## 2. 各Predictorの詳細

### 2.1 BallPredictor

BallPredictorはテニスボールの検出と追跡を行います。このPredictorは時系列情報を活用した特殊な入力形式を持ちます。

#### 主な機能
- 複数フレームを入力として、ボールの位置（x, y）と信頼度を予測
- ヒートマップベースの検出と座標回帰ベースの検出に対応
- 結果の可視化（オーバーレイ、ヒートマップ、特徴マップ）
- トラッキング結果の補間、外れ値除去

#### 入力形式と処理の詳細
BallPredictorの入力形式は、他のPredictorとは異なり、時系列フレームの集合を扱います：

- **入力形式**: `List[List[np.ndarray]]`
  - 最初のリストはバッチ（B個のクリップ）を表す
  - 内側のリストは各クリップ内の連続したT個のフレームを表す
  - 通常、T=3（3フレーム連続）を使用

- **内部処理**:
  1. 各クリップをバッチ処理用のテンソルに変換
  2. [B, C * T, H, W]形式のバッチテンソルをモデルに入力
  3. モデルは各クリップの最後のフレームにおけるボールの位置を予測

- **出力**: 各クリップの最後のフレーム（T番目のフレーム）におけるボール位置の予測結果
  - `List[dict]`形式で、各要素は `{"x": int, "y": int, "confidence": float}` の辞書

#### 使用方法
```python
predictor = BallPredictor(
    model=ball_model,              # 事前学習済みモデル
    input_size=(360, 640),         # 入力サイズ
    heatmap_size=(360, 640),       # ヒートマップサイズ
    num_frames=3,                  # 入力フレーム数(T)
    threshold=0.6,                 # 検出閾値
    device="cuda",                 # 推論デバイス
    visualize_mode="overlay",      # 可視化モード（"overlay", "heatmap", "features"）
    use_half=False,                # 半精度推論を使用するか
)

# クリップの準備 (B個のクリップ、各クリップはT=3フレーム)
clips = [
    [frame1, frame2, frame3],        # クリップ1: 3連続フレーム, frame3を推定
    [frame2, frame3, frame4],        # クリップ2: 3連続フレーム, frame4を推定
    # ... 他のクリップ
]

# 推論 - 各クリップの最後のフレーム(frame3, frame4, ...)についての結果を返す
results = predictor.predict(clips)  # -> List[dict], 長さはクリップ数(B)と同じ

# 動画処理 - 内部でスライディングウィンドウを作成し、連続予測を行う
predictor.run(
    input_path="input.mp4",        # 入力動画パス
    output_path="output.mp4",      # 出力動画パス
    batch_size=4,                  # バッチサイズ
)
```

#### 実装上の注意点
- `run()`メソッドを使用する場合、内部でスライディングウィンドウを作成し、自動的に連続フレームの処理を行います
- ボールの動きは連続性があるため、複数フレームを同時に処理することで、より安定した検出が可能になります
- バッチ処理時は、時系列の順序が維持されるように注意が必要です
- 予測結果に対して、外れ値除去と軌道補間を自動的に適用して滑らかな追跡結果を生成します

### 2.2 CourtPredictor

CourtPredictorはテニスコートの主要な15のキーポイントを検出します。

#### 主な機能
- コートの構造的なキーポイント検出（コーナー、ラインの交点など）
- 単一チャネルまたはマルチチャネル (キーポイント数) のヒートマップ出力に対応
- 結果の可視化（オーバーレイ、ヒートマップ）

#### 使用方法
```python
predictor = CourtPredictor(
    model=court_model,             # 事前学習済みモデル
    device="cuda",                 # 推論デバイス
    input_size=(256, 256),         # 入力サイズ
    num_keypoints=15,              # キーポイント数
    threshold=0.5,                 # 検出閾値
    radius=5,                      # 可視化時の円の半径
    kp_color=(0, 255, 0),          # キーポイントの色
    use_half=False,                # 半精度推論を使用するか
    visualize_mode="overlay",      # 可視化モード（"overlay", "heatmap", "heatmap_channels"）
)

# 推論
keypoints_list, heatmaps_list = predictor.predict(frames)  # frames: List[np.ndarray]

# 動画処理
predictor.run(
    input_path="input.mp4",        # 入力動画パス
    output_path="output.mp4",      # 出力動画パス
    batch_size=8,                  # バッチサイズ
)
```

### 2.3 PlayerPredictor

PlayerPredictorはテニスプレイヤーの検出を行います。

#### 主な機能
- プレイヤーのバウンディングボックス検出
- 信頼度スコアと共に検出結果を提供
- 検出結果の可視化（バウンディングボックス、ラベル、スコア）

#### 使用方法
```python
predictor = PlayerPredictor(
    model=player_model,            # 事前学習済みモデル
    processor=player_processor,    # 前処理・後処理プロセッサ
    label_map={0: "player"},       # ラベルマップ
    device="cuda",                 # 推論デバイス
    threshold=0.6,                 # 検出閾値
    use_half=False,                # 半精度推論を使用するか
)

# 推論
detections = predictor.predict(frames)  # frames: List[np.ndarray]

# 動画処理
predictor.run(
    input_path="input.mp4",        # 入力動画パス
    output_path="output.mp4",      # 出力動画パス
    batch_size=8,                  # バッチサイズ
)
```

### 2.4 PosePredictor

PosePredictorは、まずプレイヤーを検出し、検出されたプレイヤーに対して姿勢推定を行います。

#### 主な機能
- 2段階プロセス：プレイヤー検出→姿勢推定
- COCOフォーマットのキーポイント（17点）検出
- スケルトン描画による可視化

#### 使用方法
```python
predictor = PosePredictor(
    det_model=player_model,         # プレイヤー検出モデル
    det_processor=det_processor,    # 検出用プロセッサ
    pose_model=pose_model,          # 姿勢推定モデル
    pose_processor=pose_processor,  # 姿勢推定用プロセッサ
    device="cuda",                  # 推論デバイス
    player_label_id=0,              # プレイヤーのラベルID
    det_score_thresh=0.6,           # 検出閾値
    pose_score_thresh=0.6,          # 姿勢推定閾値
    use_half=False,                 # 半精度推論を使用するか
)

# 推論
pose_results = predictor.predict(frames)  # frames: List[np.ndarray]

# 動画処理
predictor.run(
    input_path="input.mp4",        # 入力動画パス
    output_path="output.mp4",      # 出力動画パス
    batch_size=8,                  # バッチサイズ
)
```

### 2.5 MultiPredictor

MultiPredictorは、ボール検出、コート検出、姿勢推定を同時に実行し、結果を統合して可視化します。

#### 主な機能
- 複数のPredictorを効率的に組み合わせて実行
- 各Predictorの実行間隔とバッチサイズを個別に設定可能
- 統合された結果の可視化

#### 使用方法
```python
predictor = MultiPredictor(
    ball_predictor=ball_predictor,    # 初期化済みBallPredictor
    court_predictor=court_predictor,  # 初期化済みCourtPredictor
    pose_predictor=pose_predictor,    # 初期化済みPosePredictor
    ball_interval=1,                  # ボール検出の実行間隔（フレーム）
    court_interval=30,                # コート検出の実行間隔（フレーム）
    pose_interval=5,                  # 姿勢推定の実行間隔（フレーム）
    ball_batch_size=1,                # ボール検出のバッチサイズ
    court_batch_size=1,               # コート検出のバッチサイズ
    pose_batch_size=1,                # 姿勢推定のバッチサイズ
)

# 動画処理
predictor.run(
    input_path="input.mp4",        # 入力動画パス
    output_path="output.mp4",      # 出力動画パス
)
```

### 2.6 FramesPredictor (FrameAnnotator)

FrameAnnotatorは、動画をフレームごとに分解し、各フレームに対して全てのPredictorを実行して結果をCOCO形式のJSONファイルとして保存します。

#### 主な機能
- 動画をフレームごとにJPG保存
- 各フレームに対するボール、コート、姿勢の推論結果をCOCO形式で統合
- バッチ処理による高速化

#### 使用方法
```python
annotator = FrameAnnotator(
    ball_predictor=ball_predictor,      # 初期化済みBallPredictor
    court_predictor=court_predictor,    # 初期化済みCourtPredictor
    pose_predictor=pose_predictor,      # 初期化済みPosePredictor
    intervals={"ball": 1, "court": 1, "pose": 1},  # 各タスクの実行間隔
    batch_sizes={"ball": 16, "court": 16, "pose": 16},  # 各タスクのバッチサイズ
    frame_fmt="frame_{:06d}.jpg",       # フレーム保存形式
    ball_vis_thresh=0.5,                # ボール可視化閾値
    court_vis_thresh=0.5,               # コート可視化閾値
    pose_vis_thresh=0.5,                # 姿勢可視化閾値
)

# 動画処理とJSON出力
annotator.run(
    input_path="input.mp4",             # 入力動画パス
    output_dir="output_frames",         # フレーム保存ディレクトリ
    output_json="annotations.json",     # 出力JSONファイルパス
)
```

### 2.7 ImageAnnotator

ImageAnnotatorは、既存の画像ディレクトリに対して全てのPredictorを実行し、結果をCOCO形式のJSONファイルとして保存します。

#### 主な機能
- 画像ディレクトリ内の各画像に対する推論
- ボール、コート、姿勢の結果をCOCO形式で統合
- バッチ処理による高速化

#### 使用方法
```python
annotator = ImageAnnotator(
    ball_predictor=ball_predictor,      # 初期化済みBallPredictor
    court_predictor=court_predictor,    # 初期化済みCourtPredictor
    pose_predictor=pose_predictor,      # 初期化済みPosePredictor
    batch_sizes={"ball": 16, "court": 16, "pose": 16},  # 各タスクのバッチサイズ
    ball_vis_thresh=0.5,                # ボール可視化閾値
    court_vis_thresh=0.5,               # コート可視化閾値
    pose_vis_thresh=0.5,                # 姿勢可視化閾値
)

# 画像ディレクトリ処理とJSON出力
annotator.run(
    input_dir="images_dir",             # 入力画像ディレクトリ
    output_json="annotations.json",     # 出力JSONファイルパス
    image_extensions=[".jpg", ".png"],  # 処理対象の画像拡張子
)
```

## 3. コマンドライン実行方法

Tennis-Analyzerは、Hydraを使用した設定管理システムを採用しており、コマンドラインから各Predictorを簡単に実行できます。

### 基本的な実行方法

```bash
# ボール検出の実行
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4

# コート検出の実行
python -m scripts.infer.infer mode=court input_path=./your_video.mp4

# プレイヤー検出の実行
python -m scripts.infer.infer mode=player input_path=./your_video.mp4

# 姿勢推定の実行
python -m scripts.infer.infer mode=pose input_path=./your_video.mp4

# 複合モード（ボール+コート+姿勢）の実行
python -m scripts.infer.infer mode=multi input_path=./your_video.mp4

# フレーム抽出とJSONL出力
python -m scripts.infer.infer mode=frames input_path=./your_video.mp4 output_jsonl_path=./annotations.jsonl

# 画像ディレクトリに対する処理
python -m scripts.infer.infer mode=image input_path=./images_dir/ output_jsonl_path=./annotations.jsonl
```

### 詳細なパラメータ設定

```bash
# 出力先を指定
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 output_path=./results/ball_track.mp4

# デバイスを指定（GPU/CPU）
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 common.device=cuda

# 半精度推論を有効化
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 common.use_half=true

# バッチサイズを変更
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 common.batch_size=8

# モデルの変更
python -m scripts.infer.infer mode=ball ball=xception input_path=./your_video.mp4

# 検出閾値の変更
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 predictors.ball.threshold=0.7

# 可視化モードの変更
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 predictors.ball.visualize_mode=heatmap
```

### ボール検出の詳細設定と時系列処理

```bash
# フレーム数を変更（デフォルトは3）
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 \
  predictors.ball.num_frames=5

# 連続フレームの処理を効率化するためのバッチサイズ設定
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 \
  common.batch_size=8

# ヒートマップ可視化モードでボールの検出パターンを確認
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 \
  predictors.ball.visualize_mode=heatmap

# 特徴マップ可視化モードでモデルの内部状態を確認
python -m scripts.infer.infer mode=ball input_path=./your_video.mp4 \
  predictors.ball.visualize_mode=features \
  predictors.ball.feature_layer=0  # 特徴抽出層の指定
```

### MultiPredictorの詳細設定

```bash
# 各モジュールの実行間隔とバッチサイズの変更
python -m scripts.infer.infer mode=multi input_path=./your_video.mp4 \
  predictors.multi.ball_interval=2 \
  predictors.multi.court_interval=60 \
  predictors.multi.pose_interval=10 \
  predictors.multi.ball_batch_size=8 \
  predictors.multi.court_batch_size=4 \
  predictors.multi.pose_batch_size=4
```

### フレーム抽出とJSONL出力モード

```bash
# フレームごとに推論し、結果をJSONLとして出力
python -m scripts.infer.infer mode=frames input_path=./your_video.mp4 \
  predictors.frames_annotator.intervals.ball=1 \
  predictors.frames_annotator.intervals.court=10 \
  predictors.frames_annotator.intervals.pose=5 \
  predictors.frames_annotator.output_dir=./output_frames
```

### 画像ディレクトリに対する処理

```bash
# 画像ディレクトリに対して推論を実行し、JSON出力のみを生成
python -m scripts.infer.infer mode=image input_path=./images_dir/ \
  predictors.image_annotator.batch_sizes.ball=16 \
  predictors.image_annotator.batch_sizes.court=16 \
  predictors.image_annotator.batch_sizes.pose=8
```

## 4. 応用例

### 4.1 複数動画の一括処理

```bash
# Bashスクリプトで複数動画を処理
for video in ./videos/*.mp4; do
  python -m scripts.infer.infer mode=multi input_path=$video output_path=./results/$(basename $video)
done
```

### 4.2 特定のモデルバリアントの使用

```bash
# 異なるボール検出モデルを使用
python -m scripts.infer.infer mode=ball ball=xception input_path=./your_video.mp4

# 異なるコート検出モデルを使用
python -m scripts.infer.infer mode=court court=swin input_path=./your_video.mp4

# 異なるプレイヤー検出モデルを使用
python -m scripts.infer.infer mode=player player=conditional_detr input_path=./your_video.mp4
```

### 4.3 高速処理のための設定

```bash
# GPUでの高速処理設定
python -m scripts.infer.infer mode=multi input_path=./your_video.mp4 \
  common.device=cuda \
  common.use_half=true \
  common.batch_size=16 \
  predictors.multi.ball_batch_size=16 \
  predictors.multi.court_batch_size=16 \
  predictors.multi.pose_batch_size=16 \
  predictors.multi.court_interval=30 \
  predictors.multi.pose_interval=10
```

### 4.4 動画からのアノテーションデータ生成

```bash
# 動画からアノテーションデータを生成し、学習データとして保存
python -m scripts.infer.infer mode=frames input_path=./your_video.mp4 \
  output_jsonl_path=./training_data.jsonl \
  predictors.frames_annotator.output_dir=./training_frames \
  predictors.frames_annotator.batch_sizes.ball=32 \
  predictors.frames_annotator.batch_sizes.court=32 \
  predictors.frames_annotator.batch_sizes.pose=16
```

## 5. 注意点と推奨事項

1. **メモリ使用量**: 高解像度動画や大きなバッチサイズを使用する場合、メモリ使用量が増加します。特にGPUメモリが限られている場合は、バッチサイズを調整してください。

2. **モデル選択**: 各タスクに対して複数のモデルバリアントが提供されています。処理速度と精度のバランスに応じて適切なモデルを選択してください。
   - ボール検出: `lite_tracknet`（軽量）、`tracknet`（標準）、`xception`（高精度）
   - コート検出: `lite_tracknet`（軽量）、`fpn`（標準）、`swin`（高精度）
   - プレイヤー検出: `rt_detr`（推奨）、`conditional_detr`（代替）

3. **インターバル設定**: MultiPredictorでは、各タスクの実行間隔を設定できます。リソースを効率的に使用するために、コート検出や姿勢推定などの変化が少ないタスクには大きなインターバルを設定することをお勧めします。

4. **可視化モード**: デバッグや分析目的に応じて、適切な可視化モードを選択してください。「overlay」はシンプルな結果表示に、「heatmap」はモデルの信頼度分布の理解に役立ちます。

5. **出力形式**: 単純な可視化だけでなく、COCO形式のJSONL出力も可能です。これは後続の分析やデータ処理に役立ちます。

6. **ボール追跡の時系列処理**:
   - BallPredictorは複数フレーム（通常T=3フレーム）を入力とする時系列モデルを使用します
   - 入力形式は `List[List[np.ndarray]]` で、各内部リストは連続したT個のフレームを表します
   - 処理の流れ:
     1. 動画フレームからスライディングウィンドウ方式で連続T個のフレームを抽出
     2. B個のクリップ（各クリップはT個のフレーム）をまとめてバッチ処理
     3. モデルは入力 [B, C*T, H, W] を処理し、各クリップのT番目（最後）のフレームについての予測を出力
   - `MultiPredictor`などで使用する場合、各フレームで予測するのではなく、フレームバッファが蓄積されてからT個のフレームを一度に処理します
   - 予測結果の整合性を保つためには、フレームの順序とバッチ処理の方法に注意が必要です
   - 欠損値補間や外れ値除去により、滑らかな軌道追跡が可能になります

7. **半精度推論**: 速度を優先する場合は`common.use_half=true`を使用してください。精度と速度のトレードオフがあります。

## 6. まとめ

Tennis-Analyzerは、テニス映像分析のための強力なツールセットを提供します。各Predictorは特定のタスクに特化しており、それらを組み合わせることで総合的な分析が可能になります。Hydraベースの設定システムにより、柔軟なパラメータ調整と実験が容易になっています。

コマンドラインインターフェースを通じて、モデル選択、パラメータ調整、出力設定などを簡単に制御できるため、研究やアプリケーション開発の両方で活用できます。また、バッチ処理や半精度推論などの最適化オプションにより、大規模なデータセットに対しても効率的に処理を行うことが可能です。

テニス映像の自動分析、プレイヤーの動作解析、コート認識など、さまざまな応用シナリオに対応できる柔軟性を備えています。 