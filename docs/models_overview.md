# Tennis Analyzer Models Overview

本ドキュメントでは、Tennis Analyzerプロジェクトで使用されている5つのタスク（Ball, Court, Player, Pose, Event）のモデル仕様と実装詳細について説明します。

## 目次

1. [Ball Detection（ボール検出）](#ball-detection)
2. [Court Detection（コート検出）](#court-detection)
3. [Player Detection（プレイヤー検出）](#player-detection)
4. [Pose Estimation（姿勢推定）](#pose-estimation)
5. [Event Detection（イベント検出）](#event-detection)
6. [モデル設計方針](#model-design-principles)

---

## Ball Detection

### 概要
ボールの位置をヒートマップで特定するタスクです。3フレームの入力から最後のフレームにおけるボールの位置を推定します。

### 利用可能なモデル

#### 1. LiteTrackNet
**メインモデル**: 軽量なU-Net風アーキテクチャ

**仕様:**
- **入力**: `(B, 9, H, W)` - 3フレーム × 3チャンネル（RGB）
- **出力**: `(B, H, W)` - ヒートマップ形式（最後のフレームのボール位置）
- **特徴**: 
  - DSConv（Depthwise Separable Convolution）
  - SE Block（Squeeze-and-Excitation）
  - PixelShuffle使用の効率的アップサンプリング
  - ResNet風のスキップ接続

**アーキテクチャ:**
```
Encoder: 16→32→64→128 (stride 1,2,2,2)
Bottleneck: 256
Decoder: 128→64→32 (PixelShuffle × 2)
Head: 32+16→16→1
```

#### 2. Video Swin Transformer
**高精度モデル**: ビデオベースのTransformerアーキテクチャ

**仕様:**
- **入力**: `(B, N, C, H, W)` - マルチフレーム対応
- **出力**: `(B, H, W)` - ヒートマップ形式
- **特徴**:
  - 時空間Attention機構
  - Window-based処理
  - SpatioTemporalSwinBlock使用

**Lightning Module**: `src.ball.lit_module.lit_lite_tracknet_focal.LitLiteTracknetFocalLoss`
**チェックポイント**: `checkpoints/ball/lite_tracknet/lite_tracknet-epoch=46-val_loss=0.0000.ckpt`
**設定ファイル**: `configs/infer/ball/lite_tracknet.yaml`

**使用方法:**
```python
from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss

# チェックポイントからモデルを読み込み
model = LitLiteTracknetFocalLoss.load_from_checkpoint(
    "checkpoints/ball/lite_tracknet/lite_tracknet-epoch=46-val_loss=0.0000.ckpt"
)
model.eval()

# 推論
with torch.no_grad():
    output = model(input_tensor)  # (B, 9, H, W) -> (B, H, W)
```

---

## Court Detection

### 概要
テニスコートのキーポイントを検出するタスクです。単一画像から最大15個のキーポイントを1枚のヒートマップから特定します。

### モデル仕様

#### LiteTrackNet（Court用）
**仕様:**
- **入力**: `(B, 3, H, W)` - 単一RGB画像
- **出力**: `(B, 1, H, W)` - 統合ヒートマップ
- **キーポイント検出**: 最大15個のコートキーポイントをピーク検出で特定
- **アーキテクチャ**: Ball用LiteTrackNetと同様だが出力チャンネルが異なる

**処理フロー:**
1. 単一画像を入力
2. 統合ヒートマップを生成
3. ピーク検出アルゴリズムで15個のキーポイント座標を抽出

**Lightning Module**: `src.court.lit_module.lit_lite_tracknet_focal.LitLiteTracknetFocal`
**チェックポイント**: `checkpoints/court/lite_tracknet/court-epoch=07-val_loss=0.000247.ckpt`
**設定ファイル**: `configs/infer/court/lite_tracknet.yaml`

**使用方法:**
```python
from src.court.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocal

# チェックポイントからモデルを読み込み
model = LitLiteTracknetFocal.load_from_checkpoint(
    "checkpoints/court/lite_tracknet/court-epoch=07-val_loss=0.000247.ckpt"
)
model.eval()

# 推論
with torch.no_grad():
    output = model(input_tensor)  # (B, 3, H, W) -> (B, 1, H, W)
    # ピーク検出でキーポイント座標を抽出
    keypoints = extract_keypoints_from_heatmap(output)
```

---

## Player Detection

### 概要
テニスプレイヤーの検出を行うタスクです。RT-DETRをファインチューニングしてプレイヤーのみを検出可能にしたモデルです。

### モデル仕様

#### RT-DETR (Real-Time Detection Transformer)
**仕様:**
- **ベースモデル**: `"PekingU/rtdetr_v2_r18vd"`
- **入力**: RGB画像
- **出力**: バウンディングボックス + 信頼度スコア
- **クラス数**: 1（プレイヤーのみ）

**特徴:**
- Transformerベースの物体検出
- リアルタイム推論対応
- 段階的学習戦略（バックボーン凍結/解凍）
- 異なる学習率設定
  - バックボーン: `lr_backbone = 1e-5`
  - その他: `lr = 1e-4`

**学習戦略:**
```python
# 初期エポック: バックボーン凍結
if epoch < num_freeze_epoch:
    freeze_backbone()
else:
    unfreeze_backbone()
```

**Lightning Module**: `src.player.lit_module.lit_rtdetr.LitRtdetr`
**チェックポイント**: `checkpoints/player/rt_detr_tracking/best-epoch=05-val_loss_total=2.2782.ckpt`
**設定ファイル**: `configs/infer/player/rt_detr.yaml`

**使用方法:**
```python
from src.player.lit_module.lit_rtdetr import LitRtdetr

# チェックポイントからモデルを読み込み
model = LitRtdetr.load_from_checkpoint(
    "checkpoints/player/rt_detr_tracking/best-epoch=05-val_loss_total=2.2782.ckpt"
)
model.eval()

# 推論
with torch.no_grad():
    outputs = model(pixel_values=input_tensor)
    # バウンディングボックスと信頼度スコアを取得
    boxes = outputs.prediction_boxes
    scores = outputs.prediction_scores
```

---

## Pose Estimation

### 概要
プレイヤー検出結果を基に姿勢推定を行うタスクです。VitPoseプロセッサにプレイヤー検出結果を入力して姿勢推定を実行します。

### モデル仕様

#### VitPose (Vision Transformer for Pose Estimation)
**仕様:**
- **ベースモデル**: `"usyd-community/vitpose-base-simple"`
- **入力**: プレイヤー検出結果（切り抜き画像）
- **出力**: キーポイント座標（関節位置）
- **アーキテクチャ**: Vision Transformerベース

**処理フロー:**
1. Player Detectionでプレイヤー領域を検出
2. 検出領域を切り抜いてVitPoseプロセッサに入力
3. 各プレイヤーの姿勢キーポイントを取得

**ベースモデル**: `"usyd-community/vitpose-base-simple"`

**使用方法:**
```python
from transformers import VitPoseForPoseEstimation

# 事前学習済みモデルを読み込み
model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-base-simple"
)
model.eval()

# プレイヤー検出結果から切り抜いた画像で推論
with torch.no_grad():
    outputs = model(cropped_player_images)
    keypoints = outputs.prediction_keypoints  # 関節位置を取得
```

---

## Event Detection

### 概要
ボール、コート、プレイヤー情報を統合してテニスイベント（hit, bounce）を検出するマルチモーダル時系列分析タスクです。

### モデル仕様

#### EventTransformerV2
**マルチモーダル時系列Transformer**

**入力データ形式:**
- **Ball特徴**: `[T, 3]` - (正規化x, 正規化y, 信頼度スコア)
- **Court特徴**: `[T, 45]` - 15キーポイント × (正規化x, 正規化y, 可視性)  
- **Player BBox**: `[T, max_players, 5]` - (x1, y1, x2, y2, スコア)
- **Player Pose**: `[T, max_players, K*3]` - K個の関節 × (x, y, 可視性)

**出力:**
- `[T, 2]` - (hit確率, bounce確率)

**アーキテクチャの特徴:**
1. **モダリティ別埋め込み**:
   - Ball: 3次元 → d_model次元
   - Court: 45次元 → d_model次元  
   - Player: (5+pose_dim)次元 → d_model次元

2. **Attention Pooling**: 複数プレイヤー情報を統合
   ```python
   player_attn_q = nn.Parameter(torch.zeros(1, 1, d_model))  # 学習可能クエリ
   ```

3. **モダリティ別Transformer**: 各特徴に専用エンコーダ適用

4. **Cross-Attention融合**: 
   ```python
   cross_query = nn.Parameter(torch.zeros(1, 1, d_model))  # 融合用クエリ
   ```

5. **スムージングターゲット**: ガウス分布でイベントラベルを時間方向に拡散

**データセット処理:**
- スライディングウィンドウでシーケンス生成
- フレームスキップによるデータ拡張
- 動的パディングで可変プレイヤー数に対応

**Lightning Module**: `src.event.lit_module.lit_transformer_v2.LitTransformerV2`
**チェックポイント**: `checkpoints/event/transformer_v2/epoch=18-step=532.ckpt`
**設定ファイル**: `configs/infer/event/transformer_v2.yaml`

**使用方法:**
```python
from src.event.lit_module.lit_transformer_v2 import LitTransformerV2

# チェックポイントからモデルを読み込み
model = LitTransformerV2.load_from_checkpoint(
    "checkpoints/event/transformer_v2/epoch=18-step=532.ckpt"
)
model.eval()

# マルチモーダル時系列データで推論
with torch.no_grad():
    outputs = model(
        ball=ball_features,        # [T, 3]
        player_bbox=player_bbox,   # [T, max_players, 5]
        player_pose=player_pose,   # [T, max_players, K*3]
        court=court_features       # [T, 45]
    )
    event_probs = torch.softmax(outputs, dim=-1)  # [T, 2] (hit, bounce)
```

---

## 統合テニス動画解析システム

### システム概要
上記5つのモデルを組み合わせることで、テニス動画の包括的な解析システムを構築予定です。

### 処理フロー
1. **Ball Detection**: 3フレームからボール位置をヒートマップで検出
2. **Court Detection**: 単一フレームからコートキーポイントを検出
3. **Player Detection**: RT-DETRでプレイヤー領域を検出
4. **Pose Estimation**: 検出されたプレイヤー領域から姿勢を推定
5. **Event Detection**: 統合された特徴量から hit/bounce イベントを分類

※ Pose Estimationは事前学習済みHugging Faceモデルを使用

---

## Model Design Principles

### 共通設計方針

#### 1. 効率性重視
- **軽量アーキテクチャ**: LiteTrackNetでDSConv、SE Block活用
- **メモリ効率**: PixelShuffleによる効率的アップサンプリング
- **リアルタイム対応**: RT-DETRによる高速物体検出

#### 2. Transformer活用
- **高精度タスク**: Video Swin Transformer、VitPose
- **マルチモーダル**: EventTransformerV2で異種データ統合
- **Attention機構**: 関連性の高い特徴に注力

#### 3. 段階的学習戦略
- **Transfer Learning**: 事前学習済みモデルの活用
- **段階的解凍**: バックボーン凍結→解凍による安定学習
- **適応的学習率**: モジュール別の学習率設定

#### 4. マルチモーダル統合
- **時系列Context**: 複数フレーム情報の活用
- **空間情報**: ボール、コート、プレイヤーの位置関係
- **階層的特徴**: 低レベル（位置）→高レベル（イベント）

### タスク別最適化

| タスク | 入力形式 | 出力形式 | 主要技術 |
|--------|----------|----------|----------|
| Ball | 3フレーム画像 | ヒートマップ | DSConv + SE |
| Court | 単一画像 | 統合ヒートマップ | ピーク検出 |
| Player | 単一画像 | BBox + スコア | RT-DETR |
| Pose | 切り抜き画像 | キーポイント | VitPose |
| Event | マルチモーダル時系列 | イベント分類 | Cross-Attention |

### パフォーマンス特性

- **リアルタイム処理**: Player Detection（RT-DETR）
- **高精度**: Ball Detection（Video Swin Transformer）、Pose Estimation（VitPose）
- **軽量化**: Ball/Court Detection（LiteTrackNet）
- **統合分析**: Event Detection（マルチモーダルTransformer）

この設計により、テニス動画の包括的分析（ボール軌道、コート認識、プレイヤー検出、姿勢推定、イベント分類）を効率的かつ高精度に実現しています。

### 統合システム利用例

```python
import torch
from src.ball.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocalLoss
from src.court.lit_module.lit_lite_tracknet_focal import LitLiteTracknetFocal
from src.player.lit_module.lit_rtdetr import LitRtdetr
from src.event.lit_module.lit_transformer_v2 import LitTransformerV2
from transformers import VitPoseForPoseEstimation

class TennisAnalysisSystem:
    def __init__(self):
        # 各モデルをチェックポイントから読み込み
        self.ball_model = LitLiteTracknetFocalLoss.load_from_checkpoint(
            "checkpoints/ball/lit_lite_tracknet/best_model.ckpt"
        )
        self.court_model = LitLiteTracknetFocal.load_from_checkpoint(
            "checkpoints/court/lit_lite_tracknet/epoch=010-val_loss=0.76632285.ckpt"
        )
        self.player_model = LitRtdetr.load_from_checkpoint(
            "checkpoints/player/lit_rt_detr/epoch=002-val_loss_total=2.38540339.ckpt"
        )
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple"
        )
        self.event_model = LitTransformerV2.load_from_checkpoint(
            "checkpoints/event/transformer_v2.py/epoch=18-step=532.ckpt"
        )
        
        # 評価モードに設定
        for model in [self.ball_model, self.court_model, self.player_model, 
                      self.pose_model, self.event_model]:
            model.eval()
    
    def analyze_tennis_video(self, video_frames):
        """テニス動画の包括的解析"""
        results = {}
        
        with torch.no_grad():
            # 1. ボール検出（3フレーム単位）
            ball_positions = []
            for i in range(len(video_frames) - 2):
                three_frames = torch.stack(video_frames[i:i+3], dim=1)  # (B, 9, H, W)
                ball_heatmap = self.ball_model(three_frames)
                ball_positions.append(ball_heatmap)
            
            # 2. コート検出（各フレーム）
            court_keypoints = []
            for frame in video_frames:
                court_heatmap = self.court_model(frame)
                keypoints = extract_keypoints_from_heatmap(court_heatmap)
                court_keypoints.append(keypoints)
            
            # 3. プレイヤー検出（各フレーム）
            player_detections = []
            for frame in video_frames:
                player_output = self.player_model(pixel_values=frame)
                player_detections.append(player_output)
            
            # 4. 姿勢推定（検出されたプレイヤー領域）
            pose_results = []
            for detection in player_detections:
                cropped_players = crop_players_from_detection(detection)
                poses = self.pose_model(cropped_players)
                pose_results.append(poses)
            
            # 5. イベント検出（統合特徴量）
            event_features = prepare_event_features(
                ball_positions, court_keypoints, player_detections, pose_results
            )
            event_predictions = self.event_model(
                ball=event_features['ball'],
                player_bbox=event_features['player_bbox'],
                player_pose=event_features['player_pose'],
                court=event_features['court']
            )
        
        return {
            'ball_positions': ball_positions,
            'court_keypoints': court_keypoints,
            'player_detections': player_detections,
            'pose_results': pose_results,
            'event_predictions': event_predictions
        }

# システムの初期化と使用
system = TennisAnalysisSystem()
results = system.analyze_tennis_video(video_frames)
```

このシステムにより、テニス動画から以下の情報を統合的に取得できます：
- ボールの軌道追跡
- コートライン・キーポイントの認識
- プレイヤーの位置と動作
- 各プレイヤーの姿勢・関節位置
- hit/bounceイベントのタイミング検出 