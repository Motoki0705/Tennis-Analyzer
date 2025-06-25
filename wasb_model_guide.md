# WASB-SBDT モデル使用ガイド 🎾

## WASB-SBDTモデルの詳細

**ファイル構造解析結果:**
- **モデルファイル**: `third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar`
- **実際のアーキテクチャ**: HRNet (High-Resolution Network)
- **設定ファイル**: `third_party/WASB_SBDT/src/configs/model/wasb.yaml`
- **入力サイズ**: 288×512 (3フレーム)
- **出力サイズ**: 288×512 (3フレーム)

## 📋 利用可能なWASBモデル

### WASB-SBDT内のモデル一覧:
1. **tracknetv2** - TrackNetV2 (U-Net based)
2. **monotrack** - MonoTrack 
3. **restracknetv2** - ChangsTrackNet (ResU-Net)
4. **hrnet** - HRNet (High-Resolution Network) ← **WASBはこれ**
5. **deepball** - DeepBall
6. **ballseg** - BallSeg

## 🎯 正しいWASBモデル使用方法

### 1. HRNetとして使用する場合 (推奨)
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.config_path=third_party/WASB_SBDT/src/configs/model/wasb.yaml \
    model.device=cpu \
    pipeline.batch_size=1 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_hrnet.mp4
```

### 2. CPU最適化版
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cpu \
    pipeline=memory_efficient \
    pipeline.batch_size=1 \
    pipeline.num_workers=1 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_cpu_optimized.mp4
```

### 3. GPU版 (メモリに注意)
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cuda \
    pipeline.batch_size=2 \
    pipeline.num_workers=2 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_gpu.mp4
```

## 🔧 デバッグ用コマンド

### 詳細ログでWASBモデルをテスト
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cpu \
    pipeline=debug \
    system.log_level=DEBUG \
    pipeline.batch_size=1 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_debug.mp4 \
    io.stats_output=outputs/ball/wasb_stats.json
```

## ⚠️ 重要な注意点

### WASBモデルの特徴:
1. **入力フォーマット**: 3フレーム連続 (288×512)
2. **メモリ使用量**: HRNetは比較的重い
3. **GPU要件**: 小さなバッチサイズから始める
4. **デバイス統一**: モデルと入力データのデバイスを統一する

### トラブルシューティング:
- **メモリエラー**: `pipeline.batch_size=1`に減らす
- **デバイスエラー**: `model.device=cpu`を使用
- **読み込みエラー**: 設定ファイルパスを指定

## 🚀 最も安全な開始コマンド

```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cpu \
    pipeline.batch_size=1 \
    pipeline.num_workers=1 \
    pipeline.queue_size=10 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_safe.mp4
```

## 📊 モデル比較

| モデル | ファイルサイズ | 推奨用途 | GPU必要 |
|--------|----------------|----------|---------|
| LiteTrackNet | 8.8MB | 高速処理 | ○ |
| Video Swin | 35.5MB | 高精度 | ○ |
| WASB-SBDT | 5.8MB | バランス | △ |

**推奨**: 最初はLiteTrackNetで動作確認後、WASBを試すのが良い順序です。