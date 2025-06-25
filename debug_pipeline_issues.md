# パイプライン問題のデバッグと解決策 🔧

## 🚨 現在の問題

### 1. Visualization Worker Error
```
[ERROR] - Visualization worker error: 'frame_number'
```
**原因**: メタデータに`frame_number`キーが不足している

### 2. Batch Inference Error
```
[ERROR] - Batch inference error: 
```
**原因**: エラー詳細が空文字列で表示されていない

### 3. Queue Overflow
```
[WARNING] - Render queue full in postprocessor 0
[WARNING] - Tensor queue full in preprocessor
```
**原因**: パイプライン設定が高性能すぎてボトルネックが発生

### 4. Thread Failures
```
[ERROR] - Thread reader failed
[ERROR] - Thread preprocessor_X failed
```
**原因**: 上記の問題が連鎖してワーカースレッドが失敗

## 💡 即座の解決策

### 1. デバッグモードで実行
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cuda \
    pipeline=debug \
    system.log_level=DEBUG \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_debug.mp4
```

### 2. メモリ効率設定で実行
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cuda \
    pipeline=memory_efficient \
    pipeline.batch_size=1 \
    pipeline.num_workers=2 \
    pipeline.queue_size=10 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_safe.mp4
```

### 3. CPU版で確実に動作確認
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cpu \
    pipeline.batch_size=1 \
    pipeline.num_workers=1 \
    pipeline.queue_size=5 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_cpu_safe.mp4
```

## 🔍 根本原因の診断

### 現在のhigh_performance設定の問題
```yaml
# 現在の設定（高負荷すぎる）
batch_size: 16      # ← 大きすぎる
num_workers: 8      # ← 多すぎる  
queue_size: 200     # ← 大きすぎる
```

### 推奨設定
```yaml
# WASB-SBDT用最適化設定
batch_size: 2       # HRNetは重いモデル
num_workers: 2      # GPUボトルネック考慮
queue_size: 20      # メモリ使用量削減
```

## 🎯 段階的テスト手順

### Step 1: 最小設定で動作確認
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cpu \
    pipeline.batch_size=1 \
    pipeline.num_workers=1 \
    pipeline.queue_size=5 \
    system.log_level=DEBUG \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/step1_minimal.mp4
```

### Step 2: GPU + 小設定
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cuda \
    pipeline.batch_size=1 \
    pipeline.num_workers=1 \
    pipeline.queue_size=10 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/step2_gpu_small.mp4
```

### Step 3: 設定を段階的に上げる
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cuda \
    pipeline.batch_size=2 \
    pipeline.num_workers=2 \
    pipeline.queue_size=20 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/step3_optimized.mp4
```

## 🛠️ 代替モデルでのテスト

LiteTrackNetで正常動作を確認：
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cuda \
    pipeline.batch_size=8 \
    pipeline.num_workers=4 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/lite_comparison.mp4
```

## 📊 パフォーマンス比較

| 設定 | バッチサイズ | ワーカー数 | 推定速度 | 安定性 |
|------|-------------|-----------|----------|--------|
| high_performance | 16 | 8 | 高速 | ❌ 不安定 |
| memory_efficient | 4 | 2 | 中速 | ✅ 安定 |
| debug | 1 | 1 | 低速 | ✅ 最安定 |
| カスタム推奨 | 2 | 2 | 中速 | ✅ 安定 |

## 🎯 推奨開始コマンド

**最も確実**:
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cpu \
    pipeline=debug \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_safe_start.mp4
```