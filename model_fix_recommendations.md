# モデル使用ガイド 🎾

## 問題の原因
現在のエラーは以下が原因です：
- **モデルタイプの不一致**: `wasb_tennis_best.pth.tar` (WASBモデル) を `lite_tracknet` として読み込もうとしている
- **デバイス配置の不一致**: モデルがCPUにあるのに入力データがGPUに送られている

## 💡 推奨解決策

### 1. LiteTrackNetモデルを使用する場合（推奨）
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cuda \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/lite_tracknet_gpu.mp4
```

### 2. WASBモデルを使用する場合
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=wasb_sbdt \
    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \
    model.device=cpu \
    pipeline=memory_efficient \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/wasb_cpu.mp4
```

### 3. Video Swin Transformerモデルを使用する場合
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=video_swin_transformer \
    model.model_path=checkpoints/ball/video_swin_transformer_focal/best_model.ckpt \
    model.device=cuda \
    pipeline=high_performance \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/swin_gpu.mp4
```

## 🚀 バッチサイズ設定例

### 高性能GPU環境
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cuda \
    pipeline.batch_size=32 \
    pipeline.num_workers=8 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/high_perf.mp4
```

### メモリ制限環境
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cpu \
    pipeline.batch_size=4 \
    pipeline.num_workers=2 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/memory_efficient.mp4
```

## 📊 バッチ処理例

### 複数動画の一括処理
```bash
python -m src.predictor.api.batch_process \
    --config-name batch_process \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cuda \
    io.input_dir=datasets/test/ \
    io.output_dir=outputs/batch_results/ \
    batch.parallel_jobs=2 \
    pipeline.batch_size=16 \
    io.report_path=outputs/batch_report.json
```

## 🔧 デバッグモード

### 詳細ログで問題を診断
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cpu \
    pipeline=debug \
    system.log_level=DEBUG \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/debug.mp4
```

## 🎯 最も確実な開始コマンド

まずはこのコマンドで動作確認：
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cpu \
    pipeline.batch_size=1 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/test_cpu.mp4
```

成功したらGPUモードを試す：
```bash
python -m src.predictor.api.inference \
    --config-name inference \
    model.type=lite_tracknet \
    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \
    model.device=cuda \
    pipeline.batch_size=8 \
    io.video=datasets/test/video_input2.mp4 \
    io.output=outputs/ball/test_gpu.mp4
```