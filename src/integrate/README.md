# Flexible Tennis Analysis Pipeline

拡張性の高いモジュラーなテニス解析パイプラインです。従来のモノリシックな設計から、タスクベースのプラグイン型アーキテクチャに改善されました。

## 主な改善点

### 1. モジュラーアーキテクチャ
- **BaseTask**: 全てのタスクが継承する抽象基底クラス
- **TaskManager**: タスクの登録、依存関係解決、実行管理
- **DataFlow**: データフローとバッファリング管理
- **FlexiblePipeline**: メインパイプライン実行エンジン

### 2. 新機能
- ✅ **設定駆動型**: YAMLファイルでタスクを柔軟に設定
- ✅ **依存関係管理**: タスク間の依存関係を自動解決
- ✅ **プラグイン型**: 新しいタスクを簡単に追加可能
- ✅ **マルチスレッド対応**: シングル/マルチスレッドモードを選択可能
- ✅ **エラー処理**: クリティカルでないタスクの失敗時に継続実行
- ✅ **パフォーマンス監視**: 実行時間とメモリ使用量の追跡

## アーキテクチャ

```
src/integrate/
├── core/                    # コアコンポーネント
│   ├── base_task.py        # タスクベースクラス
│   ├── task_manager.py     # タスク管理
│   ├── data_flow.py        # データフロー管理
│   ├── flexible_pipeline.py # メインパイプライン
│   └── video_io.py         # ビデオI/O
├── tasks/                   # タスク実装
│   ├── court_task.py       # コート検出タスク
│   ├── ball_task.py        # ボール追跡タスク
│   ├── player_task.py      # 選手検出タスク
│   └── pose_task.py        # 姿勢推定タスク
└── demo_flexible_pipeline.py # デモスクリプト
```

## 使用方法

### 基本的な使用例

```bash
# 基本実行
python src/integrate/demo_flexible_pipeline.py io.input_video=/path/to/video.mp4

# 出力ビデオを指定
python src/integrate/demo_flexible_pipeline.py \
    io.input_video=/path/to/video.mp4 \
    io.output_video=/path/to/output.mp4

# 特定のタスクのみを有効化
python src/integrate/demo_flexible_pipeline.py \
    io.input_video=/path/to/video.mp4 \
    tasks.0.enabled=true \
    tasks.1.enabled=false \
    tasks.2.enabled=false \
    tasks.3.enabled=false
```

### マルチスレッドモード

```bash
python src/integrate/demo_flexible_pipeline.py \
    io.input_video=/path/to/video.mp4 \
    threading.mode=multi \
    batch_size=8 \
    threading.queue_size=100
```

## 設定ファイル

`configs/infer/integrate/flexible_pipeline.yaml` で全ての設定を管理：

```yaml
# タスク設定
tasks:
  - name: court_detection
    module: src.integrate.tasks.court_task
    class_name: CourtDetectionTask
    enabled: true
    critical: true
    dependencies: []
    config:
      checkpoint: checkpoints/court/lite_tracknet_focal.ckpt
      score_threshold: 0.5

  - name: pose_estimation  
    module: src.integrate.tasks.pose_task
    class_name: PoseEstimationTask
    enabled: true
    critical: false
    dependencies: [player_detection]  # 選手検出に依存
    config:
      keypoint_threshold: 0.3
```

## 新しいタスクの追加

1. **BaseTaskを継承したクラスを作成**:

```python
class CustomTask(BaseTask):
    def initialize(self):
        # モデル初期化
        pass
        
    def preprocess(self, frames, metadata=None):
        # 前処理
        return processed_data, meta
        
    def inference(self, preprocessed_data, metadata):
        # 推論
        return raw_outputs
        
    def postprocess(self, raw_outputs, metadata):
        # 後処理
        return results
        
    def visualize(self, frame, results, vis_config):
        # 可視化
        return frame_with_viz
```

2. **設定ファイルに追加**:

```yaml
tasks:
  - name: custom_analysis
    module: src.integrate.tasks.custom_task
    class_name: CustomTask
    enabled: true
    dependencies: [court_detection]
    config:
      param1: value1
```

## パフォーマンス比較

| 機能 | 従来のpipeline_demo.py | 新しいFlexiblePipeline |
|------|------------------------|----------------------|
| 新タスク追加 | 大幅なコード修正が必要 | 設定ファイルのみ |
| 依存関係管理 | 手動で順序管理 | 自動解決 |
| エラー処理 | 1つ失敗で全停止 | 継続実行可能 |
| 並列化 | 固定3スレッド | 柔軟なスレッド管理 |
| 設定変更 | コード修正必要 | YAML設定のみ |
| テスト | モノリシックで困難 | 各タスク独立テスト |

## 利点

1. **拡張性**: 新しいタスクを簡単に追加
2. **保守性**: モジュラー設計で保守が容易
3. **再利用性**: タスクを他のパイプラインでも利用可能
4. **柔軟性**: 設定ファイルで動作を制御
5. **堅牢性**: エラー処理と依存関係管理
6. **パフォーマンス**: 最適化されたスレッド管理

この新しいアーキテクチャにより、テニス解析システムの機能拡張と保守が大幅に改善されます。