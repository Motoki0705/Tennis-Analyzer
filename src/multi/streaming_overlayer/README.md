# 高度拡張性を持つストリーミング処理パイプライン

## 概要

このパッケージは、テニス解析システム向けの高度拡張性を持つストリーミング処理パイプラインを提供します。関心の分離、依存性注入、イベント駆動、宣言的依存関係の4つの設計原則に基づいて構築されており、**バッチ推論機能**を含む柔軟で効率的な処理を実現します。

## アーキテクチャ構成

```
StreamingOverlayer/
├── core/                    # コアインターフェース
│   ├── interfaces.py        # 基本型定義とインターフェース
│   └── __init__.py
├── managers/                # 管理コンポーネント
│   ├── task_manager.py      # タスク配信とオーケストレーション
│   ├── result_manager.py    # 結果収集と配信
│   └── __init__.py
├── input_handlers/          # 入力処理
│   ├── video_file_input_handler.py
│   ├── frame_list_input_handler.py
│   ├── directory_input_handler.py
│   └── __init__.py
├── output_handlers/         # 出力処理
│   ├── video_overlay_output_handler.py
│   ├── json_file_output_handler.py
│   ├── callback_output_handler.py
│   └── __init__.py
├── workers/                 # 処理ワーカー
│   ├── base_worker.py       # バッチ推論対応基底クラス
│   ├── ball_worker.py       # スライディングウィンドウ対応
│   ├── court_worker.py      # 標準バッチ処理
│   ├── event_worker.py      # 依存関係統合処理
│   └── __init__.py
└── __init__.py
```

## 設計原則

### 1. 関心の分離 (Separation of Concerns)
- **Input Handlers**: データ入力の責任
- **Workers**: 個別処理の責任
- **Output Handlers**: 結果出力の責任
- **Managers**: オーケストレーションの責任

### 2. 依存性注入 (Dependency Injection)
- 各コンポーネントは外部から依存関係を注入
- テスト容易性と柔軟性を確保

### 3. イベント駆動 (Event-Driven)
- トピックベースの非同期メッセージング
- 疎結合なコンポーネント間通信

### 4. 宣言的依存関係 (Declarative Dependencies)
- ワーカーが必要な依存関係を宣言
- 自動的な実行順序決定

### 5. **バッチ推論 (Batch Inference)**
- 効率的なGPU利用とスループット向上
- 柔軟なバッチサイズとタイムアウト設定
- 特殊処理（スライディングウィンドウ等）のサポート

## バッチ推論機能

### 基本概念

すべてのワーカーは`BaseWorker`を継承し、バッチ推論機能を持ちます：

```python
class BaseWorker(ABC):
    def __init__(self, name: str, results_queue: queue.Queue,
                 batch_size: int = 1,           # バッチサイズ
                 batch_timeout: float = 0.1,    # バッチタイムアウト
                 task_queue_maxsize: int = 1000): # 大きなキューサイズ
        # ...
    
    @abstractmethod
    def process_batch(self, batch_tasks: List[BatchTask]) -> List[ResultData]:
        """バッチ処理の実装"""
        pass
```

### バッチ処理の種類

#### 1. 標準バッチ処理
個別タスクをバッチサイズまで蓄積してから一括処理

#### 2. スライディングウィンドウ処理
時系列データに対する特殊なバッチ処理

#### 3. 依存関係統合処理
複数の依存関係を統合したバッチ処理

### バッチ設定の推奨値

| ワーカータイプ | バッチサイズ | タイムアウト | 理由 |
|---------------|-------------|-------------|------|
| BallDetection | 8-16 | 0.05s | 高頻度処理、低レイテンシ |
| CourtDetection | 4-8 | 0.1s | 中程度の処理負荷 |
| EventDetection | 2-4 | 0.2s | 複雑な統合処理 |

## クイックスタートガイド

### 基本的な使用例

```python
import queue
from src.multi.streaming_overlayer import (
    TaskManager, ResultManager,
    BallDetectionWorker, CourtDetectionWorker,
    VideoFileInputHandler, VideoOverlayOutputHandler
)

# キューの作成
results_queue = queue.Queue()

# ワーカーの作成（バッチ設定付き）
ball_worker = BallDetectionWorker(
    name="ball_detector",
    predictor=ball_predictor,
    results_queue=results_queue,
    batch_size=8,           # バッチサイズ
    batch_timeout=0.05,     # 50ms タイムアウト
    task_queue_maxsize=1000 # 大きなキューサイズ
)

court_worker = CourtDetectionWorker(
    name="court_detector", 
    predictor=court_predictor,
    results_queue=results_queue,
    batch_size=4,
    batch_timeout=0.1
)

# マネージャーの作成
workers = [ball_worker, court_worker]
task_manager = TaskManager(workers)
result_manager = ResultManager(results_queue)

# 入出力ハンドラーの作成
input_handler = VideoFileInputHandler("input.mp4")
output_handler = VideoOverlayOutputHandler("output.mp4")

# パイプラインの実行
task_manager.start()
result_manager.start()

for worker in workers:
    worker.start()

# データ処理
for frame_id, frame_data in input_handler.get_data():
    task_manager.submit_task(frame_id, frame_data)

# 結果の処理
result_manager.add_output_handler(output_handler)
```

## コンポーネント詳細

### Workers

#### BaseWorker
- **バッチ推論機能**を含む基底クラス
- 統計収集とエラー処理
- 柔軟なバッチサイズとタイムアウト設定

#### BallDetectionWorker
- **スライディングウィンドウ**による時系列処理
- 上流ワーカー（依存関係なし）
- 高頻度処理に最適化

#### CourtDetectionWorker
- **標準バッチ処理**
- 上流ワーカー（依存関係なし）
- 効率的なGPU利用

#### EventDetectionWorker
- **依存関係統合**バッチ処理
- 下流ワーカー（ball, court, pose検出に依存）
- 複雑な多入力処理

### Managers

#### TaskManager
- ワーカーへのタスク配信
- 依存関係に基づく実行順序制御
- **大きなキューサイズ**でバッチ処理をサポート

#### ResultManager  
- 結果の収集と配信
- 複数の出力ハンドラーサポート
- トピックベースのルーティング

## パフォーマンス特徴

### バッチ推論の利点

1. **GPU利用効率の向上**
   - バッチ処理によるGPU並列化
   - メモリ転送コストの削減

2. **スループットの向上**
   - 推奨設定で2-5倍の処理速度向上
   - 大きなキューサイズによる安定した処理

3. **レイテンシの制御**
   - バッチタイムアウトによる最大遅延制限
   - リアルタイム処理要件への対応

### 統計とモニタリング

各ワーカーは詳細な統計を提供：

```python
stats = worker.get_stats()
print(f"処理済みタスク数: {stats['total_processed']}")
print(f"平均処理時間: {stats['average_processing_time']:.3f}s")
print(f"バッチ数: {stats['batch_count']}")
print(f"平均バッチサイズ: {stats['average_batch_size']:.1f}")
print(f"成功率: {stats['success_rate']:.2%}")
```

## カスタマイズ例

### カスタムワーカーの作成

```python
from .base_worker import BaseWorker, BatchTask

class CustomWorker(BaseWorker):
    def __init__(self, name: str, predictor: Any, results_queue: queue.Queue,
                 batch_size: int = 4, batch_timeout: float = 0.1):
        super().__init__(name, results_queue, batch_size=batch_size, 
                        batch_timeout=batch_timeout, task_queue_maxsize=1000)
        self.predictor = predictor
    
    def get_published_topic(self) -> str:
        return "custom_detection"
    
    def get_dependencies(self) -> List[str]:
        return ["ball_detection"]  # 依存関係を宣言
    
    def process_batch(self, batch_tasks: List[BatchTask]) -> List[ResultData]:
        # バッチ処理の実装
        batch_data = [task.task_data for task in batch_tasks]
        predictions = self.predictor.inference_batch(batch_data)
        
        results = []
        for task, prediction in zip(batch_tasks, predictions):
            result = self._postprocess(prediction, task.dependencies)
            results.append(result)
        
        return results
```

## トラブルシューティング

### よくある問題と解決策

#### 1. バッチサイズが小さすぎる
**症状**: GPU利用率が低い、処理速度が遅い
**解決策**: バッチサイズを増やす、タイムアウトを調整

#### 2. メモリ不足
**症状**: CUDA out of memory エラー
**解決策**: バッチサイズを減らす、キューサイズを調整

#### 3. レイテンシが高い
**症状**: リアルタイム処理で遅延が発生
**解決策**: バッチタイムアウトを短縮、バッチサイズを調整

#### 4. 依存関係エラー
**症状**: 下流ワーカーで依存関係不足エラー
**解決策**: 依存関係の宣言を確認、上流ワーカーの動作確認

## 今後の拡張予定

1. **動的バッチサイズ調整**: 負荷に応じた自動調整
2. **分散処理サポート**: 複数マシンでの処理
3. **ストリーミング最適化**: より低レイテンシな処理
4. **メトリクス収集**: Prometheus/Grafana連携
5. **設定管理**: YAML/JSON設定ファイル対応

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。 