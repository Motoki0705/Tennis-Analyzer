# 高度拡張性を持つストリーミング処理パイプライン - Core アーキテクチャ

## 概要

本アーキテクチャは、特定のアプリケーション（動画へのオーバーレイ描画）から独立し、様々な入出力や複雑な処理フローに対応可能な、**汎用的かつ再利用可能なデータ処理フレームワーク**です。

## 設計原則

- **関心の分離 (Separation of Concerns)**: 各コンポーネントは単一の明確な責任を持つ
- **依存性の注入 (Dependency Injection)**: コンポーネントは具体的な実装に依存せず、抽象的なインターフェースに依存する
- **イベント駆動 (Event-Driven)**: コンポーネント間の連携は、非同期なイベント（タスク発行、結果通知）によって駆動される
- **宣言的な依存関係**: ワーカー間の処理依存性は、コード上で明示的に宣言され、一元管理される

## コアコンポーネント

### 1. インターフェース (`interfaces.py`)

#### `BaseWorker`
- 個別の処理を実行するワーカーの基底クラス
- 依存関係の宣言（`get_dependencies()`）とトピックベースの結果パブリッシュをサポート
- 内部でスレッドプールを管理し、並列処理を実現

#### `InputHandler`
- データソースから処理単位（データアイテム）を取り出すハンドラの基底クラス
- 様々な入力形式を統一的なインターフェースで扱える

#### `OutputHandler`
- パイプラインの最終成果物を処理するハンドラの基底クラス
- 様々な出力形式を統一的なインターフェースで扱える

### 2. タスクマネージャー (`task_manager.py`)

#### `TaskManager`
- ワーカー間の依存関係を管理し、タスクの発行を制御
- 主要機能：
  - 依存関係グラフの自動構築と検証（循環依存の検出）
  - 依存関係に基づいたタスクの順次発行
  - バックプレッシャーの管理

### 3. 結果マネージャー (`result_manager.py`)

#### `ResultManager`
- ワーカーからの結果を集約し、最終成果物の生成を管理
- 主要機能：
  - ワーカーからの結果受信と保存
  - TaskManagerへの結果利用可能通知
  - 最終成果物の完成判定
  - キャッシュサイズ管理

### 4. パイプラインランナー (`pipeline_runner.py`)

#### `PipelineRunner`
- パイプライン全体のライフサイクルを管理するオーケストレーター
- 主要機能：
  - 各コンポーネントの初期化と起動
  - 入力データの読み込みとTaskManagerへの送信
  - ResultManagerからの完成結果の受信
  - OutputHandlerを通じた結果の出力
  - 全コンポーネントの安全な終了

## I/Oハンドラー実装

### InputHandler実装例
- `VideoFileInputHandler`: 動画ファイルからフレームを読み込み
- `FrameListInputHandler`: メモリ上のフレームリストから提供
- `DirectoryInputHandler`: ディレクトリ内の画像ファイルを順次読み込み

### OutputHandler実装例
- `VideoOverlayOutputHandler`: 結果をフレームにオーバーレイして動画出力
- `JsonFileOutputHandler`: 結果をJSONファイルに保存
- `CallbackOutputHandler`: 結果をコールバック関数に渡す

## ワーカー実装例

### 基本的なワーカー
- `BallDetectionWorker`: ボール検出処理（上流ワーカー、依存関係なし）
- `CourtDetectionWorker`: コート検出処理（上流ワーカー、依存関係なし）
- `PoseDetectionWorker`: ポーズ検出処理（上流ワーカー、依存関係なし）
- `EventDetectionWorker`: イベント検出処理（下流ワーカー、上記3つに依存）

### テスト用ワーカー
- `MockWorker`: ダミーの処理を行うテスト用ワーカー

## 使用例

### 基本的な使用方法

```python
from src.multi.streaming_overlayer.core import PipelineRunner
from src.multi.streaming_overlayer.core.handlers import VideoFileInputHandler
from src.multi.streaming_overlayer.core.handlers_output import VideoOverlayOutputHandler

# 1. I/Oハンドラーの作成
input_handler = VideoFileInputHandler("input_video.mp4")
output_handler = VideoOverlayOutputHandler("output_video.mp4")

# 2. ワーカーの作成
workers = [
    BallDetectionWorker(ball_predictor, results_queue),
    CourtDetectionWorker(court_predictor, results_queue),
    PoseDetectionWorker(pose_predictor, results_queue),
    EventDetectionWorker(event_predictor, results_queue)
]

# 3. パイプラインの構築と実行
pipeline = PipelineRunner(
    input_handler=input_handler,
    output_handler=output_handler,
    workers=workers,
    debug=True
)

pipeline.run()
```

### 依存関係の宣言例

```python
class EventDetectionWorker(BaseWorker):
    def get_published_topic(self) -> str:
        return "event_result"
    
    def get_dependencies(self) -> List[str]:
        return ["ball_result", "court_result", "pose_result"]
    
    def process_task(self, item_id, task_data, dependencies):
        # dependencies にball_result、court_result、pose_resultが含まれる
        ball_data = dependencies["ball_result"]
        court_data = dependencies["court_result"]
        pose_data = dependencies["pose_result"]
        
        # 統合処理を実行
        return self.predictor.predict(ball_data, court_data, pose_data)
```

## 実行フロー

1. **初期化**: PipelineRunnerが各コンポーネントを初期化
2. **依存関係解析**: TaskManagerがワーカーの依存関係グラフを構築
3. **入力処理**: InputHandlerからデータアイテムを順次取得
4. **タスク発行**: TaskManagerが依存関係のない上流ワーカーにタスクを発行
5. **結果処理**: ResultManagerがワーカーからの結果を受信し、TaskManagerに通知
6. **依存関係解決**: TaskManagerが依存関係を解決し、次のタスクを発行
7. **最終成果物生成**: ResultManagerが全結果を集約し、最終成果物を生成
8. **出力処理**: OutputHandlerが最終成果物を処理

## ファイル構成

```
src/multi/streaming_overlayer/core/
├── __init__.py                 # モジュール初期化
├── interfaces.py               # 基底インターフェース
├── task_manager.py             # タスク管理
├── result_manager.py           # 結果管理
├── pipeline_runner.py          # パイプライン制御
├── handlers.py                 # InputHandler実装
├── handlers_output.py          # OutputHandler実装
├── example_workers.py          # ワーカー実装例
└── README.md                   # 本文書

examples/
└── simple_pipeline_demo.py     # 使用例デモ
```

## 特徴と利点

### 1. 高い拡張性
- 新しいワーカーの追加が依存関係の宣言だけで可能
- 入出力形式の追加が簡単

### 2. 柔軟な依存関係管理
- 複雑な依存関係も自動的に解決
- 循環依存の検出とエラー報告

### 3. 堅牢なエラー処理
- 各レベルでの例外処理とログ記録
- ワーカーの個別障害がパイプライン全体に影響しない設計

### 4. パフォーマンス最適化
- 並列処理による高いスループット
- バックプレッシャー制御によるメモリ使用量の最適化
- 結果キャッシュサイズの管理

### 5. デバッグとモニタリング
- 詳細な統計情報の提供
- 依存関係グラフの可視化
- 処理時間とエラー率の追跡

## テストとデモ

実装されたデモプログラム（`examples/simple_pipeline_demo.py`）では、以下の機能をテストできます：

1. **シンプルなパイプライン実行**: 基本的なワーカー連携
2. **ファイル出力**: JSON形式での結果保存
3. **依存関係解析**: 複雑な依存関係グラフの構築と可視化

## 今後の拡張予定

1. **リアルタイム処理**: ストリーミングデータの低遅延処理
2. **分散処理**: 複数マシンでのワーカー分散実行
3. **設定ベース構築**: YAML/JSON設定からのパイプライン自動構築
4. **Webインターフェース**: ブラウザベースの監視・制御システム

## 設計の背景

本アーキテクチャは、既存のstreaming overlayerの課題を解決するために設計されました：

- **密結合の解消**: 各コンポーネントが独立して開発・テスト可能
- **再利用性の向上**: 他のプロジェクトでも利用可能な汎用フレームワーク
- **保守性の向上**: 明確な責任分離により、個別の修正が容易
- **拡張性の確保**: 新機能の追加が既存コードに影響しない

この新しいアーキテクチャにより、テニス解析システムの更なる発展と、他の動画解析プロジェクトへの適用が可能となります。 