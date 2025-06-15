"""
高度拡張性を持つストリーミング処理パイプライン

このパッケージは、特定のアプリケーションから独立し、様々な入出力や
複雑な処理フローに対応可能な、汎用的かつ再利用可能なデータ処理フレームワークです。

設計原則:
- 関心の分離 (Separation of Concerns)
- 依存性の注入 (Dependency Injection)  
- イベント駆動 (Event-Driven)
- 宣言的な依存関係

主要コンポーネント:
- Core: インターフェース定義とパイプライン制御
- Managers: タスク管理と結果管理
- Workers: 個別処理の実行
- Input/Output Handlers: 入出力の抽象化
"""

# Core components
from .core import (
    PipelineRunner,
    InputHandler,
    OutputHandler,
    BaseWorker,
    TaskManagerInterface,
    ResultManagerInterface,
    ItemId,
    TopicName,
    TaskData,
    ResultData
)

# Managers
from .managers import TaskManager, ResultManager

# Input handlers
from .input_handlers import (
    VideoFileInputHandler,
    FrameListInputHandler,
    DirectoryInputHandler
)

# Output handlers  
from .output_handlers import (
    VideoOverlayOutputHandler,
    JsonFileOutputHandler,
    CallbackOutputHandler
)

# Workers
from .workers import (
    BaseWorker as WorkerBaseWorker,  # Avoid naming conflict
    BallDetectionWorker,
    CourtDetectionWorker,
    EventDetectionWorker,
    BallDetectionResult,
    CourtDetectionResult,
    EventDetectionResult
)

__all__ = [
    # Core interfaces and runner
    "PipelineRunner",
    "InputHandler",
    "OutputHandler", 
    "BaseWorker",
    "TaskManagerInterface",
    "ResultManagerInterface",
    
    # Type aliases
    "ItemId",
    "TopicName",
    "TaskData", 
    "ResultData",
    
    # Managers
    "TaskManager",
    "ResultManager",
    
    # Input handlers
    "VideoFileInputHandler",
    "FrameListInputHandler", 
    "DirectoryInputHandler",
    
    # Output handlers
    "VideoOverlayOutputHandler",
    "JsonFileOutputHandler",
    "CallbackOutputHandler",
    
    # Workers and results
    "BallDetectionWorker",
    "CourtDetectionWorker",
    "EventDetectionWorker", 
    "BallDetectionResult",
    "CourtDetectionResult",
    "EventDetectionResult",
]

# Version info
__version__ = "2.0.0"
__author__ = "Tennis Analyzer Team"
__description__ = "高度拡張性を持つストリーミング処理パイプライン" 