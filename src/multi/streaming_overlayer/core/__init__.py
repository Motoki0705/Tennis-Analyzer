"""
高度拡張性を持つストリーミング処理パイプライン - Core

このパッケージは、パイプラインのコアコンポーネントを提供します。
"""

# Core interfaces
from .interfaces import (
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

# Pipeline runner
from .pipeline_runner import PipelineRunner

__all__ = [
    # Interfaces
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
    
    # Pipeline runner
    "PipelineRunner",
] 