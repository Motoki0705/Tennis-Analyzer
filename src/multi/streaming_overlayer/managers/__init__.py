"""
高度拡張性を持つストリーミング処理パイプライン - Managers

このパッケージは、パイプラインの管理コンポーネントを提供します。
"""

from .task_manager import TaskManager
from .result_manager import ResultManager

__all__ = [
    "TaskManager",
    "ResultManager",
] 