"""
高度拡張性を持つストリーミング処理パイプライン - Workers

このパッケージは、様々な処理を行うワーカーの実装を提供します。
新しいアーキテクチャに対応した実装。
"""

from .base_worker import BaseWorker
from .ball_worker import BallDetectionWorker, BallDetectionResult
from .court_worker import CourtDetectionWorker, CourtDetectionResult
from .event_worker import EventDetectionWorker, EventDetectionResult

# PoseWorkerは後で実装予定
# from .pose_worker import PoseDetectionWorker, PoseDetectionResult

__all__ = [
    # Base classes
    "BaseWorker",
    
    # Workers
    "BallDetectionWorker",
    "CourtDetectionWorker", 
    "EventDetectionWorker",
    # "PoseDetectionWorker",  # 後で追加
    
    # Result classes
    "BallDetectionResult",
    "CourtDetectionResult",
    "EventDetectionResult",
    # "PoseDetectionResult",  # 後で追加
] 