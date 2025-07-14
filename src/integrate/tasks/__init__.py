"""
Task implementations for the flexible tennis analysis pipeline.
"""

from .court_task import CourtDetectionTask
from .ball_task import BallTrackingTask
from .ball_tracking_task import BallTrackingTask as AdvancedBallTrackingTask
from .player_task import PlayerDetectionTask
from .pose_task import PoseEstimationTask
from .shot_classification_task import ShotClassificationTask

__all__ = [
    'CourtDetectionTask',
    'BallTrackingTask',
    'AdvancedBallTrackingTask',
    'PlayerDetectionTask',
    'PoseEstimationTask',
    'ShotClassificationTask'
]