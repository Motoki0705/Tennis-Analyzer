"""
Event Detection API Module

このモジュールはテニスのイベント検知（バウンド・ショット）を行うAPIを提供します。
"""

from .event_predictor import EventPredictor, create_event_predictor

__all__ = [
    "EventPredictor",
    "create_event_predictor",
] 