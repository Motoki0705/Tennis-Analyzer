"""
高度拡張性を持つストリーミング処理パイプライン - Output Handlers

このパッケージは、様々な出力形式に対応するOutputHandlerの実装を提供します。
"""

from .video_overlay_output_handler import VideoOverlayOutputHandler
from .json_file_output_handler import JsonFileOutputHandler
from .callback_output_handler import CallbackOutputHandler

__all__ = [
    "VideoOverlayOutputHandler",
    "JsonFileOutputHandler",
    "CallbackOutputHandler",
] 