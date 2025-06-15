"""
高度拡張性を持つストリーミング処理パイプライン - FrameListInputHandler

フレームのリストからフレームを読み込むInputHandler実装。
"""

import logging
from typing import Any, Dict, Iterator, List, Tuple

from ..core.interfaces import InputHandler, ItemId

logger = logging.getLogger(__name__)


class FrameListInputHandler(InputHandler):
    """
    フレームのリストからフレームを読み込むInputHandler。
    
    メモリ上のフレームリストを順次提供します。
    """
    
    def __init__(self, frames: List[Any], metadata: Dict[str, Any] = None):
        """
        Args:
            frames: フレームデータのリスト
            metadata: フレームに関するメタデータ
        """
        self.frames = frames
        self.metadata = metadata or {}
        
        if not frames:
            logger.warning("Empty frames list provided")
    
    def __iter__(self) -> Iterator[Tuple[ItemId, Any]]:
        """
        フレームのイテレータを返します。
        
        Yields:
            Tuple[int, Any]: (フレームインデックス, フレームデータ)
        """
        try:
            for idx, frame in enumerate(self.frames):
                yield idx, frame
        
        except Exception as e:
            logger.error(f"Error iterating frames: {e}")
            raise
    
    def get_properties(self) -> Dict[str, Any]:
        """
        フレームリストのプロパティを返します。
        
        Returns:
            Dict[str, Any]: フレームリストのメタ情報
        """
        properties = {
            "source_type": "frame_list",
            "total_frames": len(self.frames),
            "effective_total_frames": len(self.frames),
        }
        
        # メタデータが提供されている場合は追加
        properties.update(self.metadata)
        
        # 最初のフレームから形状情報を推定（可能な場合）
        if self.frames:
            first_frame = self.frames[0]
            if hasattr(first_frame, 'shape'):
                if len(first_frame.shape) >= 2:
                    properties.update({
                        "height": first_frame.shape[0],
                        "width": first_frame.shape[1]
                    })
                if len(first_frame.shape) >= 3:
                    properties["channels"] = first_frame.shape[2]
        
        return properties
    
    def close(self) -> None:
        """リソースを解放します（この実装では何もしません）。"""
        pass 