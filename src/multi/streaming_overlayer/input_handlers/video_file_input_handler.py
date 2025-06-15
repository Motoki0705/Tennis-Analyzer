"""
高度拡張性を持つストリーミング処理パイプライン - VideoFileInputHandler

動画ファイルからフレームを読み込むInputHandler実装。
"""

import cv2
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union

from ..core.interfaces import InputHandler, ItemId

logger = logging.getLogger(__name__)


class VideoFileInputHandler(InputHandler):
    """
    動画ファイルからフレームを読み込むInputHandler。
    
    cv2.VideoCaptureをラップし、フレームを順次提供します。
    """
    
    def __init__(self, video_path: Union[str, Path], frame_skip: int = 1):
        """
        Args:
            video_path: 動画ファイルのパス
            frame_skip: フレームスキップ間隔（1=全フレーム, 2=1フレームおき）
        """
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        self.cap = None
        self._properties = None
        
        # 動画ファイルの存在確認
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
    
    def __iter__(self) -> Iterator[Tuple[ItemId, Any]]:
        """
        フレームのイテレータを返します。
        
        Yields:
            Tuple[int, np.ndarray]: (フレームインデックス, フレーム画像)
        """
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            
            frame_idx = 0
            output_frame_idx = 0
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # フレームスキップの処理
                if frame_idx % self.frame_skip == 0:
                    yield output_frame_idx, frame
                    output_frame_idx += 1
                
                frame_idx += 1
        
        except Exception as e:
            logger.error(f"Error reading video {self.video_path}: {e}")
            raise
        
        finally:
            if self.cap:
                self.cap.release()
    
    def get_properties(self) -> Dict[str, Any]:
        """
        動画ファイルのプロパティを返します。
        
        Returns:
            Dict[str, Any]: 動画のメタ情報
        """
        if self._properties is None:
            self._properties = self._load_properties()
        
        return self._properties
    
    def _load_properties(self) -> Dict[str, Any]:
        """動画ファイルのプロパティを読み込みます。"""
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video for properties: {self.video_path}")
            
            properties = {
                "source_path": str(self.video_path),
                "source_type": "video_file",
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "frame_skip": self.frame_skip,
                "effective_total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.frame_skip
            }
            
            cap.release()
            return properties
        
        except Exception as e:
            logger.error(f"Error loading video properties: {e}")
            # フォールバック値を返す
            return {
                "source_path": str(self.video_path),
                "source_type": "video_file",
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "total_frames": 0,
                "frame_skip": self.frame_skip,
                "effective_total_frames": 0
            }
    
    def close(self) -> None:
        """リソースを解放します。"""
        if self.cap:
            self.cap.release() 