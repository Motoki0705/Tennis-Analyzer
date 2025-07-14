"""
Video I/O utilities for the flexible tennis analysis pipeline.
"""
import cv2
import logging
from typing import Iterator, Tuple, Optional, Any
import os

log = logging.getLogger(__name__)


class VideoReader:
    """
    Video reader with frame iteration support.
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self._current_frame = 0
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize video capture."""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Cache video properties
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        log.info(f"Video loaded: {self._width}x{self._height}, {self._frame_count} frames, {self._fps:.2f} fps")
    
    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        return self._frame_count
    
    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps
    
    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width
    
    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height
    
    @property
    def current_frame(self) -> int:
        """Current frame index."""
        return self._current_frame
    
    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """
        Read next frame.
        
        Returns:
            Tuple of (success, frame) where success is True if frame was read successfully
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self._current_frame += 1
        
        return ret, frame
    
    def seek(self, frame_index: int) -> bool:
        """
        Seek to a specific frame.
        
        Args:
            frame_index: Target frame index
            
        Returns:
            True if seek was successful
        """
        if self.cap is None:
            return False
        
        if frame_index < 0 or frame_index >= self._frame_count:
            return False
        
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if success:
            self._current_frame = frame_index
        
        return success
    
    def reset(self) -> bool:
        """Reset to beginning of video."""
        return self.seek(0)
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over all frames in the video."""
        self.reset()
        
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield frame
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return self.frame_count
    
    def close(self) -> None:
        """Close video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoWriter:
    """
    Video writer for output video generation.
    """
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 codec: str = 'mp4v', color_format: str = 'BGR'):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.codec = codec
        self.color_format = color_format
        self.writer = None
        self._frame_count = 0
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if it's not empty (not current directory)
            os.makedirs(output_dir, exist_ok=True)
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize video writer."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video writer for: {self.output_path}")
        
        log.info(f"Video writer initialized: {self.output_path} "
                f"({self.frame_size[0]}x{self.frame_size[1]}, {self.fps:.2f} fps)")
    
    def write(self, frame: Any) -> bool:
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if frame was written successfully
        """
        if self.writer is None:
            return False
        
        # Ensure frame has correct dimensions
        if frame.shape[:2][::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        # Convert color format if necessary
        if self.color_format == 'RGB' and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.writer.write(frame)
        self._frame_count += 1
        
        return True
    
    def write_batch(self, frames: list) -> int:
        """
        Write multiple frames.
        
        Args:
            frames: List of frames to write
            
        Returns:
            Number of frames successfully written
        """
        written_count = 0
        for frame in frames:
            if self.write(frame):
                written_count += 1
        
        return written_count
    
    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count
    
    def close(self) -> None:
        """Close video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            log.info(f"Video writer closed. Wrote {self._frame_count} frames to {self.output_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FrameBuffer:
    """
    Frame buffer for storing recent frames in memory.
    Useful for temporal tasks that need access to frame history.
    """
    
    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.frames = []
        self.frame_indices = []
        self._current_index = 0
    
    def add_frame(self, frame: Any, frame_index: int) -> None:
        """Add a frame to the buffer."""
        if len(self.frames) >= self.buffer_size:
            # Remove oldest frame
            self.frames.pop(0)
            self.frame_indices.pop(0)
        
        self.frames.append(frame)
        self.frame_indices.append(frame_index)
        self._current_index = frame_index
    
    def get_recent_frames(self, count: int) -> Tuple[list, list]:
        """
        Get the most recent N frames.
        
        Args:
            count: Number of recent frames to retrieve
            
        Returns:
            Tuple of (frames, frame_indices)
        """
        if count <= 0:
            return [], []
        
        start_idx = max(0, len(self.frames) - count)
        return (
            self.frames[start_idx:],
            self.frame_indices[start_idx:]
        )
    
    def get_frame_sequence(self, center_index: int, sequence_length: int) -> Tuple[list, list]:
        """
        Get a sequence of frames centered around a specific index.
        
        Args:
            center_index: Frame index to center the sequence around
            sequence_length: Total length of the sequence
            
        Returns:
            Tuple of (frames, frame_indices) or (None, None) if sequence cannot be built
        """
        half_length = sequence_length // 2
        
        # Find the position of center_index in our buffer
        try:
            center_pos = self.frame_indices.index(center_index)
        except ValueError:
            return None, None
        
        # Calculate start and end positions
        start_pos = max(0, center_pos - half_length)
        end_pos = min(len(self.frames), center_pos + half_length + 1)
        
        # Check if we have enough frames
        if end_pos - start_pos < sequence_length:
            return None, None
        
        return (
            self.frames[start_pos:end_pos],
            self.frame_indices[start_pos:end_pos]
        )
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        self.frames.clear()
        self.frame_indices.clear()
        self._current_index = 0
    
    @property
    def current_frame_count(self) -> int:
        """Number of frames currently in buffer."""
        return len(self.frames)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.frames) >= self.buffer_size
    
    def __len__(self) -> int:
        return len(self.frames)