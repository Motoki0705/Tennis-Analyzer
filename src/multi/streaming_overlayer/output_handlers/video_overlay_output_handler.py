"""
高度拡張性を持つストリーミング処理パイプライン - VideoOverlayOutputHandler

動画にオーバーレイを追加して出力するOutputHandler実装。
"""

import cv2
import logging
from pathlib import Path
from typing import Any, Dict, Union

from ..core.interfaces import OutputHandler, ItemId

logger = logging.getLogger(__name__)


class VideoOverlayOutputHandler(OutputHandler):
    """
    動画にオーバーレイを追加して出力するOutputHandler。
    
    処理結果を元の動画フレームに重畳し、新しい動画ファイルとして保存します。
    """
    
    def __init__(self, output_path: Union[str, Path], 
                 codec: str = 'mp4v', 
                 fps: float = 30.0,
                 frame_size: tuple = (1920, 1080)):
        """
        Args:
            output_path: 出力動画ファイルのパス
            codec: 動画エンコーダー（FOURCC）
            fps: フレームレート
            frame_size: フレームサイズ (width, height)
        """
        self.output_path = Path(output_path)
        self.codec = codec
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self._frame_count = 0
        
        # 出力ディレクトリの作成
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_output(self, properties: Dict[str, Any]) -> None:
        """
        出力を開始します。
        
        Args:
            properties: 入力のプロパティ情報
        """
        try:
            # 入力プロパティから動画設定を更新（可能な場合）
            if 'width' in properties and 'height' in properties:
                self.frame_size = (int(properties['width']), int(properties['height']))
            
            if 'fps' in properties:
                self.fps = float(properties['fps'])
            
            # VideoWriterを初期化
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size
            )
            
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.output_path}")
            
            logger.info(f"Started video output: {self.output_path} "
                       f"({self.frame_size[0]}x{self.frame_size[1]} @ {self.fps}fps)")
        
        except Exception as e:
            logger.error(f"Error starting video output: {e}")
            raise
    
    def handle_result(self, item_id: ItemId, 
                     original_item: Any, 
                     results: Dict[str, Any]) -> None:
        """
        処理結果を動画フレームに適用して出力します。
        
        Args:
            item_id: アイテムID
            original_item: 元のフレーム画像
            results: 各ワーカーからの処理結果
        """
        try:
            if self.writer is None:
                logger.error("VideoWriter not initialized")
                return
            
            # 元フレームをコピー
            output_frame = original_item.copy()
            
            # フレームサイズの調整
            if output_frame.shape[:2][::-1] != self.frame_size:
                output_frame = cv2.resize(output_frame, self.frame_size)
            
            # 結果をオーバーレイとして描画
            output_frame = self._draw_overlays(output_frame, results)
            
            # フレームを出力
            self.writer.write(output_frame)
            self._frame_count += 1
            
            if self._frame_count % 100 == 0:
                logger.debug(f"Wrote {self._frame_count} frames to {self.output_path}")
        
        except Exception as e:
            logger.error(f"Error handling result for frame {item_id}: {e}")
    
    def _draw_overlays(self, frame: Any, results: Dict[str, Any]) -> Any:
        """
        フレームに処理結果のオーバーレイを描画します。
        
        Args:
            frame: 描画対象のフレーム
            results: 処理結果の辞書
        
        Returns:
            オーバーレイが描画されたフレーム
        """
        try:
            # ボール検出結果の描画
            if 'ball_detection' in results:
                frame = self._draw_ball_detection(frame, results['ball_detection'])
            
            # コート検出結果の描画
            if 'court_detection' in results:
                frame = self._draw_court_detection(frame, results['court_detection'])
            
            # ポーズ検出結果の描画
            if 'pose_detection' in results:
                frame = self._draw_pose_detection(frame, results['pose_detection'])
            
            # イベント検出結果の描画
            if 'event_detection' in results:
                frame = self._draw_event_detection(frame, results['event_detection'])
            
            return frame
        
        except Exception as e:
            logger.error(f"Error drawing overlays: {e}")
            return frame
    
    def _draw_ball_detection(self, frame: Any, ball_result: Any) -> Any:
        """ボール検出結果を描画します。"""
        try:
            if hasattr(ball_result, 'position') and ball_result.position is not None:
                x, y = int(ball_result.position[0]), int(ball_result.position[1])
                cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (x-20, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            logger.debug(f"Error drawing ball detection: {e}")
        
        return frame
    
    def _draw_court_detection(self, frame: Any, court_result: Any) -> Any:
        """コート検出結果を描画します。"""
        try:
            if hasattr(court_result, 'keypoints') and court_result.keypoints is not None:
                # コートのキーポイントを描画
                for point in court_result.keypoints:
                    if point is not None:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        except Exception as e:
            logger.debug(f"Error drawing court detection: {e}")
        
        return frame
    
    def _draw_pose_detection(self, frame: Any, pose_result: Any) -> Any:
        """ポーズ検出結果を描画します。"""
        try:
            if hasattr(pose_result, 'poses') and pose_result.poses:
                for pose in pose_result.poses:
                    if hasattr(pose, 'keypoints') and pose.keypoints is not None:
                        # ポーズのキーポイントを描画
                        for point in pose.keypoints:
                            if point is not None and len(point) >= 2:
                                x, y = int(point[0]), int(point[1])
                                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        except Exception as e:
            logger.debug(f"Error drawing pose detection: {e}")
        
        return frame
    
    def _draw_event_detection(self, frame: Any, event_result: Any) -> Any:
        """イベント検出結果を描画します。"""
        try:
            if hasattr(event_result, 'event_type') and event_result.event_type:
                # イベント情報をテキストで表示
                text = f"Event: {event_result.event_type}"
                cv2.putText(frame, text, (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        except Exception as e:
            logger.debug(f"Error drawing event detection: {e}")
        
        return frame
    
    def finish_output(self) -> None:
        """出力を終了します。"""
        try:
            if self.writer:
                self.writer.release()
                self.writer = None
                logger.info(f"Finished video output: {self.output_path} "
                           f"({self._frame_count} frames)")
        
        except Exception as e:
            logger.error(f"Error finishing video output: {e}")
    
    def close(self) -> None:
        """リソースを解放します。"""
        self.finish_output() 