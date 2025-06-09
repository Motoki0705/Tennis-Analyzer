# streaming_annotator/video_utils.py

import queue
import threading
from pathlib import Path

import cv2
import numpy as np

class FrameLoader:
    """動画ファイルを別スレッドで読み込み、キューを介してフレームを提供するクラス。

    これにより、メインスレッドのI/O待ちを解消し、処理のボトルネックを軽減します。

    Attributes:
        cap (cv2.VideoCapture): OpenCVのビデオキャプチャオブジェクト。
        frame_queue (queue.Queue): 読み込んだフレーム (インデックス, 画像) を格納するキュー。
        running (bool): 読み込みスレッドの実行状態を制御するフラグ。
        read_thread (threading.Thread): フレーム読み込みを実行するスレッド。
    """
    def __init__(self, video_path: Path, max_queue_size: int = 128):
        """FrameLoaderを初期化します。

        Args:
            video_path (Path): 読み込む動画ファイルのパス。
            max_queue_size (int): フレームをバッファリングするキューの最大サイズ。
        """
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"動画ファイルを開けませんでした: {video_path}")
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)

    def _read_loop(self):
        """動画フレームを連続的に読み込み、キューに追加する内部ループ。"""
        frame_idx = 0
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                self.frame_queue.put((frame_idx, frame))
                frame_idx += 1
            else:
                time.sleep(0.01) # キューが満杯の場合は少し待つ
        self.frame_queue.put(None) # 終了を示すNoneを追加

    def start(self):
        """フレーム読み込みスレッドを開始します。"""
        self.read_thread.start()
        return self

    def read(self) -> Optional[Tuple[int, np.ndarray]]:
        """キューからフレームを1つ取得します。

        Returns:
            Optional[Tuple[int, np.ndarray]]: (フレームインデックス, フレーム画像)のタプル。
                                               動画の終端に達した場合はNoneを返します。
        """
        result = self.frame_queue.get()
        return result

    def get_properties(self) -> Dict[str, Union[int, float]]:
        """動画のプロパティ（解像度、FPS、総フレーム数）を取得します。

        Returns:
            Dict[str, Union[int, float]]: 動画のプロパティを含む辞書。
        """
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def release(self):
        """リソースを解放します。"""
        self.running = False
        self.read_thread.join()
        self.cap.release()