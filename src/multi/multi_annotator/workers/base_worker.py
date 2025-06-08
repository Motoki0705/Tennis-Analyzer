# multi_flow_annotator/workers/base_worker.py

import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseWorker(ABC):
    """前処理、推論、後処理のパイプラインを管理するワーカーの基底クラス。

    Attributes:
        predictor (Any): 推論を実行するモデル予測器。
        preprocess_queue (queue.Queue): 前処理タスクを保持するキュー。
        inference_queue (queue.Queue): 推論タスクを保持するキュー。
        postprocess_queue (queue.Queue): 後処理タスクを保持するキュー。
        coco_manager (Any): COCOアノテーションを管理するマネージャー。
        vis_thresh (float): 可視性の閾値。
        processed_counter (int): このワーカーによって処理されたアイテムの数。
        running (bool): ワーカーが実行中かどうかを示すフラグ。
        debug (bool): デバッグメッセージを有効にするフラグ。
        threads (List[threading.Thread]): このワーカーが管理するスレッドのリスト。
    """

    def __init__(
        self,
        name: str,
        predictor: Any,
        preprocess_queue: queue.Queue,
        inference_queue: queue.Queue,
        postprocess_queue: queue.Queue,
        coco_manager: Any,
        vis_thresh: float,
        debug: bool = False,
    ):
        """BaseWorkerのインスタンスを初期化します。

        Args:
            name (str): ワーカーの名前（例: "ball", "court"）。
            predictor (Any): 対応する予測器インスタンス。
            preprocess_queue (queue.Queue): 前処理タスク用のキュー。
            inference_queue (queue.Queue): 推論タスク用のキュー。
            postprocess_queue (queue.Queue): 後処理タスク用のキュー。
            coco_manager (Any): CocoManagerのインスタンス。
            vis_thresh (float): 可視性を判断するための信頼度閾値。
            debug (bool): デバッグログを有効にするかどうか。
        """
        self.name = name
        self.predictor = predictor
        self.preprocess_queue = preprocess_queue
        self.inference_queue = inference_queue
        self.postprocess_queue = postprocess_queue
        self.coco_manager = coco_manager
        self.vis_thresh = vis_thresh
        self.processed_counter = 0
        self.running = False
        self.debug = debug
        self.threads: List[threading.Thread] = []

    def start(self):
        """ワーカーの全スレッドを開始します。"""
        self.running = True
        self.threads = [
            threading.Thread(target=self._preprocess_loop, daemon=True),
            threading.Thread(target=self._inference_loop, daemon=True),
            threading.Thread(target=self._postprocess_loop, daemon=True),
        ]
        for t in self.threads:
            t.start()
        if self.debug:
            print(f"[{self.name.upper()}] ワーカーを開始しました。")

    def stop(self):
        """ワーカーの全スレッドを停止します。"""
        self.running = False
        for t in self.threads:
            t.join(timeout=2.0)
        if self.debug:
            print(f"[{self.name.upper()}] ワーカーを停止しました。")

    def _loop_body(self, task_queue: queue.Queue, process_func):
        """キューからタスクを取得して処理する汎用ループ。

        Args:
            task_queue (queue.Queue): タスクを取得するキュー。
            process_func (Callable): タスクを処理する関数。
        """
        while self.running:
            try:
                task = task_queue.get(timeout=0.1)
                try:
                    process_func(task)
                except Exception as e:
                    print(f"[{self.name.upper()}] タスク処理中にエラーが発生 ({task.task_id}): {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    task_queue.task_done()
            except queue.Empty:
                continue

    def _preprocess_loop(self):
        self._loop_body(self.preprocess_queue, self._process_preprocess_task)

    def _inference_loop(self):
        self._loop_body(self.inference_queue, self._process_inference_task)

    def _postprocess_loop(self):
        self.running_post = True
        self._loop_body(self.postprocess_queue, self._process_postprocess_task)

    @abstractmethod
    def _process_preprocess_task(self, task):
        """前処理タスクを実行します。サブクラスで実装する必要があります。"""
        pass

    @abstractmethod
    def _process_inference_task(self, task):
        """推論タスクを実行します。サブクラスで実装する必要があります。"""
        pass

    @abstractmethod
    def _process_postprocess_task(self, task):
        """後処理タスクを実行します。サブクラスで実装する必要があります。"""
        pass