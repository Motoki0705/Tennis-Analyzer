# common/workers/base_worker.py

import queue
import threading
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, List

class BaseWorker(ABC):
    """パイプライン処理を行うワーカーの汎用的な基底クラス。

    Attributes:
        name (str): ワーカーの名前（デバッグ用）。
        predictor (Any): 推論を実行するモデル予測器。
        preprocess_queue (queue.Queue): 前処理タスクを保持するキュー。
        inference_queue (queue.Queue): 推論タスクを保持するキュー。
        postprocess_queue (queue.Queue): 後処理タスクを保持するキュー。
        postprocess_handler (Callable): 後処理の結果を処理するためのコールバック関数。
        running (bool): ワーカーが実行中かどうかを示すフラグ。
        debug (bool): デバッグメッセージを有効にするフラグ。
        threads (List[threading.Thread]): このワーカーが管理するスレッドのリスト。
    """
    def __init__(
        self,
        name: str,
        predictor: Any,
        preprocess_q: queue.Queue,
        inference_q: queue.Queue,
        postprocess_q: queue.Queue,
        postprocess_handler: Callable,
        debug: bool = False,
    ):
        """BaseWorkerのインスタンスを初期化します。

        Args:
            name (str): ワーカーの名前（例: "ball", "court"）。
            predictor (Any): 対応する予測器インスタンス。
            preprocess_q (queue.Queue): 前処理タスク用のキュー。
            inference_q (queue.Queue): 推論タスク用のキュー。
            postprocess_q (queue.Queue): 後処理タスク用のキュー。
            postprocess_handler (Callable): 後処理の結果を扱うコールバック関数。
            debug (bool): デバッグログを有効にするかどうか。
        """
        self.name = name
        self.predictor = predictor
        self.preprocess_queue = preprocess_q
        self.inference_queue = inference_q
        self.postprocess_queue = postprocess_q
        self.postprocess_handler = postprocess_handler
        self.running = False
        self.debug = debug
        self.threads: List[threading.Thread] = []

    def start(self):
        """ワーカーの全スレッドを開始します。"""
        self.running = True
        self.threads = [
            threading.Thread(target=self._loop, args=(self.preprocess_queue, self._process_preprocess_task), daemon=True),
            threading.Thread(target=self._loop, args=(self.inference_queue, self._process_inference_task), daemon=True),
            threading.Thread(target=self._loop, args=(self.postprocess_queue, self._process_postprocess_task), daemon=True),
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

    def _loop(self, task_queue: queue.Queue, process_func: Callable):
        """キューからタスクを取得して処理する汎用ループ。"""
        while self.running:
            try:
                task = task_queue.get(timeout=0.1)
                try:
                    process_func(task)
                except Exception as e:
                    print(f"[{self.name.upper()}] タスク処理中にエラーが発生 ({getattr(task, 'task_id', 'N/A')}): {e}")
                    traceback.print_exc()
                finally:
                    task_queue.task_done()
            except queue.Empty:
                continue

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