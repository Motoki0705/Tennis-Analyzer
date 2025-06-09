# streaming_annotator/workers/base_worker.py
# (前回の回答とほぼ同じなので、要点のみ記載)
import queue
import threading
from abc import ABC, abstractmethod

class BaseWorker(ABC):
    def __init__(self, name, predictor, preprocess_q, inference_q, postprocess_q, results_q, debug=False):
        self.name = name
        self.predictor = predictor
        self.preprocess_queue = preprocess_q
        self.inference_queue = inference_q
        self.postprocess_queue = postprocess_q
        self.results_queue = results_q  # 結果を書き込む共有キュー
        self.running = False
        self.debug = debug
        self.threads = []

    # start, stop, _loop_body, _*_loop メソッドは前回と同様

    @abstractmethod
    def _process_preprocess_task(self, task): pass
    @abstractmethod
    def _process_inference_task(self, task): pass
    @abstractmethod
    def _process_postprocess_task(self, task): pass