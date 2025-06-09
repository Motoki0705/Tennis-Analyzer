# streaming_annotator/workers/base_worker.py
# (前回の回答とほぼ同じなので、要点のみ記載)
import queue
import threading
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseWorker(ABC):
    """
    ストリーミング処理用のベースワーカークラス。
    
    前処理、推論、後処理の3段階のパイプラインを並列実行します。
    """
    
    def __init__(self, name, predictor, preprocess_q, inference_q, postprocess_q, results_q, debug=False):
        """
        Args:
            name: ワーカー名
            predictor: 予測器インスタンス
            preprocess_q: 前処理キュー
            inference_q: 推論キュー
            postprocess_q: 後処理キュー
            results_q: 結果出力キュー
            debug: デバッグモード
        """
        self.name = name
        self.predictor = predictor
        self.preprocess_queue = preprocess_q
        self.inference_queue = inference_q
        self.postprocess_queue = postprocess_q
        self.results_queue = results_q
        self.running = False
        self.debug = debug
        self.threads = []

    def start(self):
        """ワーカースレッドを開始します。"""
        if self.running:
            logger.warning(f"{self.name} worker is already running")
            return
            
        self.running = True
        
        # 3つの処理段階のスレッドを作成
        threads = [
            threading.Thread(target=self._preprocess_loop, name=f"{self.name}_preprocess"),
            threading.Thread(target=self._inference_loop, name=f"{self.name}_inference"),
            threading.Thread(target=self._postprocess_loop, name=f"{self.name}_postprocess"),
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            
        logger.info(f"Started {self.name} worker with {len(self.threads)} threads")

    def stop(self):
        """ワーカースレッドを停止します。"""
        if not self.running:
            return
            
        self.running = False
        
        # スレッドの終了を待機
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        self.threads.clear()
        logger.info(f"Stopped {self.name} worker")

    def _preprocess_loop(self):
        """前処理ループ"""
        while self.running:
            try:
                task = self.preprocess_queue.get(timeout=0.1)
                self._process_preprocess_task(task)
                self.preprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} preprocess error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _inference_loop(self):
        """推論ループ"""
        while self.running:
            try:
                task = self.inference_queue.get(timeout=0.1)
                self._process_inference_task(task)
                self.inference_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} inference error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _postprocess_loop(self):
        """後処理ループ"""
        while self.running:
            try:
                task = self.postprocess_queue.get(timeout=0.1)
                self._process_postprocess_task(task)
                self.postprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.name} postprocess error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    @abstractmethod
    def _process_preprocess_task(self, task):
        """前処理タスクを処理します。継承クラスで実装してください。"""
        pass

    @abstractmethod
    def _process_inference_task(self, task):
        """推論タスクを処理します。継承クラスで実装してください。"""
        pass

    @abstractmethod
    def _process_postprocess_task(self, task):
        """後処理タスクを処理します。継承クラスで実装してください。"""
        pass