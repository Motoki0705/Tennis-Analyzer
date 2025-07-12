import argparse
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import logging
import threading
import queue
import csv
import time
from typing import List, Dict, Any, Tuple

# モジュールとユーティリティをインポート
from .pipeline_modules import CourtPreprocessor, CourtDetector, CourtPostprocessor
from .drawing_utils import draw_keypoints_on_frame, draw_court_skeleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class MultithreadedCourtDetector:
    """
    マルチスレッドパイプラインでテニスコートのキーポイントを検出するクラス。
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._initialize_device()
        self._initialize_pipeline_modules()
        self.video_writer = None
        self.video_properties: Dict[str, Any] = {}
        self.preprocess_queue: queue.Queue = queue.Queue(maxsize=args.batch_size * 2)
        self.inference_queue: queue.Queue = queue.Queue(maxsize=args.batch_size * 2)
        self.all_keypoint_results: List[Dict[str, np.ndarray]] = []
        self.is_running = threading.Event()
        self.timings: Dict[str, List[float]] = {k: [] for k in ['io_preprocess', 'inference', 'postprocess_write']}

    def _initialize_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.device == "cuda" else "cpu")
        log.info(f"Using device: {self.device}")

    def _initialize_pipeline_modules(self):
        log.info("Initializing pipeline modules...")
        # コマンドライン引数から入力サイズを取得
        input_size = tuple(self.args.input_size)
        self.preprocessor = CourtPreprocessor(input_size=input_size)
        self.detector = CourtDetector(self.args.checkpoint_path, self.device)
        self.postprocessor = CourtPostprocessor(
            multi_channel=self.args.multi_channel,
        )
        log.info("Pipeline modules initialized.")

    def _initialize_video_io(self):
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.args.video}")
        self.video_properties = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        log.info(f"Video: {self.video_properties['width']}x{self.video_properties['height']}, "
                 f"{self.video_properties['fps']:.2f}fps, {self.video_properties['total_frames']} frames")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.args.output, fourcc, self.video_properties['fps'],
                                            (self.video_properties['width'], self.video_properties['height']))

    def _worker_io_preprocess(self):
        log.info("Worker 1 (I/O & Preprocess) started.")
        cap = cv2.VideoCapture(self.args.video)
        frames_batch_rgb, frames_batch_bgr = [], []
        for frame_idx in range(self.video_properties['total_frames']):
            if not self.is_running.is_set(): break
            ret, frame = cap.read()
            if not ret: break
            
            frames_batch_bgr.append(frame)
            frames_batch_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(frames_batch_rgb) == self.args.batch_size or (frame_idx == self.video_properties['total_frames'] - 1):
                start_time = time.perf_counter()
                batch_tensor, batch_meta = self.preprocessor.process_batch(frames_batch_rgb)
                self.preprocess_queue.put((batch_tensor, batch_meta, frames_batch_bgr.copy()))
                self.timings['io_preprocess'].append(time.perf_counter() - start_time)
                frames_batch_rgb.clear()
                frames_batch_bgr.clear()
        
        self.preprocess_queue.put(None)
        cap.release()
        log.info("Worker 1 (I/O & Preprocess) finished.")

    def _worker_inference(self):
        log.info("Worker 2 (Inference) started.")
        while self.is_running.is_set():
            data = self.preprocess_queue.get()
            if data is None:
                self.inference_queue.put(None)
                break
            batch_tensor, batch_meta, original_frames_batch = data
            start_time = time.perf_counter()
            heatmap_preds = self.detector.predict(batch_tensor)
            if self.device.type == 'cuda': torch.cuda.synchronize()
            self.timings['inference'].append(time.perf_counter() - start_time)
            self.inference_queue.put((heatmap_preds, batch_meta, original_frames_batch))
        log.info("Worker 2 (Inference) finished.")

    def _worker_postprocess_write(self):
        log.info("Worker 3 (Postprocess & Write) started.")
        pbar = tqdm(total=self.video_properties['total_frames'], desc="Processing frames")
        while self.is_running.is_set():
            data = self.inference_queue.get()
            if data is None: break
            heatmap_preds, batch_meta, original_frames_batch = data
            start_time = time.perf_counter()
            
            batch_results = self.postprocessor.process_batch(heatmap_preds, batch_meta, original_frames_batch)
            
            for i, result in enumerate(batch_results):
                self.all_keypoint_results.append(result)
                output_frame = original_frames_batch[i]
                
                # --- 描画処理 ---
                draw_keypoints_on_frame(output_frame, result['keypoints'], result['scores'], self.args.score_threshold)
                draw_court_skeleton(output_frame, result['keypoints'], result['scores'], self.args.score_threshold)
                
                self.video_writer.write(output_frame)
                pbar.update(1)

            self.timings['postprocess_write'].append(time.perf_counter() - start_time)
        pbar.close()
        log.info("Worker 3 (Postprocess & Write) finished.")

    def _save_results_as_csv(self):
        # (実装は変更なし)
        ...

    def _report_timings(self):
        # (実装は変更なし)
        ...

    def run(self):
        self._initialize_video_io()
        self.is_running.set()
        threads = [
            threading.Thread(target=self._worker_io_preprocess),
            threading.Thread(target=self._worker_inference),
            threading.Thread(target=self._worker_postprocess_write)
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        log.info("All workers have finished.")
        # self._save_results_as_csv()
        self._report_timings()
        if self.video_writer: self.video_writer.release()
        log.info(f"Output video saved to: {self.args.output}")

def main():
    parser = argparse.ArgumentParser(description="Multithreaded Court Keypoint Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the trained model checkpoint (.ckpt)")
    parser.add_argument("--output", default="demo_output_court.mp4", help="Output video file")
    # parser.add_argument("--results_csv", default="court_keypoints.csv", help="Output CSV file for keypoint data")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Minimum score to visualize a keypoint")
    
    # --- 追加された引数 ---
    parser.add_argument("--input_size", type=int, nargs=2, default=[360, 640], help="Model input size (height width)")
    parser.add_argument('--multi_channel', action='store_true', help='Enable milti channel heatmap postprocess')
    
    args = parser.parse_args()

    try:
        detector_pipeline = MultithreadedCourtDetector(args)
        detector_pipeline.run()
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()