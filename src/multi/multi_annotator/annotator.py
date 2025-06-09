# multi_flow_annotator/annotator.py

import queue
import time
from pathlib import Path
from typing import List, Union, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2

from .definitions import PreprocessTask
from .coco_utils import CocoManager
from .file_utils import validate_paths, collect_and_group_images, load_frame, extract_ids_from_path
from .workers.ball_worker import BallWorker
from .workers.court_worker import CourtWorker
from .workers.pose_worker import PoseWorker

class MultiFlowAnnotator:
    """マルチフローアーキテクチャで画像アノテーションを効率的に行うクラス。

    前処理、推論、後処理のパイプラインを並列化し、GPU使用率の最大化を目指します。
    メインの責務は、データフローの管理、ワーカーの制御、タスクの投入です。
    """

    def __init__(
        self,
        ball_predictor, court_predictor, pose_predictor,
        batch_sizes: dict = None,
        vis_thresholds: dict = None,
        preprocess_workers: int = 4,
        max_queue_size: int = 8,
        debug: bool = False,
    ):
        """MultiFlowAnnotatorのインスタンスを初期化します。

        Args:
            ball_predictor: ボール検出モデルの予測器。
            court_predictor: コート検出モデルの予測器。
            pose_predictor: ポーズ推定モデルの予測器。
            batch_sizes (dict): 各モデルのバッチサイズ設定。
            vis_thresholds (dict): 各アノテーションの可視性閾値。
            preprocess_workers (int): フレーム読み込みなど前処理用ワーカー数。
            max_queue_size (int): 各処理キューの最大サイズ。
            debug (bool): デバッグログを有効にするか。
        """
        self.batch_sizes = batch_sizes or {"ball": 8, "court": 8, "pose": 8}
        self.vis_thresholds = vis_thresholds or {"ball": 0.5, "court": 0.6, "pose": 0.5}
        self.max_queue_size = max_queue_size
        self.debug = debug
        
        self.coco_manager = CocoManager()
        self.image_id_map: Dict[Path, int] = {}
        self.grouped_entries: Dict[Tuple[int, int], List[Tuple[int, Path]]] = {}

        # 処理キューの初期化
        self._initialize_queues()

        # ワーカーの初期化
        self.workers = {
            "ball": BallWorker(
                ball_predictor, self.queues["ball"]["preprocess"], self.queues["ball"]["inference"],
                self.queues["ball"]["postprocess"], self.coco_manager, self.vis_thresholds["ball"], self.debug
            ),
            "court": CourtWorker(
                court_predictor, self.queues["court"]["preprocess"], self.queues["court"]["inference"],
                self.queues["court"]["postprocess"], self.coco_manager, self.vis_thresholds["court"], self.debug
            ),
            "pose": PoseWorker(
                pose_predictor, self.queues["pose"]["preprocess"], self.queues["pose"]["inference"],
                self.queues["pose"]["postprocess"], self.coco_manager, self.vis_thresholds["pose"], self.debug
            ),
        }

    def _initialize_queues(self):
        """処理パイプライン用のキューを初期化します。"""
        self.queues = {}
        for name in ["ball", "court", "pose"]:
            self.queues[name] = {
                "preprocess": queue.Queue(maxsize=self.max_queue_size),
                "inference": queue.Queue(maxsize=self.max_queue_size),
                "postprocess": queue.Queue(maxsize=self.max_queue_size),
            }

    def _start_workers(self):
        """全てのワーカーのスレッドを開始します。"""
        for worker in self.workers.values():
            worker.start()
        print("✅ 全てのワーカースレッドを開始しました。")

    def _stop_workers(self):
        """全てのワーカーのスレッドを停止します。"""
        print("\n⏳ ワーカースレッドの停止処理を開始します...")
        for worker in self.workers.values():
            worker.stop()
        print("✅ 全てのワーカースレッドを停止しました。")

    def _prepare_image_metadata(
    self, input_dir: Path, grouped_files: Dict[Tuple[int, int], List[Path]]
) -> int:
        """【高速化版】画像メタデータを並列処理で準備し、COCOエントリを作成します。

        画像の読み込みとメタデータ生成をスレッドプールで並列化し、
        処理時間を短縮します。画像の順序はクリップごと、フレーム番号順に維持されます。

        Args:
            input_dir (Path): 入力ディレクトリのパス。
            grouped_files (Dict[Tuple[int, int], List[Path]]): グループ化された画像ファイル。

        Returns:
            int: 処理対象の総画像数。
        """
        print("🖼️ 画像メタデータの生成を開始 (高速化バージョン)")

        # 1. 処理順序を確定させた全画像パスのリストを作成
        sorted_group_keys = sorted(grouped_files.keys())
        ordered_paths = [path for key in sorted_group_keys for path in grouped_files[key]]
        total_images = len(ordered_paths)
        if total_images == 0:
            return 0

        # 2. 結果を格納するリストを事前に確保
        self.coco_manager.coco_output['images'] = [None] * total_images
        # 共有リソースを保護するためのロック
        map_lock = threading.Lock()

        def _process_single_image(path: Path, index: int):
            """単一の画像を処理してCOCOエントリを生成するワーカー関数。"""
            try:
                # 画像を読み込み、高さと幅を取得
                img = cv2.imread(str(path))
                if img is None:
                    print(f"警告: 画像読み込み失敗: {path}")
                    return None
                height, width = img.shape[:2]

                # メタデータを抽出
                game_id, clip_id = extract_ids_from_path(path)
                image_id = index + 1  # 1-based index

                # COCO画像エントリを作成
                rel_path = str(path.relative_to(input_dir))
                image_entry = {
                    "id": image_id, "file_name": path.name,
                    "original_path": rel_path, "height": height, "width": width,
                    "license": 1, "game_id": game_id, "clip_id": clip_id,
                }
                return index, image_entry, path, image_id
            except Exception as e:
                print(f"メタデータ生成エラー ({path}): {e}")
                return None

        # 3. ThreadPoolExecutorで並列処理を実行
        with ThreadPoolExecutor(max_workers=self.preprocess_workers) as executor:
            # タスクをサブミット
            futures = [
                executor.submit(_process_single_image, path, i)
                for i, path in enumerate(ordered_paths)
            ]

            with tqdm(total=total_images, desc="メタデータ並列生成中") as pbar:
                for future in futures:
                    result = future.result()
                    if result:
                        index, image_entry, path, image_id = result
                        # 4. 事前に確保したリストの正しい位置に結果を格納
                        self.coco_manager.coco_output['images'][index] = image_entry
                        # 5. 共有マップをロックして更新
                        with map_lock:
                            self.image_id_map[path] = image_id
                    pbar.update(1)

        # Noneが残っている場合（エラーで処理できなかった画像）は除去
        self.coco_manager.coco_output['images'] = [
            entry for entry in self.coco_manager.coco_output['images'] if entry is not None
        ]
        # 次のIDが正しくなるようにカウンターを更新
        self.coco_manager.image_id_counter = len(self.coco_manager.coco_output['images']) + 1

        # 6. 最後にgrouped_entriesを構築
        print("メタデータに基づき、処理グループを再構築中...")
        for group_key in sorted_group_keys:
            self.grouped_entries[group_key] = []
            for path in grouped_files[group_key]:
                if path in self.image_id_map:
                    img_id = self.image_id_map[path]
                    self.grouped_entries[group_key].append((img_id, path))

        final_image_count = len(self.coco_manager.coco_output['images'])
        print(f"✅ メタデータ生成完了。有効な画像数: {final_image_count} / {total_images}")

        return final_image_count
    
    def _preload_clip(self, id_path_pairs: List[Tuple[int, Path]]):
        """クリップ内の全フレームを並列で先読みします。"""
        frames, ids, paths = [], [], []
        with ThreadPoolExecutor() as executor:
            future_to_id_path = {executor.submit(load_frame, path): (img_id, path) for img_id, path in id_path_pairs}
            for future in future_to_id_path:
                frame = future.result()
                if frame is not None:
                    img_id, path = future_to_id_path[future]
                    frames.append(frame)
                    ids.append(img_id)
                    paths.append(path)
        return frames, ids, paths
    
    def _submit_tasks(self, frames: List, ids: List, paths: List, pbar: tqdm):
        """読み込んだフレームをバッチ化し、各前処理キューにタスクを投入します。"""
        for name, worker in self.workers.items():
            batch_size = self.batch_sizes[name]
            for i in range(0, len(frames), batch_size):
                batch_end = i + batch_size
                batch_frames = frames[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_paths = paths[i:batch_end]
                
                meta_data = list(zip(batch_ids, batch_paths))
                task_id = f"{name}_{ids[i]}_{ids[min(batch_end, len(ids)-1)]}"
                task = PreprocessTask(task_id, batch_frames, meta_data)
                self.queues[name]["preprocess"].put(task)
        
        pbar.update(len(frames))

    def _wait_for_completion(self):
        """全てのキューが空になるまで待機します。"""
        all_queues = [q for spec in self.queues.values() for q in spec.values()]
        while any(not q.empty() for q in all_queues):
            time.sleep(1)
            if self.debug:
                for name, spec in self.queues.items():
                    print(f"[{name.upper()}] Queues "
                          f"Pre: {spec['preprocess'].qsize()}, "
                          f"Inf: {spec['inference'].qsize()}, "
                          f"Post: {spec['postprocess'].qsize()}")
        
        # task_done()が呼ばれるのを待つ
        for spec in self.queues.values():
            for q in spec.values():
                q.join()

    def run(self, input_dir: Union[str, Path], output_json: Union[str, Path],
            image_extensions: List[str] = None):
        """アノテーション処理のメインフローを実行します。

        Args:
            input_dir (Union[str, Path]): 入力画像ディレクトリ。
            output_json (Union[str, Path]): 出力COCO-JSONファイルのパス。
            image_extensions (List[str], optional): 処理対象の画像拡張子。
        """
        input_dir, output_json = Path(input_dir), Path(output_json)
        image_extensions = image_extensions or ['.jpg', '.jpeg', '.png']
        
        try:
            validate_paths(input_dir, output_json)
            grouped_files = collect_and_group_images(input_dir, image_extensions)
            if not grouped_files:
                raise ValueError("指定ディレクトリに画像が見つかりません。")

            total_images = self._prepare_image_metadata(input_dir, grouped_files)
            
            self._start_workers()

            with tqdm(total=total_images, desc="アノテーション処理中") as pbar:
                sorted_group_keys = sorted(grouped_files.keys())
                for idx, group_key in enumerate(sorted_group_keys):
                    game_id, clip_id = group_key
                    print(f"\n▶️ Clip処理開始: Game {game_id}, Clip {clip_id} ({idx+1}/{len(sorted_group_keys)})")
                    
                    # ボールワーカーの状態をリセット
                    self.workers["ball"].reset_state()
                    
                    id_path_pairs = self.grouped_entries[group_key]
                    frames, ids, paths = self._preload_clip(id_path_pairs)
                    
                    if not frames:
                        print(f"警告: Clip内に有効なフレームがありません: {group_key}")
                        pbar.update(len(id_path_pairs))
                        continue
                        
                    self._submit_tasks(frames, ids, paths, pbar)
            
            print("\n⏳ 全てのタスク投入が完了。処理の完了を待っています...")
            self._wait_for_completion()

        except Exception as e:
            print(f"💥 処理中に致命的なエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._stop_workers()
            self.coco_manager.save_to_json(output_json)