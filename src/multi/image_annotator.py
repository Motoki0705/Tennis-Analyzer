import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import cv2
import numpy as np
from tqdm import tqdm

# FrameAnnotatorと同様のカテゴリ定義
PLAYER_CATEGORY = {
    "id": 2,
    "name": "player",
    "supercategory": "person",
    "keypoints": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "skeleton": [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ],
}

COURT_CATEGORY = {
    "id": 3,
    "name": "court",
    "supercategory": "field",
    "keypoints": [f"pt{i}" for i in range(15)],
    "skeleton": [],
}

BALL_CATEGORY = {
    "id": 1,
    "name": "ball",
    "supercategory": "sports",
    "keypoints": ["center"],
    "skeleton": [],
}


class ImageAnnotator:
    """
    画像ディレクトリを入力として、Ball/Court/Poseの推論結果を
    単一のCOCO形式JSONファイルに書き出す。各タスクでバッチ推論をサポート。
    """

    def __init__(
        self,
        ball_predictor,
        court_predictor,
        pose_predictor,
        batch_sizes: dict = None,
        ball_vis_thresh: float = 0.5,
        court_vis_thresh: float = 0.5,
        pose_vis_thresh: float = 0.5,
    ):
        self.ball_predictor = ball_predictor
        self.court_predictor = court_predictor
        self.pose_predictor = pose_predictor

        self.batch_sizes = batch_sizes or {"ball": 1, "court": 1, "pose": 1}

        # ボールの時系列予測に必要なスライディングウィンドウ
        self.ball_sliding_window: List[np.ndarray] = []

        self.ball_vis_thresh = ball_vis_thresh
        self.court_vis_thresh = court_vis_thresh
        self.pose_vis_thresh = pose_vis_thresh

        # 画像IDとパスのマッピング
        self.image_id_map = {}

        self.coco_output: Dict[str, Any] = {
            "info": {
                "description": "Annotated Images with Batched Predictions",
                "version": "1.1",
                "year": datetime.now().year,
                "contributor": "ImageAnnotator",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "Placeholder License", "url": ""}],
            "categories": [BALL_CATEGORY, PLAYER_CATEGORY, COURT_CATEGORY],
            "images": [],
            "annotations": [],
        }

        self.annotation_id_counter = 1
        self.image_id_counter = 1

    def _add_image_entry(
        self,
        file_path: Path,
        height: int,
        width: int,
        base_dir: Path = None,
    ) -> int:
        # 入力ディレクトリからの相対パスを取得
        if base_dir is not None and file_path.is_relative_to(base_dir):
            rel_path = str(file_path.relative_to(base_dir))
        else:
            # base_dirが指定されていないか、相対パス化に失敗した場合はフルパスを使用
            rel_path = str(file_path)

        # パスの各部分を取得
        path_parts = file_path.parts
        
        # ゲームIDとクリップIDの初期化
        game_id = None
        clip_id = None
        
        # パスの各部分を確認してゲームIDとクリップIDを抽出
        for part in path_parts:
            if part.lower().startswith('game'):
                try:
                    game_id = int(part[4:])  # 'game' を除いた部分を整数に変換
                except ValueError:
                    pass
            elif part.lower().startswith('clip'):
                try:
                    clip_id = int(part[4:])  # 'clip' を除いた部分を整数に変換
                except ValueError:
                    pass

        entry = {
            "id": self.image_id_counter,
            "file_name": file_path.name,
            "original_path": rel_path,
            "height": height,
            "width": width,
            "license": 1,
        }

        # ゲームIDとクリップIDが抽出できた場合は追加
        if game_id is not None:
            entry["game_id"] = game_id
        if clip_id is not None:
            entry["clip_id"] = clip_id

        self.coco_output["images"].append(entry)
        img_id = self.image_id_counter
        self.image_id_counter += 1
        return img_id

    def _add_ball_annotation(self, image_id: int, ball_res: Dict):
        if not ball_res:
            return
        x, y, conf = (
            ball_res.get("x"),
            ball_res.get("y"),
            ball_res.get("confidence", 0.0),
        )
        if x is None or y is None:
            return
        visibility = 2 if conf >= self.ball_vis_thresh else 1
        ann = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": BALL_CATEGORY["id"],
            "keypoints": [float(x), float(y), visibility],
            "num_keypoints": 1 if visibility > 0 else 0,
            "iscrowd": 0,
            "score": float(conf),
        }
        self.coco_output["annotations"].append(ann)
        self.annotation_id_counter += 1

    def _add_court_annotation(self, image_id: int, court_kps: List[Dict]):
        if not court_kps:
            return
        num_expected = len(COURT_CATEGORY["keypoints"])
        keypoints_flat, keypoints_scores = [], []
        num_visible = 0

        for i in range(num_expected):
            if i < len(court_kps):
                kp = court_kps[i]
                x, y, conf = (
                    kp.get("x", 0.0),
                    kp.get("y", 0.0),
                    kp.get("confidence", 0.0),
                )
            else:
                x, y, conf = 0.0, 0.0, 0.0

            if conf >= self.court_vis_thresh:
                v = 2
            elif conf > 0.01:
                v = 1
            else:
                v = 0

            keypoints_flat.extend([float(x), float(y), v])
            keypoints_scores.append(float(conf))
            if v > 0:
                num_visible += 1

        ann = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "category_id": COURT_CATEGORY["id"],
            "keypoints": keypoints_flat,
            "num_keypoints": num_visible,
            "keypoints_scores": keypoints_scores,
            "iscrowd": 0,
        }
        self.coco_output["annotations"].append(ann)
        self.annotation_id_counter += 1

    def _add_pose_annotations(self, image_id: int, pose_results: List[Dict]):
        if not pose_results:
            return
        for res in pose_results:
            bbox = res.get("bbox")
            kps = res.get("keypoints")
            scores = res.get("scores")
            det_score = res.get("det_score", 0.0)
            if not bbox or not kps or not scores:
                continue
            flat, num_vis = [], 0
            for (x, y), s in zip(kps, scores, strict=False):
                if s >= self.pose_vis_thresh:
                    v = 2
                    num_vis += 1
                elif s > 0.01:
                    v = 1
                else:
                    v = 0
                flat.extend([float(x), float(y), v])

            ann = {
                "id": self.annotation_id_counter,
                "image_id": image_id,
                "category_id": PLAYER_CATEGORY["id"],
                "bbox": [float(b) for b in bbox],
                "area": float(bbox[2] * bbox[3]),
                "keypoints": flat,  # [x0, y0, v0, x1, y1, v1, ...]
                "keypoints_scores": [float(s) for s in scores],
                "num_keypoints": num_vis,
                "iscrowd": 0,
                "score": float(det_score),
            }
            self.coco_output["annotations"].append(ann)
            self.annotation_id_counter += 1

    def _process_batch(
        self,
        predictor,
        frames_buffer: List,
        meta_buffer: List[Tuple[int, Path]],
        cache: Dict[int, Any],
    ):
        if not frames_buffer:
            return

        # モデル予測の呼び出し
        if predictor == self.ball_predictor:
            preds = predictor.predict(frames_buffer)
        else:
            out = predictor.predict(frames_buffer)
            # courtはヒートマップそのものをindex1で出力するため、out[0]だけを取り出す
            preds = out[0] if predictor == self.court_predictor else out

        # キャッシュ登録とアノテーション追加
        for (img_id, _), pred in zip(meta_buffer, preds, strict=False):
            cache[img_id] = pred
            if predictor == self.ball_predictor:
                self._add_ball_annotation(img_id, pred)
            elif predictor == self.court_predictor:
                self._add_court_annotation(img_id, pred)
            else:
                self._add_pose_annotations(img_id, pred)

        frames_buffer.clear()
        meta_buffer.clear()

    def _load_frame(self, img_path: Path, input_dir: Path) -> Tuple[np.ndarray, int, Path]:
        """
        画像を読み込み、返す（画像IDはimage_id_mapから取得）
        
        Args:
            img_path: 画像ファイルのパス
            input_dir: 入力ディレクトリ（相対パス計算用）
            
        Returns:
            tuple: (frame, image_id, img_path) のタプル、読み込みに失敗した場合はNoneを含む
        """
        # 画像読み込み
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"警告: 画像の読み込みに失敗しました: {img_path}")
            return None, None, img_path
        
        # 画像IDを取得（先に生成されているイメージエントリから）
        img_id = self.image_id_map.get(img_path)
        if img_id is None:
            print(f"警告: 画像IDが見つかりません: {img_path}")
            return None, None, img_path
        
        return frame, img_id, img_path

    def run(
        self,
        input_dir: Union[str, Path],
        output_json: Union[str, Path],
        image_extensions: List[str] = None,
    ):
        """
        画像ディレクトリ内の画像に対して推論を実行し、結果をCOCO形式のJSONファイルに出力する

        Args:
            input_dir: 入力画像が格納されているディレクトリのパス
            output_json: 出力JSONファイルのパス
            image_extensions: 処理対象の画像拡張子リスト（デフォルト: ['.jpg', '.jpeg', '.png']）
        """
        input_dir, output_json = Path(input_dir), Path(output_json)
        self._validate_input_output_paths(input_dir, output_json)
        self._initialize_coco_data()
        
        # 画像ファイルの収集とグループ化
        grouped_files = self._collect_and_group_image_files(input_dir, image_extensions)
        if not grouped_files:
            raise ValueError(f"指定されたディレクトリに画像ファイルが見つかりませんでした: {input_dir}")
        
        # ソートされたグループキー（ゲームID、クリップID順）
        sorted_group_keys = sorted(grouped_files.keys())
        
        # 画像メタデータの準備とCOCOエントリの作成
        total_images = self._prepare_image_metadata(input_dir, grouped_files, sorted_group_keys)
        
        # 進捗表示用のtqdmとバッチ処理の準備
        with tqdm(total=total_images, desc="画像アノテーション処理中") as pbar:
            self._process_clips_with_batching(input_dir, sorted_group_keys, pbar)

        # JSON 出力
        self._save_coco_annotations(output_json)

    def _validate_input_output_paths(self, input_dir: Path, output_json: Path) -> None:
        """入出力パスの検証"""
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"入力ディレクトリが存在しないか、ディレクトリではありません: {input_dir}")

        # 出力JSONファイルのディレクトリが存在しない場合は作成
        output_json.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_coco_data(self) -> None:
        """COCOデータの初期化"""
        self.coco_output["images"].clear()
        self.coco_output["annotations"].clear()
        self.annotation_id_counter = 1
        self.image_id_counter = 1
        self.image_id_map = {}

    def _collect_and_group_image_files(
        self, input_dir: Path, image_extensions: List[str] = None
    ) -> Dict[Tuple[int, int], List[Path]]:
        """画像ファイルを収集してゲームIDとクリップIDでグループ化"""
        # 処理対象の画像拡張子
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png']
        
        # 画像ファイルの収集
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_dir.glob(f"**/*{ext}")))
        
        # グループ化
        grouped_files = {}
        for img_path in image_files:
            game_id, clip_id = self._extract_game_clip_ids(img_path)
            
            # グループキーの作成（ゲームまたはクリップが特定できない場合は特別な値を使用）
            group_key = (game_id or -1, clip_id or -1)
            if group_key not in grouped_files:
                grouped_files[group_key] = []
            grouped_files[group_key].append(img_path)
        
        return grouped_files

    def _extract_game_clip_ids(self, img_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """パスからゲームIDとクリップIDを抽出"""
        game_id = None
        clip_id = None
        
        for part in img_path.parts:
            if part.lower().startswith('game'):
                try:
                    game_id = int(part[4:])
                except ValueError:
                    pass
            elif part.lower().startswith('clip'):
                try:
                    clip_id = int(part[4:])
                except ValueError:
                    pass
        
        return game_id, clip_id

    def _extract_frame_number(self, path: Path) -> int:
        """ファイル名からフレーム番号を抽出"""
        digits = ''.join(filter(str.isdigit, path.stem))
        if digits:
            return int(digits)
        return 0

    def _prepare_image_metadata(
        self, input_dir: Path, grouped_files: Dict[Tuple[int, int], List[Path]], sorted_group_keys: List[Tuple[int, int]]
    ) -> int:
        """画像メタデータを準備してCOCOエントリを作成"""
        all_image_entries = []
        
        # 各グループごとに画像メタデータを準備
        for group_key in sorted_group_keys:
            game_id, clip_id = group_key
            clip_images = grouped_files[group_key]
            
            # フレーム番号で正しくソート
            clip_images.sort(key=self._extract_frame_number)
            
            # イメージエントリを生成
            for img_path in clip_images:
                try:
                    # 画像の情報を取得（サイズなど）
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"警告: 画像の読み込みに失敗しました: {img_path}")
                        continue
                    
                    height, width = img.shape[:2]
                    
                    # 相対パスを取得
                    if img_path.is_relative_to(input_dir):
                        rel_path = str(img_path.relative_to(input_dir))
                    else:
                        rel_path = str(img_path)
                    
                    # イメージエントリを作成
                    image_entry = {
                        "id": self.image_id_counter,
                        "file_name": img_path.name,
                        "original_path": rel_path,
                        "height": height,
                        "width": width,
                        "license": 1,
                        "date_captured": datetime.now().isoformat(),
                    }
                    
                    # ゲームIDとクリップIDが抽出できた場合は追加
                    if game_id is not None and game_id != -1:
                        image_entry["game_id"] = game_id
                    if clip_id is not None and clip_id != -1:
                        image_entry["clip_id"] = clip_id
                    
                    # フレーム番号を抽出して追加
                    frame_num = self._extract_frame_number(img_path)
                    if frame_num > 0:
                        image_entry["frame_number"] = frame_num
                    
                    # エントリを追加
                    all_image_entries.append((image_entry, img_path))
                    self.image_id_counter += 1
                    
                except Exception as e:
                    print(f"画像メタデータの生成中にエラーが発生: {img_path}, {e}")
        
        # イメージエントリをCOCO出力に追加
        for entry, _ in all_image_entries:
            self.coco_output["images"].append(entry)
        
        # イメージIDとパスのマッピングを作成
        self.image_id_map = {img_path: entry["id"] for entry, img_path in all_image_entries}
        
        # グループごとに再構築（ID順に整理）
        self.grouped_entries = {}
        for group_key in sorted_group_keys:
            self.grouped_entries[group_key] = []
            for entry, img_path in all_image_entries:
                curr_game = entry.get("game_id", -1)
                curr_clip = entry.get("clip_id", -1)
                if (curr_game, curr_clip) == group_key:
                    self.grouped_entries[group_key].append((entry["id"], img_path))
        
        # 総画像数を返す
        return len(all_image_entries)

    def _process_clips_with_batching(
        self, input_dir: Path, sorted_group_keys: List[Tuple[int, int]], pbar: tqdm
    ) -> None:
        """クリップごとのバッチ処理を実行"""
        # バッファとキャッシュの初期化
        ball_buf, court_buf, pose_buf = [], [], []
        ball_meta, court_meta, pose_meta = [], [], []
        ball_cache, court_cache, pose_cache = {}, {}, {}
        
        # 先読み用の変数
        next_clip_idx = 0
        next_clip_future = None
        
        # 最初のクリップを先読み
        if next_clip_idx < len(sorted_group_keys):
            with ThreadPoolExecutor(max_workers=1) as pre_executor:
                next_group_key = sorted_group_keys[next_clip_idx]
                next_clip_future = pre_executor.submit(
                    self._preload_clip, next_group_key, input_dir
                )
                next_clip_idx += 1
        
        # 各クリップごとに処理
        for idx, group_key in enumerate(sorted_group_keys):
            game_id, clip_id = group_key
            
            # 先読みデータの取得（あれば）
            frames, frame_ids, valid_paths = self._get_clip_frames(
                idx, game_id, clip_id, next_clip_future
            )
            
            # 次のクリップの先読み開始
            if next_clip_idx < len(sorted_group_keys):
                with ThreadPoolExecutor(max_workers=1) as pre_executor:
                    next_group_key = sorted_group_keys[next_clip_idx]
                    next_clip_future = pre_executor.submit(
                        self._preload_clip, next_group_key, input_dir
                    )
                    next_clip_idx += 1
                    print(f"次のクリップを先読み中: Game {next_group_key[0]}, Clip {next_group_key[1]}")
            
            if not frames:
                print(f"警告: クリップ内に有効なフレームがありません: Game {game_id}, Clip {clip_id}")
                continue
            
            # クリップ切り替え時にスライディングウィンドウとバッファをリセット
            self.ball_sliding_window = []
            ball_buf, ball_meta = [], []
            court_buf, court_meta = [], []
            pose_buf, pose_meta = [], []
            
            # 各モデル（Court/Pose/Ball）でバッチ処理
            self._process_court_batches(frames, frame_ids, valid_paths, court_buf, court_meta, court_cache)
            self._process_pose_batches(frames, frame_ids, valid_paths, pose_buf, pose_meta, pose_cache)
            self._process_ball_batches(frames, frame_ids, valid_paths, ball_buf, ball_meta, ball_cache)
            
            print(f"完了: Game {game_id}, Clip {clip_id}")
            
            # メモリ解放
            frames.clear()
            frame_ids.clear()
            valid_paths.clear()

    def _preload_clip(self, group_key, input_dir_path):
        """クリップを先読みする"""
        game_id, clip_id = group_key
        # 新しいグループエントリー構造から取得
        id_path_pairs = self.grouped_entries[group_key]
        
        preloaded_frames = []
        preloaded_ids = []
        preloaded_paths = []
        
        # 並列読み込み処理
        max_workers = min(32, len(id_path_pairs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(cv2.imread, str(img_path)): (img_id, img_path) 
                for img_id, img_path in id_path_pairs
            }
            
            for future in future_to_path:
                try:
                    frame = future.result()
                    img_id, path = future_to_path[future]
                    if frame is not None:
                        preloaded_frames.append(frame)
                        preloaded_ids.append(img_id)
                        preloaded_paths.append(path)
                except Exception as exc:
                    img_id, path = future_to_path[future]
                    print(f"先読み中にエラーが発生: {path}, {exc}")
        
        return (preloaded_frames, preloaded_ids, preloaded_paths)

    def _get_clip_frames(self, idx, game_id, clip_id, next_clip_future):
        """クリップのフレームを取得（先読みまたは新規読み込み）"""
        if idx == 0 and next_clip_future is not None:
            try:
                frames, frame_ids, valid_paths = next_clip_future.result()
                print(f"先読みデータ使用: Game {game_id}, Clip {clip_id} - {len(frames)}フレーム")
                return frames, frame_ids, valid_paths
            except Exception as e:
                print(f"先読みデータの取得に失敗: {e}")
                return [], [], []
        else:
            # クリップ画像の取得と読み込み
            id_path_pairs = self.grouped_entries.get((game_id, clip_id), [])
            print(f"処理中: Game {game_id}, Clip {clip_id} - {len(id_path_pairs)}フレーム")
            
            # 並列処理で画像を読み込む
            frames = []
            frame_ids = []
            valid_paths = []
            
            # 並列読み込み処理
            max_workers = min(32, len(id_path_pairs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(cv2.imread, str(img_path)): (img_id, img_path) 
                    for img_id, img_path in id_path_pairs
                }
                
                for future in tqdm(
                    future_to_pair, 
                    desc=f"フレーム読み込み中 (Game {game_id}, Clip {clip_id})",
                    leave=False
                ):
                    try:
                        frame = future.result()
                        img_id, path = future_to_pair[future]
                        if frame is not None:
                            frames.append(frame)
                            frame_ids.append(img_id)
                            valid_paths.append(path)
                    except Exception as exc:
                        img_id, path = future_to_pair[future]
                        print(f"フレーム読み込み中にエラーが発生しました: {path}, {exc}")
            
            return frames, frame_ids, valid_paths

    def _process_court_batches(self, frames, frame_ids, valid_paths, court_buf, court_meta, court_cache):
        """コート検出のバッチ処理"""
        print(f"Court処理: {len(frames)}フレーム")
        for i in range(0, len(frames), self.batch_sizes["court"]):
            batch_frames = frames[i:i+self.batch_sizes["court"]]
            batch_ids = frame_ids[i:i+self.batch_sizes["court"]]
            batch_paths = valid_paths[i:i+self.batch_sizes["court"]]
            
            batch_meta = list(zip(batch_ids, batch_paths))
            court_buf = [frame.copy() for frame in batch_frames]
            
            print(f"Court バッチ処理: {len(court_buf)}枚")
            self._process_batch(
                self.court_predictor, court_buf, batch_meta, court_cache
            )

    def _process_pose_batches(self, frames, frame_ids, valid_paths, pose_buf, pose_meta, pose_cache):
        """ポーズ検出のバッチ処理"""
        print(f"Pose処理: {len(frames)}フレーム")
        for i in range(0, len(frames), self.batch_sizes["pose"]):
            batch_frames = frames[i:i+self.batch_sizes["pose"]]
            batch_ids = frame_ids[i:i+self.batch_sizes["pose"]]
            batch_paths = valid_paths[i:i+self.batch_sizes["pose"]]
            
            batch_meta = list(zip(batch_ids, batch_paths))
            pose_buf = [frame.copy() for frame in batch_frames]
            
            print(f"Pose バッチ処理: {len(pose_buf)}枚")
            self._process_batch(
                self.pose_predictor, pose_buf, batch_meta, pose_cache
            )

    def _process_ball_batches(self, frames, frame_ids, valid_paths, ball_buf, ball_meta, ball_cache):
        """ボール検出のバッチ処理（スライディングウィンドウ使用）"""
        print(f"Ball処理: {len(frames)}フレーム")
        
        # フレームごとにスライディングウィンドウを構築
        for i in range(len(frames)):
            # スライディングウィンドウに追加
            self.ball_sliding_window.append(frames[i].copy())
            if len(self.ball_sliding_window) > self.ball_predictor.num_frames:
                self.ball_sliding_window.pop(0)
            
            # スライディングウィンドウが十分な長さになったらバッファに追加
            if len(self.ball_sliding_window) == self.ball_predictor.num_frames:
                ball_buf.append(list(self.ball_sliding_window))
                ball_meta.append((frame_ids[i], valid_paths[i]))
                
                # バッチサイズに達したら処理
                if len(ball_buf) >= self.batch_sizes["ball"]:
                    print(f"Ball バッチ処理: {len(ball_buf)}枚")
                    self._process_batch(
                        self.ball_predictor, ball_buf, ball_meta, ball_cache
                    )
                    ball_buf, ball_meta = [], []
        
        # 残りのバッチを処理
        if ball_buf:
            print(f"Ball 残りバッチ処理: {len(ball_buf)}枚")
            self._process_batch(
                self.ball_predictor, ball_buf, ball_meta, ball_cache
            )

    def _save_coco_annotations(self, output_json: Path) -> None:
        """COCO形式のアノテーションを保存"""
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(self.coco_output, f, ensure_ascii=False, indent=2)

        print(f"✅ 完了！COCO形式アノテーションを保存しました: {output_json}")
        print(f"総画像数: {len(self.coco_output['images'])}")
        print(f"総アノテーション数: {len(self.coco_output['annotations'])}") 