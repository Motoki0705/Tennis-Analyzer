import json
import os
import time
import signal
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Deque
from collections import deque

import cv2
import numpy as np
import torch
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

# タスクIDを表す型エイリアス
TaskId = Union[str, int]

# 前処理・後処理のタスク型
class PreprocessTask:
    """前処理タスクを表すクラス"""
    def __init__(
        self, 
        task_id: TaskId, 
        frames: List[np.ndarray], 
        meta_data: List[Tuple[int, Path]]
    ):
        self.task_id = task_id
        self.frames = frames
        self.meta_data = meta_data  # [(image_id, image_path), ...]
        self.timestamp = time.time()

class InferenceTask:
    """推論タスクを表すクラス"""
    def __init__(
        self, 
        task_id: TaskId, 
        tensor_data: Any, 
        meta_data: List[Tuple[int, Path]],
        original_frames: Optional[List[np.ndarray]] = None
    ):
        self.task_id = task_id
        self.tensor_data = tensor_data  # モデル入力テンソル
        self.meta_data = meta_data  # [(image_id, image_path), ...]
        self.original_frames = original_frames  # 元の画像（後処理に必要な場合）
        self.timestamp = time.time()

class PostprocessTask:
    """後処理タスクを表すクラス"""
    def __init__(
        self, 
        task_id: TaskId, 
        inference_output: Any, 
        meta_data: List[Tuple[int, Path]],
        original_frames: Optional[List[np.ndarray]] = None
    ):
        self.task_id = task_id
        self.inference_output = inference_output  # 推論結果
        self.meta_data = meta_data  # [(image_id, image_path), ...]
        self.original_frames = original_frames  # 元の画像（後処理に必要な場合）
        self.timestamp = time.time()

# タイムアウトエラー用の例外クラス
class PredictionTimeoutError(Exception):
    """モデル予測がタイムアウトした場合に発生する例外"""
    pass

# タイムアウトハンドラ
def timeout_handler(signum, frame):
    raise PredictionTimeoutError("モデル予測がタイムアウトしました")

class MultiFlowAnnotator:
    """
    マルチフローアーキテクチャを利用して画像ディレクトリを効率的に処理し、
    Ball/Court/Poseの推論結果をCOCO形式JSONファイルに書き出す。
    
    前処理→推論→後処理のパイプラインを並列化し、GPU使用効率を最大化する。
    """

    def __init__(
        self,
        ball_predictor,
        court_predictor,
        pose_predictor,
        batch_sizes: dict = None,
        ball_vis_thresh: float = 0.5,
        court_vis_thresh: float = 0.6,
        pose_vis_thresh: float = 0.5,
        preprocess_workers: int = 4,  # 前処理用ワーカー数
        postprocess_workers: int = 4,  # 後処理用ワーカー数
        max_queue_size: int = 8,  # キューの最大サイズ
    ):
        # 予測器
        self.ball_predictor = ball_predictor
        self.court_predictor = court_predictor
        self.pose_predictor = pose_predictor
        
        # player_predictorは使用せず、pose_predictorの内部機能でプレーヤー検出を行う
        self.player_predictor = None
        self.use_separate_player_detector = False
        
        # バッチサイズ設定
        self.batch_sizes = batch_sizes or {"ball": 1, "court": 1, "pose": 1}
        
        # ボールの時系列予測に必要なスライディングウィンドウ
        self.ball_sliding_window: List[np.ndarray] = []
        
        # 閾値設定
        self.ball_vis_thresh = ball_vis_thresh
        self.court_vis_thresh = court_vis_thresh
        self.pose_vis_thresh = pose_vis_thresh
        
        # 画像IDとパスのマッピング
        self.image_id_map = {}
        
        # COCO出力データ構造
        self.coco_output: Dict[str, Any] = {
            "info": {
                "description": "Annotated Images with Multi-Flow Predictions",
                "version": "2.0",
                "year": datetime.now().year,
                "contributor": "MultiFlowAnnotator",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "Placeholder License", "url": ""}],
            "categories": [BALL_CATEGORY, PLAYER_CATEGORY, COURT_CATEGORY],
            "images": [],
            "annotations": [],
        }
        
        self.annotation_id_counter = 1
        self.image_id_counter = 1
        
        # スレッドプール設定
        self.preprocess_workers = preprocess_workers
        self.postprocess_workers = postprocess_workers
        self.max_queue_size = max_queue_size
        
        # 処理キュー
        self.ball_preprocess_queue = queue.Queue(maxsize=max_queue_size)
        self.ball_inference_queue = queue.Queue(maxsize=max_queue_size)
        self.ball_postprocess_queue = queue.Queue(maxsize=max_queue_size)
        
        self.court_preprocess_queue = queue.Queue(maxsize=max_queue_size)
        self.court_inference_queue = queue.Queue(maxsize=max_queue_size)
        self.court_postprocess_queue = queue.Queue(maxsize=max_queue_size)
        
        # プレーヤー検出関連のキューはpose_predictorで処理するため削除
        
        self.pose_preprocess_queue = queue.Queue(maxsize=max_queue_size)
        self.pose_inference_queue = queue.Queue(maxsize=max_queue_size)
        self.pose_postprocess_queue = queue.Queue(maxsize=max_queue_size)
        
        # スレッドプール
        self.preprocess_pool = None
        self.postprocess_pool = None
        
        # スレッド管理
        self.threads = []
        self.running = False
        
        # タスク処理ログ
        self.processed_tasks = {
            "ball": 0,
            "court": 0,
            "player": 0,  # プレーヤー検出カウントは残す（pose_predictorで処理）
            "pose": 0
        }
        
        # デバッグ用ログ出力
        self.debug = False
        
        # ロック
        self.annotation_lock = threading.Lock()

    def _add_image_entry(
        self,
        file_path: Path,
        height: int,
        width: int,
        base_dir: Path = None,
    ) -> int:
        """
        COCO形式の画像エントリを追加する
        
        Args:
            file_path: 画像ファイルのパス
            height: 画像の高さ
            width: 画像の幅
            base_dir: 基準ディレクトリ（相対パス計算用）
            
        Returns:
            追加された画像のID
        """
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

        with self.annotation_lock:
            self.coco_output["images"].append(entry)
            img_id = self.image_id_counter
            self.image_id_counter += 1
            
        return img_id

    def _add_ball_annotation(self, image_id: int, ball_res: Dict):
        """
        ボールアノテーションを追加する
        
        Args:
            image_id: 画像ID
            ball_res: ボール検出結果
        """
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
        
        with self.annotation_lock:
            self.coco_output["annotations"].append(ann)
            self.annotation_id_counter += 1

    def _add_court_annotation(self, image_id: int, court_kps: List[Dict]):
        """
        コートアノテーションを追加する
        
        Args:
            image_id: 画像ID
            court_kps: コート検出結果
        """
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
        
        with self.annotation_lock:
            self.coco_output["annotations"].append(ann)
            self.annotation_id_counter += 1

    def _add_pose_annotations(self, image_id: int, pose_results: List[Dict]):
        """
        ポーズアノテーションを追加する
        
        Args:
            image_id: 画像ID
            pose_results: ポーズ検出結果
        """
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
            
            with self.annotation_lock:
                self.coco_output["annotations"].append(ann)
                self.annotation_id_counter += 1
                # プレーヤー検出カウントを増やす
                self.processed_tasks["player"] += 1

    def _extract_game_clip_ids(self, img_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """
        パスからゲームIDとクリップIDを抽出する
        
        Args:
            img_path: 画像ファイルのパス
            
        Returns:
            (ゲームID, クリップID)のタプル
        """
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
        """
        ファイル名からフレーム番号を抽出する
        
        Args:
            path: 画像ファイルのパス
            
        Returns:
            フレーム番号
        """
        digits = ''.join(filter(str.isdigit, path.stem))
        if digits:
            return int(digits)
        return 0
        
    def _load_frame(self, img_path: Path) -> Tuple[Optional[np.ndarray], int]:
        """
        画像を読み込み、対応する画像IDとともに返す
        
        Args:
            img_path: 画像ファイルのパス
            
        Returns:
            (画像データ, 画像ID)のタプル
        """
        # 画像読み込み
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"警告: 画像の読み込みに失敗しました: {img_path}")
            return None, None
        
        # 画像IDを取得
        img_id = self.image_id_map.get(img_path)
        if img_id is None:
            print(f"警告: 画像IDが見つかりません: {img_path}")
            return None, None
        
        return frame, img_id
        
    def _validate_input_output_paths(self, input_dir: Path, output_json: Path) -> None:
        """
        入出力パスを検証する
        
        Args:
            input_dir: 入力ディレクトリのパス
            output_json: 出力JSONファイルのパス
        """
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"入力ディレクトリが存在しないか、ディレクトリではありません: {input_dir}")

        # 出力JSONファイルのディレクトリが存在しない場合は作成
        output_json.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_coco_data(self) -> None:
        """COCOデータを初期化する"""
        self.coco_output["images"].clear()
        self.coco_output["annotations"].clear()
        self.annotation_id_counter = 1
        self.image_id_counter = 1
        self.image_id_map = {}
        
    def _save_coco_annotations(self, output_json: Path) -> None:
        """
        COCO形式のアノテーションを保存する
        
        Args:
            output_json: 出力JSONファイルのパス
        """
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(self.coco_output, f, ensure_ascii=False, indent=2)

        print(f"✅ 完了！COCO形式アノテーションを保存しました: {output_json}")
        print(f"総画像数: {len(self.coco_output['images'])}")
        print(f"総アノテーション数: {len(self.coco_output['annotations'])}")
        
        # 各カテゴリの統計情報
        category_counts = {}
        for ann in self.coco_output["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in category_counts:
                category_counts[cat_id] = 0
            category_counts[cat_id] += 1
            
        for cat in self.coco_output["categories"]:
            cat_id = cat["id"]
            count = category_counts.get(cat_id, 0)
            print(f"  - {cat['name']} アノテーション数: {count}")

    def _ball_preprocess_worker(self):
        """ボール検出の前処理を行うワーカー"""
        while self.running:
            try:
                task = self.ball_preprocess_queue.get(timeout=0.1)
                if self.debug:
                    print(f"ボール前処理タスク取得: {task.task_id}")
                
                try:
                    # スライディングウィンドウを考慮
                    clips = []
                    for frame in task.frames:
                        # スライディングウィンドウに追加
                        self.ball_sliding_window.append(frame.copy())
                        if len(self.ball_sliding_window) > self.ball_predictor.num_frames:
                            self.ball_sliding_window.pop(0)
                        
                        # スライディングウィンドウが十分な長さになったら処理
                        if len(self.ball_sliding_window) == self.ball_predictor.num_frames:
                            clips.append(list(self.ball_sliding_window))
                    
                    if clips:
                        # 前処理
                        processed = self.ball_predictor.preprocess(clips)
                        
                        # 推論キューに追加
                        inference_task = InferenceTask(
                            task_id=task.task_id,
                            tensor_data=processed,
                            meta_data=task.meta_data,
                            original_frames=clips
                        )
                        self.ball_inference_queue.put(inference_task)
                        
                        if self.debug:
                            print(f"ボール推論キューに追加: {task.task_id}")
                except Exception as e:
                    print(f"ボール前処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.ball_preprocess_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"ボール前処理ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _court_preprocess_worker(self):
        """コート検出の前処理を行うワーカー"""
        while self.running:
            try:
                task = self.court_preprocess_queue.get(timeout=0.1)
                if self.debug:
                    print(f"コート前処理タスク取得: {task.task_id}")
                
                try:
                    # 前処理
                    processed, original_shapes = self.court_predictor.preprocess(task.frames)
                    
                    # 推論キューに追加
                    inference_task = InferenceTask(
                        task_id=task.task_id,
                        tensor_data=processed,
                        meta_data=task.meta_data,
                        original_frames=task.frames
                    )
                    self.court_inference_queue.put(inference_task)
                    
                    if self.debug:
                        print(f"コート推論キューに追加: {task.task_id}")
                except Exception as e:
                    print(f"コート前処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.court_preprocess_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"コート前処理ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _pose_preprocess_worker(self):
        """ポーズ検出の前処理を行うワーカー"""
        while self.running:
            try:
                task = self.pose_preprocess_queue.get(timeout=0.1)
                if self.debug:
                    print(f"ポーズ前処理タスク取得: {task.task_id}")
                
                try:
                    # 前処理
                    processed = self.pose_predictor.preprocess_detection(task.frames)
                    
                    # 推論キューに追加
                    inference_task = InferenceTask(
                        task_id=task.task_id,
                        tensor_data=processed,
                        meta_data=task.meta_data,
                        original_frames=task.frames
                    )
                    self.pose_inference_queue.put(inference_task)
                    
                    if self.debug:
                        print(f"ポーズ推論キューに追加: {task.task_id}")
                except Exception as e:
                    print(f"ポーズ前処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.pose_preprocess_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"ポーズ前処理ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _ball_inference_worker(self):
        """ボール検出の推論を行うワーカー"""
        while self.running:
            try:
                task = self.ball_inference_queue.get(timeout=0.1)
                if self.debug:
                    print(f"ボール推論タスク取得: {task.task_id}")
                
                try:
                    # 推論実行
                    with torch.no_grad():
                        preds = self.ball_predictor.inference(task.tensor_data)
                    
                    # 後処理キューに追加
                    postprocess_task = PostprocessTask(
                        task_id=task.task_id,
                        inference_output=preds,
                        meta_data=task.meta_data,
                        original_frames=task.original_frames
                    )
                    self.ball_postprocess_queue.put(postprocess_task)
                    
                    if self.debug:
                        print(f"ボール後処理キューに追加: {task.task_id}")
                except Exception as e:
                    print(f"ボール推論エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.ball_inference_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"ボール推論ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _court_inference_worker(self):
        """コート検出の推論を行うワーカー"""
        while self.running:
            try:
                task = self.court_inference_queue.get(timeout=0.1)
                if self.debug:
                    print(f"コート推論タスク取得: {task.task_id}")
                
                try:
                    # 推論実行
                    with torch.no_grad():
                        preds = self.court_predictor.inference(task.tensor_data)
                    
                    # 後処理キューに追加
                    postprocess_task = PostprocessTask(
                        task_id=task.task_id,
                        inference_output=preds,
                        meta_data=task.meta_data,
                        original_frames=task.original_frames
                    )
                    self.court_postprocess_queue.put(postprocess_task)
                    
                    if self.debug:
                        print(f"コート後処理キューに追加: {task.task_id}")
                except Exception as e:
                    print(f"コート推論エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.court_inference_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"コート推論ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _pose_inference_worker(self):
        """ポーズ検出の推論を行うワーカー"""
        while self.running:
            try:
                task = self.pose_inference_queue.get(timeout=0.1)
                if self.debug:
                    print(f"ポーズ推論タスク取得: {task.task_id}")
                
                try:
                    # 推論実行
                    with torch.no_grad():
                        preds = self.pose_predictor.inference_detection(task.tensor_data)
                    
                    # 後処理キューに追加
                    postprocess_task = PostprocessTask(
                        task_id=task.task_id,
                        inference_output=preds,
                        meta_data=task.meta_data,
                        original_frames=task.original_frames
                    )
                    self.pose_postprocess_queue.put(postprocess_task)
                    
                    if self.debug:
                        print(f"ポーズ後処理キューに追加: {task.task_id}")
                except Exception as e:
                    print(f"ポーズ推論エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.pose_inference_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"ポーズ推論ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _ball_postprocess_worker(self):
        """ボール検出の後処理を行うワーカー"""
        while self.running:
            try:
                task = self.ball_postprocess_queue.get(timeout=0.1)
                if self.debug:
                    print(f"ボール後処理タスク取得: {task.task_id}")
                
                try:
                    # 後処理実行
                    results = self.ball_predictor.postprocess(task.inference_output, task.original_frames)
                    
                    # アノテーション追加
                    for (img_id, _), result in zip(task.meta_data, results, strict=False):
                        self._add_ball_annotation(img_id, result)
                    
                    self.processed_tasks["ball"] += len(results)
                    
                    if self.debug:
                        print(f"ボール後処理完了: {task.task_id}, {len(results)}件")
                except Exception as e:
                    print(f"ボール後処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.ball_postprocess_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"ボール後処理ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _court_postprocess_worker(self):
        """コート検出の後処理を行うワーカー"""
        while self.running:
            try:
                task = self.court_postprocess_queue.get(timeout=0.1)
                if self.debug:
                    print(f"コート後処理タスク取得: {task.task_id}")
                
                try:
                    # 後処理実行
                    shapes = [frame.shape[:2] for frame in task.original_frames]
                    kps_list, _ = self.court_predictor.postprocess(task.inference_output, shapes)
                    
                    # アノテーション追加
                    for (img_id, _), kps in zip(task.meta_data, kps_list, strict=False):
                        self._add_court_annotation(img_id, kps)
                    
                    self.processed_tasks["court"] += len(kps_list)
                    
                    if self.debug:
                        print(f"コート後処理完了: {task.task_id}, {len(kps_list)}件")
                except Exception as e:
                    print(f"コート後処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.court_postprocess_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"コート後処理ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _pose_postprocess_worker(self):
        """ポーズ検出の後処理を行うワーカー"""
        while self.running:
            try:
                task = self.pose_postprocess_queue.get(timeout=0.1)
                if self.debug:
                    print(f"ポーズ後処理タスク取得: {task.task_id}")
                
                try:
                    # 後処理実行
                    # 元のコード: batch_boxes, batch_scores, batch_valid, images_for_pose = task.inference_output
                    # これは間違い。inference_outputはDETRモデルの出力で、これをpostprocess_detectionに渡す必要がある
                    
                    # 物体検出の後処理
                    det_outputs = task.inference_output
                    batch_boxes, batch_scores, batch_valid, images_for_pose = self.pose_predictor.postprocess_detection(
                        det_outputs, task.original_frames
                    )
                    
                    if not images_for_pose:
                        # プレーヤー検出が一件もない場合はスキップ
                        self.pose_postprocess_queue.task_done()
                        continue
                    
                    # ViTPose処理
                    pose_inputs = self.pose_predictor.preprocess_pose(images_for_pose, batch_boxes)
                    pose_outputs = self.pose_predictor.inference_pose(pose_inputs)
                    pose_results = self.pose_predictor.postprocess_pose(
                        pose_outputs, batch_boxes, batch_scores, batch_valid, len(task.original_frames)
                    )
                    
                    # アノテーション追加
                    for (img_id, _), poses in zip(task.meta_data, pose_results, strict=False):
                        self._add_pose_annotations(img_id, poses)
                    
                    self.processed_tasks["pose"] += len(task.meta_data)
                    
                    if self.debug:
                        print(f"ポーズ後処理完了: {task.task_id}, {len(task.meta_data)}件")
                except Exception as e:
                    print(f"ポーズ後処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                
                # タスク完了を通知
                self.pose_postprocess_queue.task_done()
            
            except queue.Empty:
                # タイムアウト - 次のループへ
                continue
            except Exception as e:
                print(f"ポーズ後処理ワーカーエラー: {e}")
                import traceback
                traceback.print_exc()
    
    def _collect_and_group_image_files(
        self, input_dir: Path, image_extensions: List[str] = None
    ) -> Dict[Tuple[int, int], List[Path]]:
        """
        画像ファイルを収集してゲームIDとクリップIDでグループ化する
        
        Args:
            input_dir: 入力ディレクトリのパス
            image_extensions: 処理対象の画像拡張子リスト
            
        Returns:
            ゲームIDとクリップIDでグループ化された画像ファイルの辞書
        """
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
        
    def _prepare_image_metadata(
        self, input_dir: Path, grouped_files: Dict[Tuple[int, int], List[Path]], sorted_group_keys: List[Tuple[int, int]]
    ) -> int:
        """
        画像メタデータを準備してCOCOエントリを作成する
        
        Args:
            input_dir: 入力ディレクトリのパス
            grouped_files: ゲームIDとクリップIDでグループ化された画像ファイルの辞書
            sorted_group_keys: ソートされたグループキーのリスト
            
        Returns:
            画像の総数
        """
        all_image_entries = []
        
        # 画像の総数を計算
        total_images = sum(len(grouped_files[key]) for key in sorted_group_keys)
        
        print(f"画像メタデータの生成を開始: 総数 {total_images}枚")
        
        # 全体の進捗表示用
        with tqdm(total=total_images, desc="画像メタデータ生成中") as meta_pbar:
            # 各グループごとに画像メタデータを準備
            for group_key in sorted_group_keys:
                game_id, clip_id = group_key
                clip_images = grouped_files[group_key]
                
                # グループの情報表示
                print(f"メタデータ処理中: Game {game_id}, Clip {clip_id} - {len(clip_images)}フレーム")
                
                # フレーム番号で正しくソート
                clip_images.sort(key=self._extract_frame_number)
                
                # イメージエントリを生成
                for img_path in clip_images:
                    try:
                        # 画像の情報を取得（サイズなど）
                        img = cv2.imread(str(img_path))
                        if img is None:
                            print(f"警告: 画像が読み込めません: {img_path}")
                            meta_pbar.update(1)
                            continue
                        
                        height, width = img.shape[:2]
                        
                        # COCOイメージエントリ追加
                        img_id = self._add_image_entry(img_path, height, width, input_dir)
                        
                        # 画像ID マッピング
                        self.image_id_map[img_path] = img_id
                        
                        # エントリーとパスのペアを保存
                        entry = {"id": img_id, "game_id": game_id, "clip_id": clip_id}
                        all_image_entries.append((entry, img_path))
                        
                        meta_pbar.update(1)
                    except Exception as e:
                        print(f"画像メタデータ処理中にエラーが発生しました: {img_path}, {e}")
                        meta_pbar.update(1)
                        continue
        
        # グループごとに再構築（ID順に整理）
        self.grouped_entries = {}
        for group_key in sorted_group_keys:
            self.grouped_entries[group_key] = []
            for entry, img_path in all_image_entries:
                curr_game = entry.get("game_id", -1)
                curr_clip = entry.get("clip_id", -1)
                if (curr_game, curr_clip) == group_key:
                    self.grouped_entries[group_key].append((entry["id"], img_path))
        
        print(f"画像メタデータの生成とグループ化が完了しました")
        
        # 総画像数を返す
        return len(all_image_entries)
        
    def _start_workers(self):
        """ワーカースレッドを開始する"""
        self.running = True
        
        # 前処理スレッドプール
        self.preprocess_pool = ThreadPoolExecutor(max_workers=self.preprocess_workers)
        
        # 後処理スレッドプール
        self.postprocess_pool = ThreadPoolExecutor(max_workers=self.postprocess_workers)
        
        # ボールワーカー
        ball_preprocess_thread = threading.Thread(target=self._ball_preprocess_worker, daemon=True)
        ball_inference_thread = threading.Thread(target=self._ball_inference_worker, daemon=True)
        ball_postprocess_thread = threading.Thread(target=self._ball_postprocess_worker, daemon=True)
        
        # コートワーカー
        court_preprocess_thread = threading.Thread(target=self._court_preprocess_worker, daemon=True)
        court_inference_thread = threading.Thread(target=self._court_inference_worker, daemon=True)
        court_postprocess_thread = threading.Thread(target=self._court_postprocess_worker, daemon=True)
        
        # ポーズワーカー (プレーヤー検出も担当)
        pose_preprocess_thread = threading.Thread(target=self._pose_preprocess_worker, daemon=True)
        pose_inference_thread = threading.Thread(target=self._pose_inference_worker, daemon=True)
        pose_postprocess_thread = threading.Thread(target=self._pose_postprocess_worker, daemon=True)
        
        ball_preprocess_thread.start()
        ball_inference_thread.start()
        ball_postprocess_thread.start()
        
        court_preprocess_thread.start()
        court_inference_thread.start()
        court_postprocess_thread.start()
        
        pose_preprocess_thread.start()
        pose_inference_thread.start()
        pose_postprocess_thread.start()
        
        self.threads.extend([
            ball_preprocess_thread,
            ball_inference_thread,
            ball_postprocess_thread,
            court_preprocess_thread,
            court_inference_thread,
            court_postprocess_thread,
            pose_preprocess_thread,
            pose_inference_thread,
            pose_postprocess_thread,
        ])
        
        print(f"全ワーカースレッドを開始しました")
    
    def _stop_workers(self):
        """ワーカースレッドを停止する"""
        print("ワーカースレッドの停止を開始...")
        self.running = False
        
        # キューのタスクが処理されるまで待機
        timeout = 30.0  # 最大待機時間（秒）
        start_time = time.time()
        
        # 処理中のタスクをモニター
        while time.time() - start_time < timeout:
            # 各キューの状態をチェック
            queues_empty = (
                self.ball_preprocess_queue.empty() and
                self.ball_inference_queue.empty() and
                self.ball_postprocess_queue.empty() and
                self.court_preprocess_queue.empty() and
                self.court_inference_queue.empty() and
                self.court_postprocess_queue.empty() and
                self.pose_preprocess_queue.empty() and
                self.pose_inference_queue.empty() and
                self.pose_postprocess_queue.empty()
            )
            
            if queues_empty:
                break
                
            print(f"キュー内のタスク完了を待機中... (経過: {time.time() - start_time:.1f}秒)")
            time.sleep(1.0)
        
        # スレッドプールのシャットダウン
        if self.preprocess_pool:
            self.preprocess_pool.shutdown(wait=False)
        
        if self.postprocess_pool:
            self.postprocess_pool.shutdown(wait=False)
        
        # スレッド終了を待機
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        self.threads.clear()
        print("全ワーカースレッドを停止しました")
        
    def _preload_clip(self, group_key):
        """
        クリップを先読みする
        
        Args:
            group_key: (ゲームID, クリップID)のタプル
            
        Returns:
            (フレームリスト, 画像IDリスト, パスリスト)のタプル
        """
        game_id, clip_id = group_key
        
        # グループエントリーから取得
        id_path_pairs = self.grouped_entries[group_key]
        
        preloaded_frames = []
        preloaded_ids = []
        preloaded_paths = []
        
        # 並列読み込み処理
        max_workers = min(32, len(id_path_pairs))
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(self._load_frame, img_path): (img_id, img_path) 
                    for img_id, img_path in id_path_pairs
                }
                
                for future in future_to_path:
                    try:
                        frame, img_id = future.result()
                        if frame is not None and img_id is not None:
                            preloaded_frames.append(frame)
                            preloaded_ids.append(img_id)
                            preloaded_paths.append(future_to_path[future][1])
                        
                    except Exception as exc:
                        img_id, path = future_to_path[future]
                        print(f"先読み中にエラーが発生: {path}, {exc}")
        except Exception as e:
            print(f"先読み処理全体でエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        
        return (preloaded_frames, preloaded_ids, preloaded_paths)
        
    def _process_frames_in_batches(self, frames, frame_ids, frame_paths, pbar=None):
        """
        フレームをバッチで処理する
        
        Args:
            frames: 画像フレームのリスト
            frame_ids: 画像IDのリスト
            frame_paths: 画像パスのリスト
            pbar: 進捗表示用のtqdmオブジェクト
        """
        if not frames:
            return
            
        # バッチ処理のパラメータ
        ball_batch_size = self.batch_sizes.get("ball", 1)
        court_batch_size = self.batch_sizes.get("court", 1)
        pose_batch_size = self.batch_sizes.get("pose", 1)  # ポーズ処理はプレーヤー検出も含む
        
        # ボール処理
        for i in range(0, len(frames), ball_batch_size):
            batch_end = min(i + ball_batch_size, len(frames))
            batch_frames = frames[i:batch_end]
            batch_ids = frame_ids[i:batch_end]
            batch_paths = frame_paths[i:batch_end]
            
            # メタデータ生成
            meta_data = list(zip(batch_ids, batch_paths))
            
            # 前処理タスク生成
            task_id = f"ball_{i}_{batch_end}"
            task = PreprocessTask(task_id, batch_frames, meta_data)
            
            # 前処理キューに追加
            self.ball_preprocess_queue.put(task)
            
            if self.debug:
                print(f"ボール前処理タスク追加: {task_id}, {len(batch_frames)}フレーム")
        
        # コート処理
        for i in range(0, len(frames), court_batch_size):
            batch_end = min(i + court_batch_size, len(frames))
            batch_frames = frames[i:batch_end]
            batch_ids = frame_ids[i:batch_end]
            batch_paths = frame_paths[i:batch_end]
            
            # メタデータ生成
            meta_data = list(zip(batch_ids, batch_paths))
            
            # 前処理タスク生成
            task_id = f"court_{i}_{batch_end}"
            task = PreprocessTask(task_id, batch_frames, meta_data)
            
            # 前処理キューに追加
            self.court_preprocess_queue.put(task)
            
            if self.debug:
                print(f"コート前処理タスク追加: {task_id}, {len(batch_frames)}フレーム")
        
        # ポーズ処理（プレーヤー検出を含む）
        for i in range(0, len(frames), pose_batch_size):
            batch_end = min(i + pose_batch_size, len(frames))
            batch_frames = frames[i:batch_end]
            batch_ids = frame_ids[i:batch_end]
            batch_paths = frame_paths[i:batch_end]
            
            # メタデータ生成
            meta_data = list(zip(batch_ids, batch_paths))
            
            # 前処理タスク生成
            task_id = f"pose_{i}_{batch_end}"
            task = PreprocessTask(task_id, batch_frames, meta_data)
            
            # 前処理キューに追加
            self.pose_preprocess_queue.put(task)
            
            if self.debug:
                print(f"ポーズ前処理タスク追加: {task_id}, {len(batch_frames)}フレーム")
        
        # 進捗バーを更新
        if pbar:
            pbar.update(len(frames))
    
    def _wait_for_completion(self, timeout=1.0, max_attempts=60):
        """すべてのキューが空になるまで待機する"""
        attempts = 0
        while attempts < max_attempts:
            queues_empty = (
                self.ball_preprocess_queue.empty() and
                self.ball_inference_queue.empty() and
                self.ball_postprocess_queue.empty() and
                self.court_preprocess_queue.empty() and
                self.court_inference_queue.empty() and
                self.court_postprocess_queue.empty() and
                self.pose_preprocess_queue.empty() and
                self.pose_inference_queue.empty() and
                self.pose_postprocess_queue.empty()
            )
            
            if queues_empty:
                print("すべてのキューが空になりました。処理完了を待機中...")
                break
                
            time.sleep(timeout)
            attempts += 1
            
            if attempts % 10 == 0:
                print(f"キュー待機中... 試行回数: {attempts}")
        
        # スレッドプールのシャットダウン
        if self.preprocess_pool:
            self.preprocess_pool.shutdown(wait=True)
        if self.postprocess_pool:
            self.postprocess_pool.shutdown(wait=True)
        
        # 処理統計の出力
        if self.debug:
            print(f"処理統計:")
            print(f"ボール処理数: {self.processed_tasks['ball']}")
            print(f"コート処理数: {self.processed_tasks['court']}")
            print(f"プレーヤー処理数: {self.processed_tasks['player']}")
            print(f"ポーズ処理数: {self.processed_tasks['pose']}")
        
        return queues_empty
        
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
        try:
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
            
            # ワーカースレッドを開始
            self._start_workers()
            
            # 先読み用の変数
            next_clip_idx = 0
            next_clip_future = None
            
            # 進捗表示用のtqdm
            print(f"\n推論処理を開始します: 総画像数 {total_images}枚")
            with tqdm(total=total_images, desc="アノテーション処理中") as pbar:
                # 最初のクリップを先読み
                if next_clip_idx < len(sorted_group_keys):
                    with ThreadPoolExecutor(max_workers=1) as pre_executor:
                        next_group_key = sorted_group_keys[next_clip_idx]
                        next_clip_future = pre_executor.submit(
                            self._preload_clip, next_group_key
                        )
                        next_clip_idx += 1
                
                # 処理済みクリップ数のカウンタ
                processed_clips = 0
                
                # 各クリップごとに処理
                for idx, group_key in enumerate(sorted_group_keys):
                    game_id, clip_id = group_key
                    print(f"処理中: Game {game_id}, Clip {clip_id} [{idx+1}/{len(sorted_group_keys)}]")
                    
                    try:
                        # 先読みデータの取得（あれば）
                        if next_clip_future is not None:
                            try:
                                frames, frame_ids, frame_paths = next_clip_future.result()
                                print(f"先読みデータ使用: Game {game_id}, Clip {clip_id} - {len(frames)}フレーム")
                            except Exception as e:
                                print(f"先読みデータの取得に失敗: {e}")
                                frames, frame_ids, frame_paths = [], [], []
                        else:
                            # 通常の読み込み
                            frames, frame_ids, frame_paths = self._preload_clip(group_key)
                        
                        # 次のクリップの先読み開始
                        if next_clip_idx < len(sorted_group_keys):
                            with ThreadPoolExecutor(max_workers=1) as pre_executor:
                                next_group_key = sorted_group_keys[next_clip_idx]
                                next_clip_future = pre_executor.submit(
                                    self._preload_clip, next_group_key
                                )
                                next_clip_idx += 1
                                print(f"次のクリップを先読み中: Game {next_group_key[0]}, Clip {next_group_key[1]}")
                        else:
                            next_clip_future = None
                        
                        if not frames:
                            print(f"警告: クリップ内に有効なフレームがありません: Game {game_id}, Clip {clip_id}")
                            continue
                        
                        # スライディングウィンドウをリセット
                        self.ball_sliding_window.clear()
                        
                        # フレーム処理
                        self._process_frames_in_batches(frames, frame_ids, frame_paths, pbar)
                        
                        print(f"バッチタスク登録完了: Game {game_id}, Clip {clip_id}")
                        processed_clips += 1
                        
                        # メモリ解放
                        frames.clear()
                        frame_ids.clear()
                        frame_paths.clear()
                        
                        # 明示的にGCを呼び出してメモリを解放
                        import gc
                        gc.collect()
                        
                    except Exception as e:
                        print(f"クリップ処理中にエラーが発生しました: Game {game_id}, Clip {clip_id}, エラー: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"全クリップの処理タスク登録完了: {processed_clips}/{len(sorted_group_keys)}クリップ")
                
                # すべてのタスクが完了するまで待機
                self._wait_for_completion()
            
            # ワーカースレッドを停止
            self._stop_workers()
            
            # JSON 出力
            self._save_coco_annotations(output_json)
            
        except Exception as e:
            print(f"処理中に例外が発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            # ワーカースレッドを停止
            self._stop_workers() 