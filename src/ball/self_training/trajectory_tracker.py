import copy
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm

# ロガー設定
logger = logging.getLogger(__name__)


class BallTrajectoryTracker:
    """
    テニスボールの軌跡を追跡し、自己学習データを生成するクラス。
    
    高信頼度の検出結果から時間的に一貫したボールの軌跡を推定し、
    低信頼度の検出や欠落フレームを補完します。
    
    Attributes
    ----------
    annotations : List[Dict]
        アノテーションのリスト（更新される）
    confidence_threshold : float
        高信頼度と見なす信頼度の閾値
    temporal_window : int
        軌跡推定に使用する時間ウィンドウのサイズ
    max_trajectory_gap : int
        補間可能な最大フレームギャップ
    min_trajectory_length : int
        有効な軌跡とみなす最小長
    interpolation_method : str
        補間方法（'linear', 'quadratic', 'cubic'）
    smoothing_window : int
        平滑化ウィンドウサイズ
    max_ball_speed : float
        最大ボール速度（ピクセル/フレーム）
    """

    def __init__(
        self,
        annotations: List[Dict],
        confidence_threshold: float = 0.7,
        temporal_window: int = 9,
        max_trajectory_gap: int = 5,
        min_trajectory_length: int = 7,
        interpolation_method: str = "quadratic",
        smoothing_window: int = 5,
        max_ball_speed: float = 100.0,
    ):
        """
        初期化

        Parameters
        ----------
        annotations : List[Dict]
            アノテーションのリスト（更新される）
        confidence_threshold : float, optional
            高信頼度と見なす信頼度の閾値（デフォルト: 0.7）
        temporal_window : int, optional
            軌跡推定に使用する時間ウィンドウのサイズ（デフォルト: 9）
        max_trajectory_gap : int, optional
            補間可能な最大フレームギャップ（デフォルト: 5）
        min_trajectory_length : int, optional
            有効な軌跡とみなす最小長（デフォルト: 7）
        interpolation_method : str, optional
            補間方法（デフォルト: 'quadratic'）
        smoothing_window : int, optional
            平滑化ウィンドウサイズ（デフォルト: 5）
        max_ball_speed : float, optional
            最大ボール速度（デフォルト: 100.0 ピクセル/フレーム）
        """
        self.annotations = annotations
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.max_trajectory_gap = max_trajectory_gap
        self.min_trajectory_length = min_trajectory_length
        self.interpolation_method = interpolation_method
        self.smoothing_window = smoothing_window
        self.max_ball_speed = max_ball_speed
        
        # 統計情報
        self.stats = {
            "total_trajectories": 0,
            "valid_trajectories": 0,
            "interpolated_points": 0,
            "filtered_outliers": 0,
        }

    def track_ball_in_clip(self, clip_imgs: List[Dict]) -> None:
        """
        クリップ内でボールを追跡し、アノテーションを更新する

        Parameters
        ----------
        clip_imgs : List[Dict]
            クリップの画像情報リスト
        """
        try:
            # 時系列順にソート
            clip_imgs.sort(key=lambda img: img["file_name"])
            frame_idx_map = {img["id"]: idx for idx, img in enumerate(clip_imgs)}

            # 各フレームのボールアノテーションを抽出
            frames_to_ball_anns = defaultdict(list)
            for ann in self.annotations:
                if ann.get("image_id") in frame_idx_map and ann.get("category_id") == 1:  # ボールカテゴリ
                    frames_to_ball_anns[ann["image_id"]].append(ann)

            # 高信頼度の検出からボールの軌跡を構築
            trajectories = self._build_trajectories(clip_imgs, frames_to_ball_anns, frame_idx_map)
            
            # 軌跡に基づいてアノテーションを更新
            self._update_annotations_from_trajectories(trajectories, frame_idx_map)
            
        except Exception as e:
            logger.error(f"Error tracking ball in clip: {e}")
            raise

    def _build_trajectories(
        self, clip_imgs: List[Dict], frames_to_ball_anns: Dict, frame_idx_map: Dict
    ) -> List[List[Dict]]:
        """
        高信頼度の検出からボールの軌跡を構築

        Parameters
        ----------
        clip_imgs : List[Dict]
            クリップの画像情報リスト
        frames_to_ball_anns : Dict
            フレームIDからボールアノテーションへのマッピング
        frame_idx_map : Dict
            フレームIDからインデックスへのマッピング

        Returns
        -------
        trajectories : List[List[Dict]]
            ボールの軌跡リスト
        """
        # 高信頼度の検出ポイントを収集
        high_conf_points = []
        for img in clip_imgs:
            frame_id = img["id"]
            frame_idx = frame_idx_map[frame_id]
            
            for ann in frames_to_ball_anns.get(frame_id, []):
                confidence = ann.get("score", 0.0)
                if confidence >= self.confidence_threshold:
                    x, y, w, h = ann["bbox"]
                    center_x, center_y = x + w / 2, y + h / 2
                    high_conf_points.append({
                        "frame_idx": frame_idx,
                        "frame_id": frame_id,
                        "position": (center_x, center_y),
                        "confidence": confidence,
                        "annotation": ann,
                        "width": w,
                        "height": h,
                    })
        
        # 高信頼度ポイントが不足している場合は早期リターン
        if len(high_conf_points) < self.min_trajectory_length:
            logger.debug(f"Not enough high confidence points: {len(high_conf_points)} < {self.min_trajectory_length}")
            return []
        
        # フレームインデックスでソート
        high_conf_points.sort(key=lambda p: p["frame_idx"])
        
        # 外れ値を除去
        high_conf_points = self._filter_outliers(high_conf_points)
        
        # 軌跡を構築
        trajectories = self._connect_points_to_trajectories(high_conf_points)
        
        # 統計情報を更新
        self.stats["total_trajectories"] += len(trajectories)
        self.stats["valid_trajectories"] += sum(1 for t in trajectories if len(t) >= self.min_trajectory_length)
        
        return trajectories

    def _filter_outliers(self, points: List[Dict]) -> List[Dict]:
        """
        外れ値を除去する

        Parameters
        ----------
        points : List[Dict]
            検出ポイントのリスト

        Returns
        -------
        filtered_points : List[Dict]
            外れ値を除去したポイントのリスト
        """
        if len(points) < 3:
            return points
        
        filtered_points = []
        
        for i, point in enumerate(points):
            # 前後のポイントとの距離を確認
            is_outlier = False
            
            if 0 < i < len(points) - 1:
                prev_point = points[i-1]
                next_point = points[i+1]
                
                # 前後のポイントが近い場合のみチェック
                if next_point["frame_idx"] - prev_point["frame_idx"] <= 2 * self.max_trajectory_gap:
                    # 速度チェック
                    prev_dist = self._compute_distance(prev_point["position"], point["position"])
                    next_dist = self._compute_distance(point["position"], next_point["position"])
                    
                    time_diff_prev = point["frame_idx"] - prev_point["frame_idx"]
                    time_diff_next = next_point["frame_idx"] - point["frame_idx"]
                    
                    if time_diff_prev > 0 and time_diff_next > 0:
                        speed_prev = prev_dist / time_diff_prev
                        speed_next = next_dist / time_diff_next
                        
                        if speed_prev > self.max_ball_speed or speed_next > self.max_ball_speed:
                            is_outlier = True
                            self.stats["filtered_outliers"] += 1
            
            if not is_outlier:
                filtered_points.append(point)
        
        return filtered_points

    def _connect_points_to_trajectories(self, points: List[Dict]) -> List[List[Dict]]:
        """
        ポイントを軌跡に接続する

        Parameters
        ----------
        points : List[Dict]
            検出ポイントのリスト

        Returns
        -------
        trajectories : List[List[Dict]]
            軌跡のリスト
        """
        trajectories = []
        current_trajectory = []
        
        for i, point in enumerate(points):
            if not current_trajectory:
                current_trajectory.append(point)
                continue
                
            last_point = current_trajectory[-1]
            frame_gap = point["frame_idx"] - last_point["frame_idx"]
            
            # 大きなギャップがあれば新しい軌跡を開始
            if frame_gap > self.max_trajectory_gap:
                if len(current_trajectory) >= self.min_trajectory_length:
                    trajectories.append(current_trajectory)
                current_trajectory = [point]
            else:
                # 速度チェック
                distance = self._compute_distance(last_point["position"], point["position"])
                speed = distance / frame_gap if frame_gap > 0 else 0
                
                if speed > self.max_ball_speed:
                    # 速度が速すぎる場合は新しい軌跡を開始
                    if len(current_trajectory) >= self.min_trajectory_length:
                        trajectories.append(current_trajectory)
                    current_trajectory = [point]
                else:
                    current_trajectory.append(point)
        
        # 最後の軌跡を追加
        if len(current_trajectory) >= self.min_trajectory_length:
            trajectories.append(current_trajectory)
            
        return trajectories

    def _compute_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        2点間の距離を計算

        Parameters
        ----------
        pos1 : Tuple[float, float]
            点1の座標 (x, y)
        pos2 : Tuple[float, float]
            点2の座標 (x, y)

        Returns
        -------
        distance : float
            2点間の距離
        """
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _update_annotations_from_trajectories(
        self, trajectories: List[List[Dict]], frame_idx_map: Dict
    ) -> None:
        """
        軌跡に基づいてアノテーションを更新

        Parameters
        ----------
        trajectories : List[List[Dict]]
            ボールの軌跡リスト
        frame_idx_map : Dict
            フレームIDからインデックスへのマッピング
        """
        # 各軌跡の補間曲線を作成し、欠落フレームを補完
        for trajectory in trajectories:
            if len(trajectory) < 3:  # 補間には少なくとも3点必要
                continue
            
            try:
                # 軌跡を平滑化
                smoothed_trajectory = self._smooth_trajectory(trajectory)
                
                # 補間を実行
                self._interpolate_trajectory(smoothed_trajectory, frame_idx_map)
                
            except Exception as e:
                logger.warning(f"Failed to process trajectory: {e}")
                continue

    def _smooth_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        軌跡を平滑化する

        Parameters
        ----------
        trajectory : List[Dict]
            軌跡ポイントのリスト

        Returns
        -------
        smoothed_trajectory : List[Dict]
            平滑化された軌跡
        """
        if len(trajectory) < self.smoothing_window:
            return trajectory
        
        try:
            # 座標を抽出
            positions_x = [p["position"][0] for p in trajectory]
            positions_y = [p["position"][1] for p in trajectory]
            
            # Savitzky-Golayフィルタで平滑化
            smoothed_x = savgol_filter(positions_x, self.smoothing_window, 2)
            smoothed_y = savgol_filter(positions_y, self.smoothing_window, 2)
            
            # 平滑化された軌跡を作成
            smoothed_trajectory = []
            for i, point in enumerate(trajectory):
                smoothed_point = copy.deepcopy(point)
                smoothed_point["position"] = (smoothed_x[i], smoothed_y[i])
                smoothed_trajectory.append(smoothed_point)
            
            return smoothed_trajectory
            
        except Exception as e:
            logger.warning(f"Failed to smooth trajectory: {e}")
            return trajectory

    def _interpolate_trajectory(self, trajectory: List[Dict], frame_idx_map: Dict) -> None:
        """
        軌跡を補間して欠落フレームを埋める

        Parameters
        ----------
        trajectory : List[Dict]
            軌跡ポイントのリスト
        frame_idx_map : Dict
            フレームIDからインデックスへのマッピング
        """
        frame_indices = [p["frame_idx"] for p in trajectory]
        positions_x = [p["position"][0] for p in trajectory]
        positions_y = [p["position"][1] for p in trajectory]
        
        # 補間関数を作成
        try:
            # 補間方法の選択
            if len(trajectory) >= 4 and self.interpolation_method in ["quadratic", "cubic"]:
                f_x = interp1d(frame_indices, positions_x, kind=self.interpolation_method, bounds_error=False)
                f_y = interp1d(frame_indices, positions_y, kind=self.interpolation_method, bounds_error=False)
            else:
                # ポイントが少ない場合は線形補間
                f_x = interp1d(frame_indices, positions_x, kind='linear', bounds_error=False)
                f_y = interp1d(frame_indices, positions_y, kind='linear', bounds_error=False)
        except ValueError as e:
            logger.warning(f"Failed to create interpolation functions: {e}")
            return
        
        # 補間対象のフレーム範囲
        start_frame = min(frame_indices)
        end_frame = max(frame_indices)
        
        # 逆マッピングでフレームIDを取得
        frame_id_to_idx = {v: k for k, v in frame_idx_map.items()}
        
        # 欠落フレームを補間
        for frame_idx in range(start_frame, end_frame + 1):
            # 既に高信頼度のポイントがある場合はスキップ
            if frame_idx in frame_indices:
                continue
            
            # フレームインデックスの範囲外はスキップ
            if frame_idx not in frame_id_to_idx:
                continue
                
            frame_id = frame_id_to_idx[frame_idx]
            
            # 補間位置を計算
            try:
                interp_x = float(f_x(frame_idx))
                interp_y = float(f_y(frame_idx))
                
                if np.isnan(interp_x) or np.isnan(interp_y):
                    continue
                
                # このフレームの既存のボールアノテーションを取得
                existing_ball_anns = [ann for ann in self.annotations 
                                    if ann.get("image_id") == frame_id and ann.get("category_id") == 1]
                
                if existing_ball_anns:
                    # 既存のアノテーションを更新
                    for ann in existing_ball_anns:
                        x, y, w, h = ann["bbox"]
                        ann["bbox"] = [interp_x - w/2, interp_y - h/2, w, h]
                        ann["score"] = max(ann.get("score", 0.0), 0.5)  # 信頼度を更新
                        ann["is_trajectory_refined"] = True
                else:
                    # 新しいアノテーションを作成
                    self._create_interpolated_annotation(
                        frame_id, frame_idx, interp_x, interp_y, trajectory, frame_indices
                    )
                
                self.stats["interpolated_points"] += 1
                
            except Exception as e:
                logger.debug(f"Failed to interpolate frame {frame_idx}: {e}")
                continue

    def _create_interpolated_annotation(
        self,
        frame_id: int,
        frame_idx: int,
        interp_x: float,
        interp_y: float,
        trajectory: List[Dict],
        frame_indices: List[int],
    ) -> None:
        """
        補間されたアノテーションを作成

        Parameters
        ----------
        frame_id : int
            フレームID
        frame_idx : int
            フレームインデックス
        interp_x : float
            補間されたX座標
        interp_y : float
            補間されたY座標
        trajectory : List[Dict]
            軌跡ポイントのリスト
        frame_indices : List[int]
            軌跡のフレームインデックスリスト
        """
        # 近くの既存アノテーションのサイズを参考に
        nearby_indices = [i for i, idx in enumerate(frame_indices) 
                        if abs(idx - frame_idx) <= self.temporal_window]
        
        if not nearby_indices:
            return
            
        # 近くのボールサイズの平均を計算
        nearby_points = [trajectory[i] for i in nearby_indices]
        avg_width = sum(p.get("width", 30) for p in nearby_points) / len(nearby_points)
        avg_height = sum(p.get("height", 30) for p in nearby_points) / len(nearby_points)
        
        # 新しいアノテーション
        new_ann = {
            "image_id": frame_id,
            "category_id": 1,  # ボール
            "bbox": [interp_x - avg_width/2, interp_y - avg_height/2, avg_width, avg_height],
            "score": 0.5,  # 中程度の信頼度
            "area": avg_width * avg_height,
            "is_trajectory_refined": True,
            "is_interpolated": True
        }
        
        # IDを生成（既存アノテーションの最大ID + 1）
        max_id = max([ann.get("id", 0) for ann in self.annotations], default=0)
        new_ann["id"] = max_id + 1
        
        self.annotations.append(new_ann)

    def get_statistics(self) -> Dict[str, int]:
        """
        軌跡追跡の統計情報を取得

        Returns
        -------
        stats : Dict[str, int]
            統計情報
        """
        return self.stats.copy()


def run_ball_trajectory_tracking(
    input_json: Union[str, Path],
    output_json: Union[str, Path],
    confidence_threshold: float = 0.7,
    temporal_window: int = 9,
    max_trajectory_gap: int = 5,
    min_trajectory_length: int = 7,
    interpolation_method: str = "quadratic",
    smoothing_window: int = 5,
    max_ball_speed: float = 100.0,
) -> Dict[str, int]:
    """
    ボールの軌跡追跡を実行する

    Parameters
    ----------
    input_json : Union[str, Path]
        入力COCOアノテーションファイル
    output_json : Union[str, Path]
        出力COCOアノテーションファイル
    confidence_threshold : float, optional
        高信頼度の閾値（デフォルト: 0.7）
    temporal_window : int, optional
        時間ウィンドウサイズ（デフォルト: 9）
    max_trajectory_gap : int, optional
        補間可能な最大フレームギャップ（デフォルト: 5）
    min_trajectory_length : int, optional
        有効な軌跡の最小長（デフォルト: 7）
    interpolation_method : str, optional
        補間方法（デフォルト: 'quadratic'）
    smoothing_window : int, optional
        平滑化ウィンドウサイズ（デフォルト: 5）
    max_ball_speed : float, optional
        最大ボール速度（デフォルト: 100.0）

    Returns
    -------
    stats : Dict[str, int]
        軌跡追跡の統計情報
    """
    input_json = Path(input_json)
    output_json = Path(output_json)

    try:
        # 入力ファイルを読み込み
        with input_json.open("r", encoding="utf-8") as f:
            coco = json.load(f)
        new_coco = copy.deepcopy(coco)

        annotations = new_coco["annotations"]
        images = new_coco["images"]

        # (game_id, clip_id) でグルーピング
        clips = defaultdict(list)
        for img in images:
            clips[(img.get("game_id", 0), img.get("clip_id", 0))].append(img)

        # トラッカーを初期化
        tracker = BallTrajectoryTracker(
            annotations=annotations,
            confidence_threshold=confidence_threshold,
            temporal_window=temporal_window,
            max_trajectory_gap=max_trajectory_gap,
            min_trajectory_length=min_trajectory_length,
            interpolation_method=interpolation_method,
            smoothing_window=smoothing_window,
            max_ball_speed=max_ball_speed,
        )

        # 各クリップで軌跡追跡を実行
        for (gid, cid), clip_imgs in tqdm(clips.items(), desc="Tracking Ball Trajectories"):
            try:
                tracker.track_ball_in_clip(clip_imgs)
            except Exception as e:
                logger.warning(f"Failed to track clip (game_id={gid}, clip_id={cid}): {e}")
                continue

        # 出力ファイルを保存
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(new_coco, f, indent=2)
        logger.info(f"Saved: {output_json}")
        
        # 統計情報を取得
        stats = tracker.get_statistics()
        logger.info(f"Tracking statistics: {stats}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in ball trajectory tracking: {e}")
        raise


if __name__ == "__main__":
    # 使用例
    input_json = "path/to/input.json"
    output_json = "path/to/output.json"
    
    stats = run_ball_trajectory_tracking(
        input_json=input_json,
        output_json=output_json,
        confidence_threshold=0.7,
        temporal_window=9,
        max_trajectory_gap=5,
        min_trajectory_length=7,
        smoothing_window=5,
        max_ball_speed=100.0,
    )
    
    print(f"Tracking completed with statistics: {stats}") 