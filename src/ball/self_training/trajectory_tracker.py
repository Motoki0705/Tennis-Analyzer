import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm


class BallTrajectoryTracker:
    """
    テニスボールの軌跡を追跡し、自己学習データを生成するクラス。
    高信頼度の検出結果から時間的に一貫したボールの軌跡を推定し、
    低信頼度の検出や欠落フレームを補完します。
    """

    def __init__(
        self,
        annotations: List[Dict],
        confidence_threshold: float = 0.7,
        temporal_window: int = 9,
        max_trajectory_gap: int = 5,
        min_trajectory_length: int = 7,
        interpolation_method: str = "quadratic",
    ):
        """
        初期化

        Parameters
        ----------
        annotations : アノテーションのリスト（更新される）
        confidence_threshold : 高信頼度と見なす信頼度の閾値
        temporal_window : 軌跡推定に使用する時間ウィンドウのサイズ
        max_trajectory_gap : 補間可能な最大フレームギャップ
        min_trajectory_length : 有効な軌跡とみなす最小長
        interpolation_method : 補間方法（'linear', 'quadratic', 'cubic'）
        """
        self.annotations = annotations
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.max_trajectory_gap = max_trajectory_gap
        self.min_trajectory_length = min_trajectory_length
        self.interpolation_method = interpolation_method

    def track_ball_in_clip(self, clip_imgs: List[Dict]) -> None:
        """
        クリップ内でボールを追跡し、アノテーションを更新する

        Parameters
        ----------
        clip_imgs : クリップの画像情報リスト
        """
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

    def _build_trajectories(
        self, clip_imgs: List[Dict], frames_to_ball_anns: Dict, frame_idx_map: Dict
    ) -> List[Dict]:
        """
        高信頼度の検出からボールの軌跡を構築

        Parameters
        ----------
        clip_imgs : クリップの画像情報リスト
        frames_to_ball_anns : フレームIDからボールアノテーションへのマッピング
        frame_idx_map : フレームIDからインデックスへのマッピング

        Returns
        -------
        trajectories : ボールの軌跡リスト
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
                        "annotation": ann
                    })
        
        # 高信頼度ポイントが不足している場合は早期リターン
        if len(high_conf_points) < self.min_trajectory_length:
            return []
        
        # フレームインデックスでソート
        high_conf_points.sort(key=lambda p: p["frame_idx"])
        
        # 軌跡を構築
        trajectories = []
        current_trajectory = []
        
        for i, point in enumerate(high_conf_points):
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
                current_trajectory.append(point)
        
        # 最後の軌跡を追加
        if len(current_trajectory) >= self.min_trajectory_length:
            trajectories.append(current_trajectory)
            
        return trajectories

    def _update_annotations_from_trajectories(
        self, trajectories: List[List[Dict]], frame_idx_map: Dict
    ) -> None:
        """
        軌跡に基づいてアノテーションを更新

        Parameters
        ----------
        trajectories : ボールの軌跡リスト
        frame_idx_map : フレームIDからインデックスへのマッピング
        """
        # 各軌跡の補間曲線を作成し、欠落フレームを補完
        for trajectory in trajectories:
            if len(trajectory) < 3:  # 補間には少なくとも3点必要
                continue
                
            frame_indices = [p["frame_idx"] for p in trajectory]
            positions_x = [p["position"][0] for p in trajectory]
            positions_y = [p["position"][1] for p in trajectory]
            
            # 補間関数を作成
            try:
                f_x = interp1d(frame_indices, positions_x, kind=self.interpolation_method, bounds_error=False)
                f_y = interp1d(frame_indices, positions_y, kind=self.interpolation_method, bounds_error=False)
            except ValueError:
                # 補間に失敗した場合は線形補間を試みる
                f_x = interp1d(frame_indices, positions_x, kind='linear', bounds_error=False)
                f_y = interp1d(frame_indices, positions_y, kind='linear', bounds_error=False)
            
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
                    # 新しいアノテーションを作成（近くの既存アノテーションのサイズを参考に）
                    nearby_indices = [i for i, idx in enumerate(frame_indices) 
                                    if abs(idx - frame_idx) <= self.temporal_window]
                    
                    if not nearby_indices:
                        continue
                        
                    # 近くのボールサイズの平均を計算
                    nearby_anns = [trajectory[i]["annotation"] for i in nearby_indices]
                    avg_width = sum(ann["bbox"][2] for ann in nearby_anns) / len(nearby_anns)
                    avg_height = sum(ann["bbox"][3] for ann in nearby_anns) / len(nearby_anns)
                    
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


def run_ball_trajectory_tracking(
    input_json: Union[str, Path],
    output_json: Union[str, Path],
    confidence_threshold: float = 0.7,
    temporal_window: int = 9,
    max_trajectory_gap: int = 5,
    min_trajectory_length: int = 7,
    interpolation_method: str = "quadratic",
) -> None:
    """
    ボールの軌跡追跡を実行する

    Parameters
    ----------
    input_json : 入力COCOアノテーションファイル
    output_json : 出力COCOアノテーションファイル
    confidence_threshold : 高信頼度の閾値
    temporal_window : 時間ウィンドウサイズ
    max_trajectory_gap : 補間可能な最大フレームギャップ
    min_trajectory_length : 有効な軌跡の最小長
    interpolation_method : 補間方法
    """
    input_json = Path(input_json)
    output_json = Path(output_json)

    with input_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    new_coco = copy.deepcopy(coco)

    annotations = new_coco["annotations"]
    images = new_coco["images"]

    # (game_id, clip_id) でグルーピング
    clips = defaultdict(list)
    for img in images:
        clips[(img.get("game_id", 0), img.get("clip_id", 0))].append(img)

    tracker = BallTrajectoryTracker(
        annotations=annotations,
        confidence_threshold=confidence_threshold,
        temporal_window=temporal_window,
        max_trajectory_gap=max_trajectory_gap,
        min_trajectory_length=min_trajectory_length,
        interpolation_method=interpolation_method,
    )

    for (gid, cid), clip_imgs in tqdm(clips.items(), desc="Tracking Ball Trajectories"):
        tracker.track_ball_in_clip(clip_imgs)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    # 使用例
    input_json = "path/to/input.json"
    output_json = "path/to/output.json"
    
    run_ball_trajectory_tracking(
        input_json=input_json,
        output_json=output_json,
        confidence_threshold=0.7,
        temporal_window=9,
        max_trajectory_gap=5,
        min_trajectory_length=7,
    ) 