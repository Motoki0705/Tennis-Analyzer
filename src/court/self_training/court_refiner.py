import copy
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm


class CourtLineRefiner:
    """
    コート検出結果を自己学習により洗練するクラス。
    同じビデオクリップ内でのコートラインの一貫性を利用して、
    低品質な検出結果を高品質な検出結果に基づいて修正します。
    """

    def __init__(
        self,
        annotations: List[Dict],
        confidence_threshold: float = 0.7,
        similarity_threshold: float = 0.8,
        max_keypoint_shift: float = 20.0,
        min_high_quality_frames: int = 3,
    ):
        """
        初期化

        Parameters
        ----------
        annotations : アノテーションのリスト（更新される）
        confidence_threshold : 高品質と見なす信頼度の閾値
        similarity_threshold : キーポイント類似度の閾値
        max_keypoint_shift : キーポイント移動の最大許容距離
        min_high_quality_frames : 必要な高品質フレームの最小数
        """
        self.annotations = annotations
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.max_keypoint_shift = max_keypoint_shift
        self.min_high_quality_frames = min_high_quality_frames

    def refine_court_in_clip(self, clip_imgs: List[Dict]) -> None:
        """
        クリップ内のコート検出結果を洗練する

        Parameters
        ----------
        clip_imgs : クリップの画像情報リスト
        """
        # 時系列順にソート
        clip_imgs.sort(key=lambda img: img["file_name"])
        frame_idx_map = {img["id"]: idx for idx, img in enumerate(clip_imgs)}

        # 各フレームのコートアノテーションを抽出
        frames_to_court_anns = defaultdict(list)
        for ann in self.annotations:
            if ann.get("image_id") in frame_idx_map and ann.get("category_id") == 4:  # コートカテゴリ
                frames_to_court_anns[ann["image_id"]].append(ann)

        # 高品質なコート検出を特定
        high_quality_courts = self._identify_high_quality_courts(clip_imgs, frames_to_court_anns)
        
        # 高品質なコート検出が不足している場合は処理をスキップ
        if len(high_quality_courts) < self.min_high_quality_frames:
            return
            
        # 代表的なコートモデルを構築
        reference_court = self._build_reference_court(high_quality_courts)
        
        # すべてのフレームのコート検出を改善
        self._refine_all_courts(clip_imgs, frames_to_court_anns, reference_court, frame_idx_map)

    def _identify_high_quality_courts(
        self, clip_imgs: List[Dict], frames_to_court_anns: Dict
    ) -> List[Dict]:
        """
        高品質なコート検出を特定する

        Parameters
        ----------
        clip_imgs : クリップの画像情報リスト
        frames_to_court_anns : フレームIDからコートアノテーションへのマッピング

        Returns
        -------
        high_quality_courts : 高品質なコートアノテーションのリスト
        """
        high_quality_courts = []
        
        for img in clip_imgs:
            frame_id = img["id"]
            court_anns = frames_to_court_anns.get(frame_id, [])
            
            for ann in court_anns:
                # 信頼度チェック
                confidence = ann.get("score", 0.0)
                
                # キーポイントの可視性チェック
                keypoints = ann.get("keypoints", [])
                if not keypoints:
                    continue
                    
                # キーポイントは [x, y, visibility] の3つ組で構成
                # 可視性スコアの平均を計算
                visibility_scores = [keypoints[i+2] for i in range(0, len(keypoints), 3)]
                avg_visibility = sum(visibility_scores) / len(visibility_scores) if visibility_scores else 0
                
                # 高品質判定
                if confidence >= self.confidence_threshold and avg_visibility >= self.similarity_threshold:
                    high_quality_courts.append(ann)
        
        return high_quality_courts

    def _build_reference_court(self, high_quality_courts: List[Dict]) -> Dict:
        """
        代表的なコートモデルを構築する

        Parameters
        ----------
        high_quality_courts : 高品質なコートアノテーションのリスト

        Returns
        -------
        reference_court : 代表的なコートモデル
        """
        # キーポイント数を確認
        keypoints_sample = high_quality_courts[0].get("keypoints", [])
        num_keypoints = len(keypoints_sample) // 3
        
        # 各キーポイントの平均位置を計算
        reference_keypoints = []
        for kp_idx in range(num_keypoints):
            x_values = []
            y_values = []
            v_values = []
            
            for court in high_quality_courts:
                keypoints = court.get("keypoints", [])
                x = keypoints[kp_idx * 3]
                y = keypoints[kp_idx * 3 + 1]
                v = keypoints[kp_idx * 3 + 2]
                
                if v > 0:  # 可視のキーポイントのみ
                    x_values.append(x)
                    y_values.append(y)
                    v_values.append(v)
            
            if x_values and y_values:
                avg_x = sum(x_values) / len(x_values)
                avg_y = sum(y_values) / len(y_values)
                avg_v = sum(v_values) / len(v_values)
                reference_keypoints.extend([avg_x, avg_y, avg_v])
            else:
                # 可視データがない場合は0を入れる
                reference_keypoints.extend([0, 0, 0])
        
        # 参照コートモデル
        reference_court = {
            "keypoints": reference_keypoints,
            "num_keypoints": num_keypoints,
            "confidence": 1.0
        }
        
        return reference_court

    def _refine_all_courts(
        self, 
        clip_imgs: List[Dict], 
        frames_to_court_anns: Dict,
        reference_court: Dict,
        frame_idx_map: Dict
    ) -> None:
        """
        すべてのフレームのコート検出を改善する

        Parameters
        ----------
        clip_imgs : クリップの画像情報リスト
        frames_to_court_anns : フレームIDからコートアノテーションへのマッピング
        reference_court : 代表的なコートモデル
        frame_idx_map : フレームIDからインデックスへのマッピング
        """
        ref_keypoints = reference_court["keypoints"]
        num_keypoints = reference_court["num_keypoints"]
        
        for img in clip_imgs:
            frame_id = img["id"]
            court_anns = frames_to_court_anns.get(frame_id, [])
            
            # このフレームにアノテーションがない場合は新規作成
            if not court_anns:
                # 新しいコートアノテーション
                new_ann = {
                    "image_id": frame_id,
                    "category_id": 4,  # コート
                    "keypoints": ref_keypoints.copy(),
                    "score": 0.6,  # 中程度の信頼度
                    "is_refined": True,
                    "is_generated": True
                }
                
                # IDを生成（既存アノテーションの最大ID + 1）
                max_id = max([ann.get("id", 0) for ann in self.annotations], default=0)
                new_ann["id"] = max_id + 1
                
                self.annotations.append(new_ann)
                continue
            
            # 既存のアノテーションがある場合は改善
            for ann in court_anns:
                keypoints = ann.get("keypoints", [])
                
                # キーポイントがない場合はリファレンスをそのまま使用
                if not keypoints:
                    ann["keypoints"] = ref_keypoints.copy()
                    ann["score"] = max(ann.get("score", 0.0), 0.6)
                    ann["is_refined"] = True
                    continue
                
                # 既存のキーポイントとリファレンスを比較し、必要に応じて修正
                refined_keypoints = []
                for kp_idx in range(num_keypoints):
                    x = keypoints[kp_idx * 3]
                    y = keypoints[kp_idx * 3 + 1]
                    v = keypoints[kp_idx * 3 + 2]
                    
                    ref_x = ref_keypoints[kp_idx * 3]
                    ref_y = ref_keypoints[kp_idx * 3 + 1]
                    ref_v = ref_keypoints[kp_idx * 3 + 2]
                    
                    # 可視性が低いまたは距離が大きい場合は参照を使用
                    if v < 0.3 or (ref_v > 0.5 and self._distance((x, y), (ref_x, ref_y)) > self.max_keypoint_shift):
                        refined_keypoints.extend([ref_x, ref_y, ref_v])
                    else:
                        # 元のキーポイントを保持
                        refined_keypoints.extend([x, y, max(v, 0.3)])  # 可視性を少し上げる
                
                ann["keypoints"] = refined_keypoints
                ann["score"] = max(ann.get("score", 0.0), 0.6)
                ann["is_refined"] = True
    
    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        2点間のユークリッド距離を計算する

        Parameters
        ----------
        p1 : 点1の(x, y)座標
        p2 : 点2の(x, y)座標

        Returns
        -------
        distance : 距離
        """
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def run_court_line_refinement(
    input_json: Union[str, Path],
    output_json: Union[str, Path],
    confidence_threshold: float = 0.7,
    similarity_threshold: float = 0.8,
    max_keypoint_shift: float = 20.0,
    min_high_quality_frames: int = 3,
) -> None:
    """
    コートライン洗練を実行する

    Parameters
    ----------
    input_json : 入力COCOアノテーションファイル
    output_json : 出力COCOアノテーションファイル
    confidence_threshold : 高品質の閾値
    similarity_threshold : キーポイント類似度の閾値
    max_keypoint_shift : キーポイント移動の最大許容距離
    min_high_quality_frames : 必要な高品質フレームの最小数
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

    refiner = CourtLineRefiner(
        annotations=annotations,
        confidence_threshold=confidence_threshold,
        similarity_threshold=similarity_threshold,
        max_keypoint_shift=max_keypoint_shift,
        min_high_quality_frames=min_high_quality_frames,
    )

    for (gid, cid), clip_imgs in tqdm(clips.items(), desc="Refining Court Lines"):
        refiner.refine_court_in_clip(clip_imgs)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    # 使用例
    input_json = "path/to/input.json"
    output_json = "path/to/output.json"
    
    run_court_line_refinement(
        input_json=input_json,
        output_json=output_json,
        confidence_threshold=0.7,
        similarity_threshold=0.8,
        max_keypoint_shift=20.0,
        min_high_quality_frames=3,
    ) 