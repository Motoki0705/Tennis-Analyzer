import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# ------------------- Utility ------------------- #
def bbox_center(bbox: List[float]) -> np.ndarray:
    """[x, y, w, h] → (cx, cy)"""
    x, y, w, h = bbox
    return np.array([x + w / 2, y + h / 2], dtype=np.float32)


def evaluate_track_strength(track_anns: List[Dict]) -> float:
    """
    track の優位性スコア:
        (#出現フレーム) × (平均 score)
    """
    frame_ids = {ann["image_id"] for ann in track_anns}
    num_frames = len(frame_ids)
    avg_score = float(np.mean([ann.get("score", 1.0) for ann in track_anns]))
    return num_frames * avg_score


# ------------------- Tracker ------------------- #
class ClipwiseGlobalPlayerTracker:
    def __init__(
        self,
        annotations: List[Dict],
        eps: float = 50.0,
        min_samples: int = 3,
        player_threshold: int = 1,
        top_k: int = 2,
        decay_rate: float = 0.7,
        min_eps_ratio: float = 0.2,
        max_refine: int = 5,
    ):
        """
        Parameters
        ----------
        annotations : COCO annotations (dict list)  — 書き換えられる
        eps          : 初期 DBSCAN eps
        min_samples  : DBSCAN min_samples
        player_threshold : クラスタを player 候補にするための最小 player bbox 数
        top_k        : 最終的に player とするクラスタ数 (実際のプレーヤー人数)
        decay_rate   : 衝突クラスタ再分割時の eps 縮小率
        min_eps_ratio: eps の最小比率 (eps * ratio が下限)
        max_refine   : eps 縮小の最大反復回数
        """
        self.annotations = annotations
        self.eps = eps
        self.min_samples = min_samples
        self.player_threshold = player_threshold
        self.top_k = top_k
        self.decay_rate = decay_rate
        self.min_eps = eps * min_eps_ratio
        self.max_refine = max_refine

    # --------- Core: process one clip --------- #
    def track_clip(self, clip_imgs: List[Dict]) -> None:
        # 時系列順ソート
        clip_imgs.sort(key=lambda img: img["file_name"])
        frame_idx_map = {img["id"]: idx for idx, img in enumerate(clip_imgs)}

        # 特徴行列と参照
        feats, refs = [], []
        for img in clip_imgs:
            fid = img["id"]
            fidx = frame_idx_map[fid]
            for ann in self.annotations:
                if ann["image_id"] == fid and ann["category_id"] in (2, 3):
                    c = bbox_center(ann["bbox"])
                    feats.append([c[0], c[1], fidx])
                    refs.append(ann)

        if not feats:
            return

        feats = np.asarray(feats, dtype=np.float32)
        labels = self._dbscan(feats, self.eps)

        # 初期クラスタ辞書
        raw_clusters: Dict[int, List[Dict]] = defaultdict(list)
        for lbl, ann in zip(labels, refs, strict=False):
            if lbl == -1:
                continue
            ann["_frame_idx"] = frame_idx_map[ann["image_id"]]
            raw_clusters[lbl].append(ann)

        # 衝突クラスタを再分割
        cluster_to_anns: Dict[int, List[Dict]] = {}
        tmp_id = 0  # 新しいサブクラスタIDを連番生成
        for lbl, anns in raw_clusters.items():
            if self._has_frame_collision(anns):
                sub = self._split_invalid_cluster(anns, self.eps)
                for _, sub_anns in sub.items():
                    cluster_to_anns[tmp_id] = sub_anns
                    tmp_id += 1
            else:
                cluster_to_anns[tmp_id] = anns
                tmp_id += 1

        # ----------------- player 候補 ----------------- #
        candidate = [
            cid
            for cid, anns in cluster_to_anns.items()
            if sum(a["category_id"] == 2 for a in anns) >= self.player_threshold
        ]

        # 3つ以上ならスコアで上位 top_k
        if len(candidate) > self.top_k:
            scores = {
                cid: evaluate_track_strength(cluster_to_anns[cid]) for cid in candidate
            }
            selected = sorted(scores, key=lambda k: -scores[k])[: self.top_k]
        else:
            selected = candidate

        # -------------- アノテーション書き込み -------------- #
        for cid, anns in cluster_to_anns.items():
            is_player = cid in selected
            for ann in anns:
                ann["player_id"] = int(cid)
                ann["is_track_verified"] = bool(is_player)
                ann["category_id"] = 2 if is_player else 3

        # 補助フィールド削除
        for ann in refs:
            ann.pop("_frame_idx", None)

    # --------- Helper: DBSCAN --------- #
    def _dbscan(self, X: np.ndarray, eps: float) -> np.ndarray:
        return DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(X)

    # --------- Helper: frame collision --------- #
    @staticmethod
    def _has_frame_collision(track_anns: List[Dict]) -> bool:
        frames = [a["_frame_idx"] for a in track_anns]
        return len(frames) != len(set(frames))

    # --------- refine invalid cluster --------- #
    def _split_invalid_cluster(
        self, anns: List[Dict], eps_init: float
    ) -> Dict[int, List[Dict]]:
        eps = eps_init * self.decay_rate
        refine_cnt = 0
        pool = anns
        clusters: Dict[int, List[Dict]] = {}

        while refine_cnt < self.max_refine and eps >= self.min_eps:
            feats = np.array(
                [bbox_center(a["bbox"]).tolist() + [a["_frame_idx"]] for a in pool],
                dtype=np.float32,
            )
            labels = self._dbscan(feats, eps)
            tmp = defaultdict(list)
            for lbl, ann in zip(labels, pool, strict=False):
                tmp[lbl].append(ann)

            # valid / invalid 判定
            invalid_tmp = {
                lbl: t
                for lbl, t in tmp.items()
                if lbl != -1 and self._has_frame_collision(t)
            }
            clusters.update(
                {
                    lbl: t
                    for lbl, t in tmp.items()
                    if lbl == -1 or lbl not in invalid_tmp
                }
            )

            pool = [ann for sub in invalid_tmp.values() for ann in sub]
            if not pool:
                break

            eps *= self.decay_rate
            refine_cnt += 1

        # 残った pool をフレーム単位で分離
        if pool:
            by_frame = defaultdict(list)
            for ann in pool:
                by_frame[ann["_frame_idx"]].append(ann)
            for sub in by_frame.values():
                clusters[len(clusters)] = sub
        return clusters


# ------------------- top-level ------------------- #
def run_tracking(
    input_json: str | Path,
    output_json: str | Path,
    eps: float = 50.0,
    min_samples: int = 3,
    player_threshold: int = 1,
    top_k: int = 2,
    decay_rate: float = 0.7,
    min_eps_ratio: float = 0.2,
    max_refine: int = 5,
) -> None:
    input_json = Path(input_json)
    output_json = Path(output_json)

    with input_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    new_coco = copy.deepcopy(coco)

    annotations = new_coco["annotations"]
    images = new_coco["images"]

    # (game_id, clip_id) でグルーピング
    clips: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for img in images:
        clips[(img["game_id"], img["clip_id"])].append(img)

    tracker = ClipwiseGlobalPlayerTracker(
        annotations=annotations,
        eps=eps,
        min_samples=min_samples,
        player_threshold=player_threshold,
        top_k=top_k,
        decay_rate=decay_rate,
        min_eps_ratio=min_eps_ratio,
        max_refine=max_refine,
    )

    for (gid, cid), clip_imgs in tqdm(clips.items(), desc="Tracking Clips"):
        tracker.track_clip(clip_imgs)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    input_json = r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\annotation_jsons\coco_annotations_final.json"
    output_json = r"C:\Users\kamim\code\Tennis-Analyzer\BallDetection\data\annotation_jsons\coco_annotations_globally_tracked.json"

    run_tracking(
        input_json=input_json,
        output_json=output_json,
        eps=50,  # DBSCANのクラスタリング半径（調整可能）
        min_samples=3,  # クラスタリングに必要な最小bbox数（調整可能）
        player_threshold=1,  # プレーヤーとするために必要な最低player判定数（調整可能）
        top_k=2,
    )
