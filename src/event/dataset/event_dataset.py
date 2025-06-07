import json
import random
from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------
# ソフトターゲット生成ユーティリティ
# --------------------------------------------------
def create_soft_target(
    event_statuses: List[List[int]],
    sigma: float = 1.5,
    clip: float = 1.0,
) -> torch.Tensor:
    """
    スパースなイベントラベル（[T, 2] one-hot）を
    ガウス状に時間方向へ拡散し、滑らかなターゲットに変換する。

    Args:
        event_statuses: [[hit,bounce], …] 形のリスト（要素は0/1）
        sigma        : ガウス分布の標準偏差（フレーム単位）
        clip         : 最大値をどこでクリップするか（0–1）

    Returns:
        torch.FloatTensor [T, 2]
    """
    T = len(event_statuses)
    soft = np.zeros((T, 2), dtype=np.float32)

    radius = int(3 * sigma)  # 有効範囲 ≈ ±3σ
    for t, (hit, bounce) in enumerate(event_statuses):
        for cls, flag in enumerate((hit, bounce)):
            if flag == 0:
                continue
            for dt in range(-radius, radius + 1):
                tt = t + dt
                if 0 <= tt < T:
                    weight = np.exp(-(dt**2) / (2 * sigma**2))
                    soft[tt, cls] += weight

    # 最大値を 1.0 に正規化してクリップ
    soft = np.clip(soft / soft.max(initial=1.0), 0.0, clip)
    return torch.from_numpy(soft)


# --------------------------------------------------
# EventDataset
# --------------------------------------------------
class EventDataset(Dataset):
    """
    シーケンスから
        * ball_features      [T, 3]
        * player_bbox_feats  [T, max_players, 5]
        * player_pose_feats  [T, max_players, K*3]
        * court_features     [T, C*3]
    を生成し、スムージング済みターゲット (hit, bounce) を返す。
    """

    def __init__(
        self,
        annotation_file: str,
        T: int,
        split: str = "train",
        use_group_split: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        output_type: str = "all",          # {"all", "last"}
        skip_frames_range: Tuple[int, int] = (1, 5),
        smoothing_sigma: float = 1.5,      # ★ 追加：ターゲット平滑化用 σ
    ):
        assert output_type in {"all", "last"}
        self.T = T
        self.split = split
        self.skip_min, self.skip_max = skip_frames_range
        self.output_type = output_type
        self.sigma = smoothing_sigma

        # ---------- アノテーション読み込み ----------
        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.images: Dict[int, dict] = {img["id"]: img for img in data["images"]}
        self.ball_anns_by_image: Dict[int, dict] = {}
        self.player_anns_by_image: Dict[int, list] = defaultdict(list)
        self.court_anns_by_image: Dict[int, dict] = {}
        self.event_status_by_image: Dict[int, int] = {}

        cat_ids = {c["name"]: c["id"] for c in data["categories"]}
        ball_id = cat_ids.get("ball", 1)
        player_id = cat_ids.get("player", 2)
        court_id = cat_ids.get("court", 3)

        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cid = ann["category_id"]
            if cid == ball_id:
                self.ball_anns_by_image[img_id] = ann
                if "event_status" in ann:
                    self.event_status_by_image[img_id] = ann["event_status"]
            elif cid == player_id:
                self.player_anns_by_image[img_id].append(ann)
            elif cid == court_id:
                self.court_anns_by_image[img_id] = ann

        # ---------- クリップ単位のグループ化 ----------
        clip_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for img in data["images"]:
            key = (img["game_id"], img["clip_id"])
            clip_groups[key].append(img["id"])
        self.clip_groups = clip_groups

        keys = list(clip_groups.keys())
        train_k, val_k, test_k = self.group_split_clip(keys, train_ratio, val_ratio, seed)
        target_keys = {"train": train_k, "val": val_k, "test": test_k}[split]

        # ---------- スライディングウィンドウ ----------
        self.windows: List[Tuple[Tuple[int, int], int]] = []
        for key in target_keys:
            ids_sorted = sorted(clip_groups[key])
            if len(ids_sorted) < T:
                continue
            for s in range(0, len(ids_sorted) - T + 1):
                self.windows.append((key, s))

        print(f"[{split.upper()}] clips: {len(target_keys)}, windows: {len(self.windows)}")

    # --------------------------------------------------

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        clip_key, start = self.windows[idx]
        ids_sorted = sorted(self.clip_groups[clip_key])
        L = len(ids_sorted)

        # フレーム skip
        if self.T > 1 and self.split == "train" and (self.skip_min, self.skip_max) != (1, 1):
            max_allowed = (L - 1 - start) // (self.T - 1)
            skip = random.randint(self.skip_min, min(self.skip_max, max_allowed) or 1)
            frame_ids = [ids_sorted[start + i * skip] for i in range(self.T)]
        else:
            frame_ids = ids_sorted[start:start + self.T]

        # ---------- 各フレームの特徴量抽出 ----------
        ball_feats, player_bbox_lst, player_pose_lst, court_feats, event_labels = \
            self._extract_sequence_features(frame_ids)

        # stack / padding
        ball_tensor = torch.stack(ball_feats)                       # [T, 3]
        court_tensor = torch.stack(court_feats)                     # [T, C*3]
        player_bbox_tensor, player_pose_tensor = self._pad_player_features(player_bbox_lst, player_pose_lst)

        # ---------- スムージングターゲット ----------
        target_tensor = create_soft_target(event_labels, sigma=self.sigma)   # [T, 2]
        if self.output_type == "last":
            target_tensor = target_tensor[-1]                                  # [2]

        last_img_info = self.images[frame_ids[-1]].copy()
        last_img_info["id"] = frame_ids[-1]

        return ball_tensor, player_bbox_tensor, player_pose_tensor, court_tensor, target_tensor, last_img_info

    # --------------------------------------------------
    # 内部ユーティリティ
    # --------------------------------------------------
    def _extract_sequence_features(self, frame_ids):
        ball_feats, player_bbox_lst, player_pose_lst, court_feats, event_labels = [], [], [], [], []
        max_players = 0

        # 1st pass: 最大プレイヤー数
        for img_id in frame_ids:
            max_players = max(max_players, len(self.player_anns_by_image.get(img_id, [])))

        for img_id in frame_ids:
            img = self.images[img_id]
            w, h = img["width"], img["height"]

            # --- Ball ---
            ball_ann = self.ball_anns_by_image.get(img_id)
            if ball_ann:
                x, y, v = ball_ann["keypoints"][:3]
                ball_score = ball_ann.get("keypoints_scores", [1.0])[0] if ball_ann.get("keypoints_scores") else (1.0 if v == 2 else 0.5 if v == 1 else 0.0)
                ball_feats.append(torch.tensor([x / w, y / h, ball_score], dtype=torch.float32))
                evt = self.event_status_by_image.get(img_id, 0)
                event_labels.append([1, 0] if evt == 1 else [0, 1] if evt == 2 else [0, 0])
            else:
                ball_feats.append(torch.zeros(3))
                event_labels.append([0, 0])

            # --- Players ---
            bbox_feats, pose_feats = [], []
            for p in self.player_anns_by_image.get(img_id, []):
                x1, y1, bw, bh = p["bbox"]
                x2, y2 = x1 + bw, y1 + bh
                bbox_feats.append(torch.tensor([x1 / w, y1 / h, x2 / w, y2 / h, p.get("score", 1.0)], dtype=torch.float32))

                kps, kps_s = p["keypoints"], p.get("keypoints_scores", [])
                kp_out = []
                for i in range(0, len(kps), 3):
                    kx, ky, kv = kps[i:i + 3]
                    if kps_s and i // 3 < len(kps_s):
                        kp_score = kps_s[i // 3]
                        kv = 2 if kp_score >= 0.5 else 1 if kp_score > 0.01 else 0
                    kp_out.extend([kx / w, ky / h, kv])
                pose_feats.append(torch.tensor(kp_out, dtype=torch.float32))

            player_bbox_lst.append(bbox_feats)
            player_pose_lst.append(pose_feats)

            # --- Court ---
            court_ann = self.court_anns_by_image.get(img_id)
            if court_ann:
                kps, kps_s = court_ann["keypoints"], court_ann.get("keypoints_scores", [])
                court_vec = []
                for i in range(0, len(kps), 3):
                    cx, cy, cv = kps[i:i + 3]
                    if kps_s and i // 3 < len(kps_s):
                        cscore = kps_s[i // 3]
                        cv = 2 if cscore >= 0.6 else 1 if cscore > 0.01 else 0
                    court_vec.extend([cx / w, cy / h, cv])
                court_feats.append(torch.tensor(court_vec, dtype=torch.float32))
            else:
                court_feats.append(torch.zeros(45))  # 15kp*3 の例

        return ball_feats, player_bbox_lst, player_pose_lst, court_feats, event_labels

    def _pad_player_features(self, bbox_lst, pose_lst):
        max_players = max(len(b) for b in bbox_lst)
        bbox_dim = 5
        pose_dim = pose_lst[0][0].shape[0] if pose_lst and pose_lst[0] else 0

        padded_bbox, padded_pose = [], []
        for bbox_f, pose_f in zip(bbox_lst, pose_lst):
            if bbox_f:
                bbox_t = torch.stack(bbox_f)
                pose_t = torch.stack(pose_f)
            else:
                bbox_t = torch.zeros((0, bbox_dim))
                pose_t = torch.zeros((0, pose_dim))

            if len(bbox_f) < max_players:
                pad_n = max_players - len(bbox_f)
                bbox_t = torch.cat([bbox_t, torch.zeros((pad_n, bbox_dim))], dim=0)
                pose_t = torch.cat([pose_t, torch.zeros((pad_n, pose_dim))], dim=0)

            padded_bbox.append(bbox_t)
            padded_pose.append(pose_t)

        return torch.stack(padded_bbox), torch.stack(padded_pose)

    # --------------------------------------------------
    @staticmethod
    def group_split_clip(keys, train_ratio, val_ratio, seed):
        random.seed(seed)
        random.shuffle(keys)
        n = len(keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:]
