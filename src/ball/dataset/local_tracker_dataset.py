import json
import os
import random
from typing import List, Tuple, Dict, Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class LocalTrackerDataset(Dataset):
    """
    小モデル（Local Tracker）学習用のデータセットクラス。
    アノテーションデータから、ボール周辺の画像パッチ（RoI）と、
    対応する目標ヒートマップ、存在スコアを生成する。
    """
    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        input_size: List[int],
        heatmap_size: List[int],
        transform: A.Compose,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        negative_ratio: float = 0.5,
        heatmap_sigma: float = 1.5,
    ):
        """
        Args:
            annotation_file (str): COCO-like形式のアノテーションファイルへのパス。
            image_root (str): フレーム画像が格納されているルートディレクトリ。
            input_size (List[int]): モデルに入力するRoIのサイズ [width, height]。
            heatmap_size (List[int]): 出力ヒートマップのサイズ [width, height]。
            transform (A.Compose): Albumentationsによるデータ拡張パイプライン。
            split (str): "train", "val", "test" のいずれか。
            train_ratio (float): データセット全体に占める訓練データの割合。
            val_ratio (float): データセット全体に占める検証データの割合。
            seed (int): データ分割の再現性を保つための乱数シード。
            negative_ratio (float): __getitem__でNegativeサンプルを返す確率。
            heatmap_sigma (float): 目標ガウシアンヒートマップの標準偏差。
        """
        super().__init__()
        # 1. パラメータの保存
        self.image_root = image_root
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.transform = transform
        self.negative_ratio = negative_ratio
        self.heatmap_sigma = heatmap_sigma

        # 2. アノテーションファイルの読み込みとインデックス作成
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images_by_id = {img['id']: img for img in data['images']}
        annotations_by_image = {anno['image_id']: anno for anno in data['annotations']}

        # 3. クリップ単位でのデータ分割
        # 時間的リークを防ぐため、フレーム単位ではなくクリップ単位で分割する
        clips = defaultdict(list)
        for img_id, img_info in self.images_by_id.items():
            clip_id = img_info.get('clip_id')
            if clip_id is not None:
                # visibility > 0 (ボールが存在する)アノテーションのみをPositive候補として追加
                if img_id in annotations_by_image and annotations_by_image[img_id].get('visibility', 0) > 0:
                    clips[clip_id].append(annotations_by_image[img_id])
        
        clip_ids = sorted(list(clips.keys()))
        random.Random(seed).shuffle(clip_ids)
        
        n_clips = len(clip_ids)
        n_train = int(n_clips * train_ratio)
        n_val = int(n_clips * val_ratio)
        
        split_map = {
            "train": clip_ids[:n_train],
            "val": clip_ids[n_train:n_train + n_val],
            "test": clip_ids[n_train + n_val:],
        }
        
        target_clip_ids = set(split_map[split])

        # 4. 対象スプリットのサンプルリストを作成
        self.positive_annotations = []
        for clip_id in target_clip_ids:
            self.positive_annotations.extend(clips[clip_id])

        # Negativeサンプリング用に、対象スプリットの全画像リストを保持
        self.all_image_ids_in_split = [
            img_id for img_id, img_info in self.images_by_id.items() 
            if img_info.get('clip_id') in target_clip_ids
        ]

        if not self.positive_annotations:
            raise ValueError(f"No positive samples found for the '{split}' split. Check your data and ratios.")

        print(f"[{split.upper()}] split loaded. Positive samples: {len(self.positive_annotations)}. Total images for sampling: {len(self.all_image_ids_in_split)}")

    def __len__(self) -> int:
        # 1エポックあたりの総サンプル数を定義 (PositiveとNegativeが指定比率で含まれるように)
        return int(len(self.positive_annotations) / (1.0 - self.negative_ratio))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < self.negative_ratio:
            return self._generate_negative_sample()
        else:
            return self._generate_positive_sample()

    def _generate_positive_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ボールが存在するRoIと対応するラベルを生成する"""
        # 1. ランダムにPositiveアノテーションを選択
        anno = random.choice(self.positive_annotations)
        image_info = self.images_by_id[anno['image_id']]
        image_path = os.path.join(self.image_root, image_info['file_name'])
        image = self._load_image(image_path)

        # 2. RoIの中心点をキーポイント座標から少しずらし、頑健性を向上
        kpt = anno['keypoints']
        center_xy = np.array([kpt[0], kpt[1]], dtype=np.float32)
        offset = (np.random.rand(2) - 0.5) * np.array(self.input_size) * 0.5 # RoIサイズの最大±25%ずらす
        center_xy += offset

        # 3. RoI画像を切り出し、データ拡張を適用
        roi, _ = self._crop_roi(image, center_xy, self.input_size)
        transformed = self.transform(image=roi)
        roi_tensor = transformed['image']

        # 4. 目標ヒートマップを生成
        original_kpt_in_roi = kpt[:2] - (center_xy - np.array(self.input_size) / 2)
        heatmap_kpt = original_kpt_in_roi * (np.array(self.heatmap_size) / np.array(self.input_size))
        heatmap = self._generate_gaussian_heatmap(self.heatmap_size, heatmap_kpt, self.heatmap_sigma)
        heatmap = torch.from_numpy(heatmap).float().unsqueeze(0) # (1, H, W) チャンネル次元を追加

        # 5. 存在スコアを設定
        existence_score = torch.tensor([1.0], dtype=torch.float32)
        
        return roi_tensor, heatmap, existence_score

    def _generate_negative_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ボールが存在しないRoIと対応するラベルを生成する"""
        # 1. ランダムに画像を選択
        image_id = random.choice(self.all_image_ids_in_split)
        image_info = self.images_by_id[image_id]
        image_path = os.path.join(self.image_root, image_info['file_name'])
        image = self._load_image(image_path)
        h, w = image.shape[:2]

        # 2. ランダムな位置からRoIを切り出し
        center_xy = (random.uniform(0, w), random.uniform(0, h))
        roi, _ = self._crop_roi(image, center_xy, self.input_size)
        transformed = self.transform(image=roi)
        roi_tensor = transformed['image']

        # 3. 目標ヒートマップと存在スコアを生成
        heatmap = torch.zeros((1, *self.heatmap_size), dtype=torch.float32)
        existence_score = torch.tensor([0.0], dtype=torch.float32)

        return roi_tensor, heatmap, existence_score

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _crop_roi(image: np.ndarray, center_xy: Tuple[float, float], size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, int]]:
        """画像からRoIを切り出す。境界外はパディングする。"""
        img_h, img_w = image.shape[:2]
        roi_w, roi_h = size
        center_x, center_y = center_xy

        x1 = int(round(center_x - roi_w / 2))
        y1 = int(round(center_y - roi_h / 2))
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        # パディング量を計算
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)
        
        # 画像から有効な領域を切り出し
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(img_w, x2)
        crop_y2 = min(img_h, y2)
        
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # パディングを実行
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            padded_image = cv2.copyMakeBorder(
                cropped_image, pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            padded_image = cropped_image

        padding_info = {'top': pad_top, 'bottom': pad_bottom, 'left': pad_left, 'right': pad_right}
        return padded_image, padding_info

    @staticmethod
    def _generate_gaussian_heatmap(size: Tuple[int, int], center_xy: Tuple[float, float], sigma: float) -> np.ndarray:
        """指定された中心に2Dガウシアンを持つヒートマップを生成する。"""
        w, h = size
        x, y = center_xy
        
        # ヒートマップの座標グリッドを生成
        xs = np.arange(w)
        ys = np.arange(h)
        xx, yy = np.meshgrid(xs, ys)
        
        # ガウス分布を計算
        heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        return heatmap