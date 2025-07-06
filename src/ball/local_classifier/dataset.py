"""
Ball Patch Dataset
既存のプロジェクトデータ構造に合わせた16x16パッチ2値分類データセット
"""

import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BallPatchDataset(Dataset):
    """
    16x16パッチでのボール2値分類データセット
    
    既存のプロジェクトのアノテーション形式に対応:
    - game_id/clip_id 構造
    - original_path フィールド
    - keypoints [x, y, visibility] 形式
    """
    
    def __init__(self,
                 annotation_file: str,
                 images_dir: str,
                 patch_size: int = 16,
                 negative_ratio: float = 2.0,
                 min_ball_visibility: float = 0.5,
                 transform: Optional[A.Compose] = None,
                 split: str = "train",
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 seed: int = 42,
                 position_noise: int = 2,  # ±ピクセル数の位置ノイズ
                 cache_images: bool = False):
        """
        Args:
            annotation_file (str): プロジェクト形式アノテーションファイル
            images_dir (str): 画像ディレクトリ (datasets/ball/images)
            patch_size (int): パッチサイズ
            negative_ratio (float): 負例の割合（正例の何倍か）
            min_ball_visibility (float): 最小可視性閾値
            transform (Optional): データ拡張
            split (str): "train", "val", "test"
            train_ratio (float): 学習データ割合
            val_ratio (float): 検証データ割合
            seed (int): データ分割シード
            position_noise (int): ボール位置の±ピクセルノイズ（検出誤差再現）
            cache_images (bool): 画像をメモリにキャッシュするか
        """
        self.annotation_file = annotation_file
        self.images_dir = Path(images_dir)
        self.patch_size = patch_size
        self.negative_ratio = negative_ratio
        self.min_ball_visibility = min_ball_visibility
        self.cache_images = cache_images
        self.split = split
        self.position_noise = position_noise
        
        # Load annotations
        self.data = self._load_annotations()
        
        # Split data by clip (to avoid temporal leakage)
        self.split_data = self._split_data_by_clip(train_ratio, val_ratio, seed)
        
        # Generate patch samples
        self.samples = self._generate_samples()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        # Image cache
        self.image_cache = {} if cache_images else None
        
        logger.info(f"Dataset initialized [{split}]: {len(self.samples)} samples "
                   f"({self._count_positive()} positive, {self._count_negative()} negative)")
        logger.info(f"Position noise: ±{self.position_noise}px (simulating detection uncertainty)")
        
    def _load_annotations(self) -> Dict:
        """アノテーションの読み込み"""
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
            
        # Create mappings
        self.images_by_id = {img['id']: img for img in data['images']}
        
        # Group annotations by image (only ball category)
        self.annotations_by_image = {}
        for ann in data['annotations']:
            if ann.get('category_id') == 1:  # Ball category
                image_id = ann['image_id']
                self.annotations_by_image[image_id] = ann
                
        logger.info(f"Loaded {len(self.images_by_id)} images, {len(self.annotations_by_image)} ball annotations")
        return data
        
    def _split_data_by_clip(self, train_ratio: float, val_ratio: float, seed: int) -> List[int]:
        """クリップ単位でのデータ分割（時間的リークを防ぐため）"""
        # Group by (game_id, clip_id)
        clip_groups = defaultdict(list)
        for img_id, img_info in self.images_by_id.items():
            key = (img_info['game_id'], img_info['clip_id'])
            clip_groups[key].append(img_id)
            
        # Split clips
        clip_keys = list(clip_groups.keys())
        random.Random(seed).shuffle(clip_keys)
        
        n_clips = len(clip_keys)
        n_train = int(n_clips * train_ratio)
        n_val = int(n_clips * val_ratio)
        
        split_map = {
            "train": clip_keys[:n_train],
            "val": clip_keys[n_train:n_train + n_val],
            "test": clip_keys[n_train + n_val:],
        }
        
        # Get image IDs for this split
        target_clips = split_map[self.split]
        split_image_ids = []
        for clip_key in target_clips:
            split_image_ids.extend(clip_groups[clip_key])
            
        logger.info(f"Split [{self.split}]: {len(target_clips)} clips, {len(split_image_ids)} images")
        print(f"📊 {self.split}データ: {len(target_clips)}クリップ, {len(split_image_ids)}画像")
        return split_image_ids
        
    def _generate_samples(self) -> List[Dict]:
        """正例・負例サンプルの生成（高速化版）"""
        samples = []
        
        print(f"🔄 {self.split} split: パッチ生成中...")
        
        # Step 1: 画像寸法キャッシュの構築
        print("📏 画像寸法キャッシュ構築中...")
        image_dimensions = {}
        
        with tqdm(self.split_data, desc="寸法取得", unit="images", ncols=100) as pbar:
            for image_id in pbar:
                img_info = self.images_by_id[image_id]
                image_path = self.images_dir / img_info['original_path']
                
                if image_path.exists():
                    try:
                        img = cv2.imread(str(image_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            image_dimensions[image_id] = (w, h, str(image_path))
                    except Exception as e:
                        logger.warning(f"Failed to load image {image_path}: {e}")
                        continue
                        
                pbar.set_postfix({'キャッシュ済み': len(image_dimensions)})
        
        print(f"✅ 画像寸法キャッシュ完了: {len(image_dimensions)}画像")
        
        # Step 2: 正例サンプル生成（高速化済み）
        positive_count = 0
        print("📍 正例パッチ生成中...")
        
        with tqdm(self.split_data, desc="正例生成", unit="images", ncols=100) as pbar:
            for image_id in pbar:
                if image_id not in self.annotations_by_image or image_id not in image_dimensions:
                    continue
                    
                ann = self.annotations_by_image[image_id]
                w, h, image_path = image_dimensions[image_id]
                
                # Check visibility
                keypoints = ann['keypoints']
                if len(keypoints) >= 3 and keypoints[2] >= self.min_ball_visibility:
                    # Get original ball position
                    orig_x, orig_y = int(keypoints[0]), int(keypoints[1])
                    
                    # Add realistic position noise to simulate detection uncertainty
                    noise_x = random.randint(-self.position_noise, self.position_noise)
                    noise_y = random.randint(-self.position_noise, self.position_noise)
                    x = orig_x + noise_x
                    y = orig_y + noise_y
                    
                    # Check if patch fits in image bounds (with noise)
                    half_patch = self.patch_size // 2
                    
                    if half_patch <= x < w - half_patch and half_patch <= y < h - half_patch:
                        samples.append({
                            'image_path': image_path,
                            'center_x': x,
                            'center_y': y,
                            'label': 1,  # Positive
                            'image_id': image_id,
                            'visibility': keypoints[2]
                        })
                        positive_count += 1
                        
                pbar.set_postfix({'正例数': positive_count})
                    
        print(f"✅ 正例生成完了: {positive_count}サンプル")
        
        # Step 3: 負例サンプル生成（大幅高速化）
        negative_count = int(positive_count * self.negative_ratio)
        negative_generated = 0
        
        print(f"📍 負例パッチ生成中 (目標: {negative_count}サンプル)...")
        
        # 正例位置のインデックス構築
        positive_positions_by_image = {}
        for sample in samples:
            if sample['label'] == 1:
                image_id = sample['image_id']
                if image_id not in positive_positions_by_image:
                    positive_positions_by_image[image_id] = []
                positive_positions_by_image[image_id].append((sample['center_x'], sample['center_y']))
        
        # 効率的な負例生成
        available_images = [img_id for img_id in image_dimensions.keys()]
        random.shuffle(available_images)
        
        with tqdm(total=negative_count, desc="負例生成", unit="samples", ncols=100) as pbar:
            for image_id in available_images:
                if negative_generated >= negative_count:
                    break
                    
                w, h, image_path = image_dimensions[image_id]
                positive_positions = positive_positions_by_image.get(image_id, [])
                
                # 各画像から複数の負例を効率的に生成
                samples_per_image = min(8, negative_count - negative_generated)  # 増量
                batch_samples = self._generate_negative_batch(
                    image_path, image_id, w, h, positive_positions, samples_per_image
                )
                
                samples.extend(batch_samples)
                negative_generated += len(batch_samples)
                
                pbar.update(len(batch_samples))
                pbar.set_postfix({'残り': negative_count - negative_generated})
                    
        print(f"✅ 負例生成完了: {negative_generated}サンプル")
        print(f"🎯 総サンプル数: {positive_count + negative_generated} (正例: {positive_count}, 負例: {negative_generated})")
        
        logger.info(f"Generated {positive_count} positive, {negative_generated} negative samples")
        return samples
    
    def _generate_negative_batch(self, image_path: str, image_id: int, width: int, height: int, 
                               positive_positions: List[Tuple[int, int]], num_samples: int) -> List[Dict]:
        """効率的な負例バッチ生成"""
        batch_samples = []
        half_patch = self.patch_size // 2
        min_distance = self.patch_size
        
        # 有効な座標範囲
        x_min, x_max = half_patch, width - half_patch - 1
        y_min, y_max = half_patch, height - half_patch - 1
        
        if x_min >= x_max or y_min >= y_max:
            return []  # 画像が小さすぎる
        
        # グリッドベースサンプリング（効率的）
        attempts = 0
        max_attempts = num_samples * 20  # 制限を増やす
        
        while len(batch_samples) < num_samples and attempts < max_attempts:
            # ランダム位置生成
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            
            # 正例位置との距離チェック（ベクトル化）
            too_close = False
            for pos_x, pos_y in positive_positions:
                if abs(x - pos_x) < min_distance and abs(y - pos_y) < min_distance:
                    # 簡易距離チェック（高速）
                    too_close = True
                    break
                    
            if not too_close:
                batch_samples.append({
                    'image_path': image_path,
                    'center_x': x,
                    'center_y': y,
                    'label': 0,  # Negative
                    'image_id': image_id,
                    'visibility': 0.0
                })
                
            attempts += 1
            
        return batch_samples
        
    def _get_default_transforms(self) -> A.Compose:
        """デフォルトデータ拡張"""
        if self.split == "train":
            # Training augmentations
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=10.0, p=0.3),  
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Validation/test - minimal augmentation
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """画像の読み込み（キャッシュ対応）"""
        if self.image_cache is not None:
            if image_path not in self.image_cache:
                img = cv2.imread(image_path)
                if img is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_cache[image_path] = img
            return self.image_cache[image_path]
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    def _count_positive(self) -> int:
        """正例数のカウント"""
        return sum(1 for sample in self.samples if sample['label'] == 1)
        
    def _count_negative(self) -> int:
        """負例数のカウント"""
        return sum(1 for sample in self.samples if sample['label'] == 0)
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load image
            img = self._load_image(sample['image_path'])
            
            # Extract patch
            half_patch = self.patch_size // 2
            center_x, center_y = sample['center_x'], sample['center_y']
            
            x1 = center_x - half_patch
            y1 = center_y - half_patch
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size
            
            # Handle boundary cases
            h, w = img.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                # Pad image if necessary
                pad_left = max(0, -x1)
                pad_top = max(0, -y1)
                pad_right = max(0, x2 - w)
                pad_bottom = max(0, y2 - h)
                
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                           mode='reflect')
                
                # Adjust coordinates
                x1 += pad_left
                y1 += pad_top
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                
            patch = img[y1:y2, x1:x2]
            
            # Ensure patch size
            if patch.shape[:2] != (self.patch_size, self.patch_size):
                patch = cv2.resize(patch, (self.patch_size, self.patch_size))
                
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=patch)
                patch = transformed['image']
                
            label = torch.tensor(sample['label'], dtype=torch.float32)
            
            return patch, label
            
        except Exception as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            # Return a zero patch with negative label as fallback
            zero_patch = torch.zeros(3, self.patch_size, self.patch_size)
            return zero_patch, torch.tensor(0.0, dtype=torch.float32)


def create_dataloaders(annotation_file: str,
                      images_dir: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      patch_size: int = 16,
                      negative_ratio: float = 2.0,
                      min_ball_visibility: float = 0.5,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      seed: int = 42,
                      position_noise: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    学習・検証用データローダーの作成
    
    Args:
        annotation_file (str): アノテーションファイル
        images_dir (str): 画像ディレクトリ
        batch_size (int): バッチサイズ
        num_workers (int): ワーカー数
        patch_size (int): パッチサイズ
        negative_ratio (float): 負例の割合
        min_ball_visibility (float): 最小可視性
        train_ratio (float): 学習データ割合
        val_ratio (float): 検証データ割合
        seed (int): データ分割シード
        position_noise (int): ボール位置ノイズ（±ピクセル）
        
    Returns:
        Tuple[DataLoader, DataLoader]: 学習・検証データローダー
    """
    # Create datasets
    train_dataset = BallPatchDataset(
        annotation_file=annotation_file,
        images_dir=images_dir,
        patch_size=patch_size,
        negative_ratio=negative_ratio,
        min_ball_visibility=min_ball_visibility,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        position_noise=position_noise
    )
    
    val_dataset = BallPatchDataset(
        annotation_file=annotation_file,
        images_dir=images_dir,
        patch_size=patch_size,
        negative_ratio=negative_ratio,
        min_ball_visibility=min_ball_visibility,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        position_noise=0  # 検証時はノイズなし
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Test function
if __name__ == "__main__":
    # Test dataset loading
    annotation_file = "datasets/ball/coco_annotations_ball_pose_court.json"
    images_dir = "datasets/ball/images"
    
    dataset = BallPatchDataset(
        annotation_file=annotation_file,
        images_dir=images_dir,
        patch_size=16,
        split="train"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    if len(dataset) > 0:
        patch, label = dataset[0]
        print(f"Patch shape: {patch.shape}")
        print(f"Label: {label}")
        print("Dataset loading successful!")
    else:
        print("No samples found in dataset!") 