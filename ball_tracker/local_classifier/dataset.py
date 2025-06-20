"""
Ball Patch Dataset
アノテーション付きデータから16x16パッチを生成する2値分類データセット
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

logger = logging.getLogger(__name__)


class BallPatchDataset(Dataset):
    """
    16x16パッチでのボール2値分類データセット
    
    正例: ボール中心の16x16パッチ
    負例: ランダム位置の16x16パッチ（ボールを含まない）
    """
    
    def __init__(self,
                 annotation_file: str,
                 images_dir: str,
                 patch_size: int = 16,
                 negative_ratio: float = 2.0,
                 min_ball_visibility: float = 0.5,
                 transform: Optional[A.Compose] = None,
                 cache_images: bool = False):
        """
        Args:
            annotation_file (str): COCO形式アノテーションファイル
            images_dir (str): 画像ディレクトリ
            patch_size (int): パッチサイズ
            negative_ratio (float): 負例の割合（正例の何倍か）
            min_ball_visibility (float): 最小可視性閾値
            transform (Optional): データ拡張
            cache_images (bool): 画像をメモリにキャッシュするか
        """
        self.annotation_file = annotation_file
        self.images_dir = Path(images_dir)
        self.patch_size = patch_size
        self.negative_ratio = negative_ratio
        self.min_ball_visibility = min_ball_visibility
        self.cache_images = cache_images
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Generate patch samples
        self.samples = self._generate_samples()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        # Image cache
        self.image_cache = {} if cache_images else None
        
        logger.info(f"Dataset initialized: {len(self.samples)} samples "
                   f"({self._count_positive()} positive, {self._count_negative()} negative)")
        
    def _load_annotations(self) -> Dict:
        """COCOアノテーションの読み込み"""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
            
        # Create image_id to filename mapping
        self.id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)
            
        return annotations
        
    def _generate_samples(self) -> List[Dict]:
        """正例・負例サンプルの生成"""
        samples = []
        
        for image_id, filename in self.id_to_filename.items():
            image_path = self.images_dir / filename
            if not image_path.exists():
                continue
                
            # 画像サイズ取得
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # 正例サンプル（ボール中心）
            positive_samples = self._generate_positive_samples(image_id, filename, w, h)
            samples.extend(positive_samples)
            
            # 負例サンプル（ランダム位置）
            num_negatives = int(len(positive_samples) * self.negative_ratio)
            negative_samples = self._generate_negative_samples(image_id, filename, w, h, num_negatives)
            samples.extend(negative_samples)
            
        return samples
        
    def _generate_positive_samples(self, image_id: int, filename: str, width: int, height: int) -> List[Dict]:
        """正例サンプルの生成"""
        samples = []
        
        if image_id not in self.image_annotations:
            return samples
            
        for ann in self.image_annotations[image_id]:
            # Visibility check
            if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                visibility = ann['keypoints'][2]
                if visibility < self.min_ball_visibility:
                    continue
                    
                # Keypoint position
                x, y = ann['keypoints'][0], ann['keypoints'][1]
            elif 'bbox' in ann:
                # Use bbox center
                bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                x, y = bbox_x + bbox_w/2, bbox_y + bbox_h/2
            else:
                continue
                
            # Check if patch fits in image
            half_patch = self.patch_size // 2
            if (half_patch <= x < width - half_patch and 
                half_patch <= y < height - half_patch):
                
                samples.append({
                    'image_path': str(self.images_dir / filename),
                    'center_x': int(x),
                    'center_y': int(y),
                    'label': 1,  # Positive
                    'image_id': image_id
                })
                
        return samples
        
    def _generate_negative_samples(self, image_id: int, filename: str, 
                                 width: int, height: int, num_samples: int) -> List[Dict]:
        """負例サンプルの生成"""
        samples = []
        half_patch = self.patch_size // 2
        
        # Get positive positions to avoid
        positive_positions = set()
        if image_id in self.image_annotations:
            for ann in self.image_annotations[image_id]:
                if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                    x, y = ann['keypoints'][0], ann['keypoints'][1]
                    positive_positions.add((int(x), int(y)))
                    
        # Generate random negative samples
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(samples) < num_samples and attempts < max_attempts:
            x = random.randint(half_patch, width - half_patch - 1)
            y = random.randint(half_patch, height - half_patch - 1)
            
            # Check distance from positive samples
            too_close = False
            for pos_x, pos_y in positive_positions:
                if abs(x - pos_x) < self.patch_size and abs(y - pos_y) < self.patch_size:
                    too_close = True
                    break
                    
            if not too_close:
                samples.append({
                    'image_path': str(self.images_dir / filename),
                    'center_x': x,
                    'center_y': y,
                    'label': 0,  # Negative
                    'image_id': image_id
                })
                
            attempts += 1
            
        return samples
        
    def _get_default_transforms(self) -> A.Compose:
        """デフォルトデータ拡張"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """画像の読み込み（キャッシュ対応）"""
        if self.image_cache is not None:
            if image_path not in self.image_cache:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_cache[image_path] = img
            return self.image_cache[image_path]
        else:
            img = cv2.imread(image_path)
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
        
        # Load image
        img = self._load_image(sample['image_path'])
        
        # Extract patch
        half_patch = self.patch_size // 2
        center_x, center_y = sample['center_x'], sample['center_y']
        
        x1 = center_x - half_patch
        y1 = center_y - half_patch
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


class BalancedBallPatchDataset(BallPatchDataset):
    """
    バランスの取れたパッチデータセット
    各バッチで正例・負例を均等に含む
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Separate positive and negative samples
        self.positive_samples = [s for s in self.samples if s['label'] == 1]
        self.negative_samples = [s for s in self.samples if s['label'] == 0]
        
        # Calculate effective dataset size
        self.effective_size = min(len(self.positive_samples), len(self.negative_samples)) * 2
        
    def __len__(self) -> int:
        return self.effective_size
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Alternate between positive and negative
        if idx % 2 == 0:
            # Positive sample
            sample_idx = (idx // 2) % len(self.positive_samples)
            sample = self.positive_samples[sample_idx]
        else:
            # Negative sample
            sample_idx = (idx // 2) % len(self.negative_samples)
            sample = self.negative_samples[sample_idx]
            
        # Same processing as parent class
        return super(BallPatchDataset, self).__getitem__(self.samples.index(sample))


def create_dataloaders(annotation_file: str,
                      images_dir: str,
                      train_ratio: float = 0.8,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    学習・検証用データローダーの作成
    
    Args:
        annotation_file (str): アノテーションファイル
        images_dir (str): 画像ディレクトリ
        train_ratio (float): 学習データの割合
        batch_size (int): バッチサイズ
        num_workers (int): ワーカー数
        **dataset_kwargs: データセット追加パラメータ
        
    Returns:
        Tuple[DataLoader, DataLoader]: 学習・検証データローダー
    """
    # Full dataset
    full_dataset = BallPatchDataset(annotation_file, images_dir, **dataset_kwargs)
    
    # Train/validation split
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Dataset test
    print("🏐 Ball Patch Dataset Test")
    print("=" * 40)
    
    # テスト用の仮パラメータ
    annotation_file = "path/to/annotations.json"  # 実際のパスに変更
    images_dir = "path/to/images"  # 実際のパスに変更
    
    if os.path.exists(annotation_file) and os.path.exists(images_dir):
        dataset = BallPatchDataset(
            annotation_file=annotation_file,
            images_dir=images_dir,
            patch_size=16,
            negative_ratio=2.0
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Positive samples: {dataset._count_positive()}")
        print(f"Negative samples: {dataset._count_negative()}")
        
        # Sample data
        if len(dataset) > 0:
            patch, label = dataset[0]
            print(f"Patch shape: {patch.shape}")
            print(f"Label: {label.item()}")
    else:
        print("⚠️ Test annotation file or images directory not found") 