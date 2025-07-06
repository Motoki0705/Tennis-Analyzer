"""
Patch Generator Utilities
画像からパッチを生成するユーティリティ関数群
"""

import cv2
import numpy as np
import json
from typing import List, Tuple, Dict, Optional, Generator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    16x16パッチ生成器
    """
    
    def __init__(self, patch_size: int = 16):
        """
        Args:
            patch_size (int): パッチサイズ
        """
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        
    def extract_patch(self, 
                     image: np.ndarray, 
                     center_x: int, 
                     center_y: int,
                     ensure_size: bool = True) -> Optional[np.ndarray]:
        """
        指定位置からパッチを抽出
        
        Args:
            image (np.ndarray): 入力画像 [H, W, C]
            center_x (int): パッチ中心X座標
            center_y (int): パッチ中心Y座標 
            ensure_size (bool): サイズ保証のためのリサイズを行うか
            
        Returns:
            Optional[np.ndarray]: パッチ または None (境界外)
        """
        h, w = image.shape[:2]
        
        # 境界チェック
        if (center_x < self.half_patch or center_x >= w - self.half_patch or
            center_y < self.half_patch or center_y >= h - self.half_patch):
            
            if not ensure_size:
                return None
                
            # パディングして抽出
            return self._extract_patch_with_padding(image, center_x, center_y)
            
        # 通常の抽出
        x1 = center_x - self.half_patch
        y1 = center_y - self.half_patch
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size
        
        patch = image[y1:y2, x1:x2]
        
        # サイズ確認
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            if ensure_size:
                patch = cv2.resize(patch, (self.patch_size, self.patch_size))
            else:
                return None
                
        return patch
        
    def _extract_patch_with_padding(self, 
                                   image: np.ndarray, 
                                   center_x: int, 
                                   center_y: int) -> np.ndarray:
        """境界を超える場合のパディング付き抽出"""
        h, w = image.shape[:2]
        
        # パディングサイズ計算
        pad_top = max(0, self.half_patch - center_y)
        pad_bottom = max(0, center_y + self.half_patch - h + 1)
        pad_left = max(0, self.half_patch - center_x)
        pad_right = max(0, center_x + self.half_patch - w + 1)
        
        # パディング適用
        if len(image.shape) == 3:
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                           mode='reflect')
        else:
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                           mode='reflect')
            
        # 調整された座標で抽出
        adj_center_x = center_x + pad_left
        adj_center_y = center_y + pad_top
        
        x1 = adj_center_x - self.half_patch
        y1 = adj_center_y - self.half_patch
        x2 = x1 + self.patch_size
        y2 = y1 + self.patch_size
        
        return padded[y1:y2, x1:x2]
        
    def generate_positive_patches(self, 
                                 image: np.ndarray,
                                 ball_positions: List[Tuple[int, int]]) -> List[np.ndarray]:
        """
        ボール位置からの正例パッチ生成
        
        Args:
            image (np.ndarray): 入力画像
            ball_positions (List[Tuple]): ボール位置のリスト [(x, y), ...]
            
        Returns:
            List[np.ndarray]: 正例パッチのリスト
        """
        patches = []
        
        for x, y in ball_positions:
            patch = self.extract_patch(image, x, y)
            if patch is not None:
                patches.append(patch)
                
        return patches
        
    def generate_negative_patches(self, 
                                 image: np.ndarray,
                                 ball_positions: List[Tuple[int, int]],
                                 num_negatives: int,
                                 min_distance: int = None) -> List[np.ndarray]:
        """
        ランダム位置からの負例パッチ生成
        
        Args:
            image (np.ndarray): 入力画像
            ball_positions (List[Tuple]): 避けるべきボール位置
            num_negatives (int): 生成する負例数
            min_distance (int): ボール位置からの最小距離
            
        Returns:
            List[np.ndarray]: 負例パッチのリスト
        """
        h, w = image.shape[:2]
        patches = []
        
        if min_distance is None:
            min_distance = self.patch_size
            
        attempts = 0
        max_attempts = num_negatives * 10
        
        while len(patches) < num_negatives and attempts < max_attempts:
            # ランダム位置生成
            x = np.random.randint(self.half_patch, w - self.half_patch)
            y = np.random.randint(self.half_patch, h - self.half_patch)
            
            # ボール位置との距離チェック
            too_close = False
            for ball_x, ball_y in ball_positions:
                distance = np.sqrt((x - ball_x)**2 + (y - ball_y)**2)
                if distance < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                patch = self.extract_patch(image, x, y)
                if patch is not None:
                    patches.append(patch)
                    
            attempts += 1
            
        return patches
        
    def generate_patches_from_coco(self, 
                                  annotation_file: str,
                                  images_dir: str,
                                  negative_ratio: float = 2.0,
                                  min_visibility: float = 0.5) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        COCOアノテーションからパッチを生成
        
        Args:
            annotation_file (str): COCOアノテーションファイル
            images_dir (str): 画像ディレクトリ
            negative_ratio (float): 負例の割合
            min_visibility (float): 最小可視性閾値
            
        Yields:
            Tuple[np.ndarray, int]: (patch, label) のペア
        """
        # Load annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
            
        # Create image mapping
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
            
        # Process each image
        for image_id, filename in id_to_filename.items():
            image_path = Path(images_dir) / filename
            if not image_path.exists():
                continue
                
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get ball positions
            ball_positions = []
            if image_id in image_annotations:
                for ann in image_annotations[image_id]:
                    if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                        x, y, visibility = ann['keypoints'][:3]
                        if visibility >= min_visibility:
                            ball_positions.append((int(x), int(y)))
                    elif 'bbox' in ann:
                        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                        x = int(bbox_x + bbox_w / 2)
                        y = int(bbox_y + bbox_h / 2)
                        ball_positions.append((x, y))
                        
            # Generate positive patches
            positive_patches = self.generate_positive_patches(image, ball_positions)
            for patch in positive_patches:
                yield patch, 1
                
            # Generate negative patches
            num_negatives = int(len(positive_patches) * negative_ratio)
            negative_patches = self.generate_negative_patches(image, ball_positions, num_negatives)
            for patch in negative_patches:
                yield patch, 0
                
    def visualize_patches(self, 
                         patches: List[np.ndarray],
                         labels: List[int],
                         save_path: str,
                         max_display: int = 50):
        """
        パッチの可視化
        
        Args:
            patches (List[np.ndarray]): パッチのリスト
            labels (List[int]): ラベルのリスト
            save_path (str): 保存パス
            max_display (int): 最大表示数
        """
        import matplotlib.pyplot as plt
        
        # Separate positive and negative patches
        pos_patches = [p for p, l in zip(patches, labels) if l == 1]
        neg_patches = [p for p, l in zip(patches, labels) if l == 0]
        
        # Limit display count
        pos_patches = pos_patches[:max_display//2]
        neg_patches = neg_patches[:max_display//2]
        
        fig, axes = plt.subplots(2, max(len(pos_patches), len(neg_patches)), 
                               figsize=(20, 6))
        
        # Plot positive patches
        for i, patch in enumerate(pos_patches):
            if i < axes.shape[1]:
                axes[0, i].imshow(patch)
                axes[0, i].set_title('Positive', color='green')
                axes[0, i].axis('off')
                
        # Plot negative patches
        for i, patch in enumerate(neg_patches):
            if i < axes.shape[1]:
                axes[1, i].imshow(patch)
                axes[1, i].set_title('Negative', color='red')
                axes[1, i].axis('off')
                
        # Hide unused axes
        for i in range(max(len(pos_patches), len(neg_patches)), axes.shape[1]):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Patch visualization saved: {save_path}")


def create_training_patches(annotation_file: str,
                           images_dir: str,
                           output_dir: str,
                           patch_size: int = 16,
                           negative_ratio: float = 2.0,
                           min_visibility: float = 0.5,
                           max_patches_per_class: int = 10000):
    """
    学習用パッチデータセットの作成
    
    Args:
        annotation_file (str): COCOアノテーションファイル
        images_dir (str): 画像ディレクトリ
        output_dir (str): 出力ディレクトリ
        patch_size (int): パッチサイズ
        negative_ratio (float): 負例の割合
        min_visibility (float): 最小可視性
        max_patches_per_class (int): クラス毎の最大パッチ数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    (output_dir / "positive").mkdir(exist_ok=True)
    (output_dir / "negative").mkdir(exist_ok=True)
    
    generator = PatchGenerator(patch_size)
    
    pos_count = 0
    neg_count = 0
    
    logger.info("Generating training patches...")
    
    for patch, label in generator.generate_patches_from_coco(
        annotation_file, images_dir, negative_ratio, min_visibility):
        
        if label == 1 and pos_count < max_patches_per_class:
            # Save positive patch
            filename = f"pos_{pos_count:06d}.png"
            cv2.imwrite(str(output_dir / "positive" / filename), 
                       cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            pos_count += 1
            
        elif label == 0 and neg_count < max_patches_per_class:
            # Save negative patch
            filename = f"neg_{neg_count:06d}.png"
            cv2.imwrite(str(output_dir / "negative" / filename),
                       cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            neg_count += 1
            
        # Progress logging
        if (pos_count + neg_count) % 1000 == 0:
            logger.info(f"Generated {pos_count} positive, {neg_count} negative patches")
            
        # Check completion
        if pos_count >= max_patches_per_class and neg_count >= max_patches_per_class:
            break
            
    logger.info(f"Patch generation completed:")
    logger.info(f"  Positive patches: {pos_count}")
    logger.info(f"  Negative patches: {neg_count}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Save metadata
    metadata = {
        "patch_size": patch_size,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "negative_ratio": negative_ratio,
        "min_visibility": min_visibility,
        "annotation_file": str(annotation_file),
        "images_dir": str(images_dir)
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training patches")
    parser.add_argument("--annotation_file", required=True, help="COCO annotation file")
    parser.add_argument("--images_dir", required=True, help="Images directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--negative_ratio", type=float, default=2.0, help="Negative ratio")
    parser.add_argument("--max_patches", type=int, default=10000, help="Max patches per class")
    
    args = parser.parse_args()
    
    create_training_patches(
        annotation_file=args.annotation_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        negative_ratio=args.negative_ratio,
        max_patches_per_class=args.max_patches
    ) 