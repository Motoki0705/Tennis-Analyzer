"""
Patch Generator
アノテーションデータからパッチを生成するユーティリティ
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    COCOアノテーションから16x16パッチを生成するクラス
    """
    
    def __init__(self, patch_size: int = 16):
        """
        Args:
            patch_size (int): パッチサイズ
        """
        self.patch_size = patch_size
        
    def generate_patches_from_coco(self,
                                  annotation_file: str,
                                  images_dir: str,
                                  output_dir: str,
                                  negative_ratio: float = 2.0,
                                  min_visibility: float = 0.5) -> Dict:
        """
        COCOアノテーションからパッチを生成
        
        Args:
            annotation_file (str): COCOアノテーションファイル
            images_dir (str): 画像ディレクトリ
            output_dir (str): パッチ出力ディレクトリ
            negative_ratio (float): 負例の比率
            min_visibility (float): 最小可視性
            
        Returns:
            Dict: 生成統計
        """
        # Load annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
            
        # Setup output directories
        output_dir = Path(output_dir)
        positive_dir = output_dir / "positive"
        negative_dir = output_dir / "negative"
        
        for dir_path in [positive_dir, negative_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create image mapping
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
            
        stats = {'positive_patches': 0, 'negative_patches': 0, 'processed_images': 0}
        
        # Process each image
        for image_id, filename in tqdm(id_to_filename.items(), desc="Generating patches"):
            image_path = Path(images_dir) / filename
            if not image_path.exists():
                continue
                
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Generate positive patches
            positive_patches = self._generate_positive_patches(
                image_rgb, image_annotations.get(image_id, []), 
                min_visibility, image_id
            )
            
            # Save positive patches
            for i, patch_data in enumerate(positive_patches):
                patch_filename = f"{image_id:06d}_{i:03d}_pos.png"
                cv2.imwrite(str(positive_dir / patch_filename), 
                           cv2.cvtColor(patch_data['patch'], cv2.COLOR_RGB2BGR))
                           
            stats['positive_patches'] += len(positive_patches)
            
            # Generate negative patches
            num_negatives = int(len(positive_patches) * negative_ratio)
            negative_patches = self._generate_negative_patches(
                image_rgb, image_annotations.get(image_id, []), 
                num_negatives, w, h
            )
            
            # Save negative patches
            for i, patch_data in enumerate(negative_patches):
                patch_filename = f"{image_id:06d}_{i:03d}_neg.png"
                cv2.imwrite(str(negative_dir / patch_filename),
                           cv2.cvtColor(patch_data['patch'], cv2.COLOR_RGB2BGR))
                           
            stats['negative_patches'] += len(negative_patches)
            stats['processed_images'] += 1
            
        # Save metadata
        metadata = {
            'patch_size': self.patch_size,
            'negative_ratio': negative_ratio,
            'min_visibility': min_visibility,
            'statistics': stats,
            'directories': {
                'positive': str(positive_dir),
                'negative': str(negative_dir)
            }
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Patch generation completed:")
        logger.info(f"  Positive patches: {stats['positive_patches']}")
        logger.info(f"  Negative patches: {stats['negative_patches']}")
        logger.info(f"  Processed images: {stats['processed_images']}")
        
        return stats
        
    def _generate_positive_patches(self, 
                                  image: np.ndarray,
                                  annotations: List[Dict],
                                  min_visibility: float,
                                  image_id: int) -> List[Dict]:
        """正例パッチの生成"""
        patches = []
        h, w = image.shape[:2]
        half_patch = self.patch_size // 2
        
        for ann in annotations:
            # Extract ball position
            if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                x, y, visibility = ann['keypoints'][:3]
                if visibility < min_visibility:
                    continue
            elif 'bbox' in ann:
                bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                x, y = bbox_x + bbox_w/2, bbox_y + bbox_h/2
                visibility = 2  # Assume visible
            else:
                continue
                
            x, y = int(x), int(y)
            
            # Check bounds
            if (x < half_patch or x >= w - half_patch or
                y < half_patch or y >= h - half_patch):
                continue
                
            # Extract patch
            x1, y1 = x - half_patch, y - half_patch
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size
            patch = image[y1:y2, x1:x2]
            
            if patch.shape[:2] == (self.patch_size, self.patch_size):
                patches.append({
                    'patch': patch,
                    'center': (x, y),
                    'visibility': visibility,
                    'image_id': image_id
                })
                
        return patches
        
    def _generate_negative_patches(self,
                                  image: np.ndarray,
                                  annotations: List[Dict],
                                  num_patches: int,
                                  width: int,
                                  height: int) -> List[Dict]:
        """負例パッチの生成"""
        patches = []
        half_patch = self.patch_size // 2
        
        # Get ball positions to avoid
        ball_positions = set()
        for ann in annotations:
            if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                x, y = ann['keypoints'][:2]
                ball_positions.add((int(x), int(y)))
            elif 'bbox' in ann:
                bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                x, y = bbox_x + bbox_w/2, bbox_y + bbox_h/2
                ball_positions.add((int(x), int(y)))
                
        attempts = 0
        max_attempts = num_patches * 10
        
        while len(patches) < num_patches and attempts < max_attempts:
            x = random.randint(half_patch, width - half_patch - 1)
            y = random.randint(half_patch, height - half_patch - 1)
            
            # Check distance from ball positions
            too_close = False
            for ball_x, ball_y in ball_positions:
                if abs(x - ball_x) < self.patch_size and abs(y - ball_y) < self.patch_size:
                    too_close = True
                    break
                    
            if not too_close:
                # Extract patch
                x1, y1 = x - half_patch, y - half_patch
                x2, y2 = x1 + self.patch_size, y1 + self.patch_size
                patch = image[y1:y2, x1:x2]
                
                if patch.shape[:2] == (self.patch_size, self.patch_size):
                    patches.append({
                        'patch': patch,
                        'center': (x, y),
                        'visibility': 0,  # Negative
                        'image_id': 0
                    })
                    
            attempts += 1
            
        return patches
        
    def visualize_patches(self, 
                         patches_dir: str,
                         output_path: str,
                         num_samples: int = 50):
        """パッチサンプルの可視化"""
        import matplotlib.pyplot as plt
        
        patches_dir = Path(patches_dir)
        positive_dir = patches_dir / "positive"
        negative_dir = patches_dir / "negative"
        
        # Load sample patches
        pos_files = list(positive_dir.glob("*.png"))[:num_samples//2]
        neg_files = list(negative_dir.glob("*.png"))[:num_samples//2]
        
        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        
        for i, ax in enumerate(axes.flat):
            if i < len(pos_files):
                # Positive sample
                img = cv2.imread(str(pos_files[i]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title("Positive", color='green', fontsize=8)
            elif i - len(pos_files) < len(neg_files):
                # Negative sample
                img = cv2.imread(str(neg_files[i - len(pos_files)]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title("Negative", color='red', fontsize=8)
            else:
                ax.axis('off')
                continue
                
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Patch visualization saved: {output_path}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate patches from COCO annotations")
    parser.add_argument("--annotation_file", required=True, help="COCO annotation file")
    parser.add_argument("--images_dir", required=True, help="Images directory")
    parser.add_argument("--output_dir", required=True, help="Output patches directory")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--negative_ratio", type=float, default=2.0, help="Negative patch ratio")
    parser.add_argument("--min_visibility", type=float, default=0.5, help="Minimum visibility threshold")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    
    args = parser.parse_args()
    
    # Generate patches
    generator = PatchGenerator(patch_size=args.patch_size)
    stats = generator.generate_patches_from_coco(
        annotation_file=args.annotation_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        negative_ratio=args.negative_ratio,
        min_visibility=args.min_visibility
    )
    
    # Generate visualization if requested
    if args.visualize:
        vis_path = Path(args.output_dir) / "patch_samples.png"
        generator.visualize_patches(args.output_dir, str(vis_path))
        
    print(f"\n✅ Patch generation completed!")
    print(f"Positive patches: {stats['positive_patches']}")
    print(f"Negative patches: {stats['negative_patches']}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main() 