import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Set

import numpy as np
import torchvision
from PIL import Image

from src.player.arguments.prepare_transform import prepare_transform


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        annotation_file,
        cat_id_map,
        use_original_path,
        split="train",
        transform=None,
        seed=42,
    ):
        """
        - img_folder: 画像フォルダパス
        - annotation_file: COCOアノテーションファイル
        - cat_id_map: カテゴリIDのマッピング
        - use_original_path: Trueなら 'original_path' を使う
        - split: 'train', 'val', 'test'
        - transform: albumentationsなどの変換
        - seed: ランダムシード
        """
        super().__init__(img_folder, annotation_file)
        self.img_folder = img_folder
        self.cat_id_map = cat_id_map
        self.use_original_path = use_original_path
        self.transform = transform
        self.seed = seed
        self.ids = self.split_ids(split=split)

    def __getitem__(self, idx):
        """
        データセットから1つのサンプルを取得する。
        
        Args:
            idx: インデックス
            
        Returns:
            tuple: (image, target) または None（処理に失敗した場合）
        """
        try:
            if self.use_original_path:
                img, target = self.getitem_from_original_path(idx)
            else:
                img, target = super().__getitem__(idx)

            img = np.array(img)

            # category_id=1（ボール）を除去、カテゴリマップ適用
            target = [
                {
                    **ann,
                    "category_id": self.cat_id_map.get(
                        ann["category_id"], ann["category_id"]
                    ),
                }
                for ann in target
                if ann["category_id"] != 1 and self._is_valid_bbox(ann.get("bbox"))
            ]

            if len(target) == 0:
                print(f"target has no valid item for idx {idx}")
                return None
                
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

        if self.transform is not None:
            img, bboxes, labels = self.apply_transform(img, target)
            if len(bboxes) == 0:
                print("boxes has no item")
                return None

            new_annotations = []
            img_height, img_width = img.shape[:2]
            
            for box, lab in zip(bboxes, labels, strict=False):
                x, y, w, h = box
                
                # bbox の座標を正規化して範囲チェック・修正
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(1, min(w, img_width - x))
                h = max(1, min(h, img_height - y))
                
                # 面積が最小値未満の場合はスキップ
                if w * h < 1:
                    continue
                    
                new_annotations.append(
                    {
                        "image_id": target[0].get("image_id", -1),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "category_id": int(lab),
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )
                
            if len(new_annotations) == 0:
                print("All bboxes were filtered out")
                return None
                
            target = new_annotations

        target_for_proc = {
            "image_id": target[0].get("image_id", -1),
            "annotations": target,
        }

        return img, target_for_proc

    def _is_valid_bbox(self, bbox):
        """
        bbox が有効かどうかをチェックする。
        
        Args:
            bbox: [x, y, width, height] 形式の bbox
            
        Returns:
            bool: bbox が有効な場合 True
        """
        if bbox is None or len(bbox) != 4:
            return False
            
        x, y, w, h = bbox
        
        # 座標と大きさが正の値であることを確認
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return False
            
        # 面積が最小値以上であることを確認
        if w * h < 4:  # 最小面積 2x2 ピクセル
            return False
            
        return True

    def apply_transform(self, img, target):
        """
        albumentations transform を適用する。
        COCO 形式の bbox を適切に処理する。
        
        Args:
            img: PIL Image または numpy array
            target: アノテーションリスト
            
        Returns:
            tuple: (transformed_img, transformed_bboxes, transformed_labels)
        """
        # 画像を numpy array に変換
        if hasattr(img, 'size'):  # PIL Image の場合
            img_array = np.array(img)
        else:
            img_array = img
            
        try:
            bboxes_for_transform = [ann["bbox"] for ann in target]
            labels_for_transform = [ann["category_id"] for ann in target]
            
            # albumentations transform を適用
            argumented = self.transform(
                image=img_array, 
                bboxes=bboxes_for_transform, 
                category_id=labels_for_transform
            )
            
            img_transformed = argumented["image"]
            bboxes_transformed = argumented["bboxes"]
            labels_transformed = argumented["category_id"]
            
            return img_transformed, bboxes_transformed, labels_transformed
            
        except Exception as e:
            print(f"Transform failed: {e}")
            print(f"Image shape: {img_array.shape}")
            print(f"Bboxes: {bboxes_for_transform}")
            # transform に失敗した場合は元の値を返す
            return img_array, [ann["bbox"] for ann in target], [ann["category_id"] for ann in target]

    def split_ids(self, split="train") -> List[int]:
        """
        プレーヤーが2人いる画像だけを有効とし、ランダムにtrain/val/testに分割する。
        """
        images = self.coco.dataset.get("images", [])
        annotations = self.coco.dataset.get("annotations", [])

        # 🔥 プレーヤーが2人いる画像IDだけを選定
        valid_image_ids = self.filtering_valid_img_ids(annotations)

        images = [img for img in images if img["id"] in valid_image_ids]
        all_ids = [img["id"] for img in images]
        random.seed(self.seed)
        random.shuffle(all_ids)

        n = len(all_ids)
        n_train = int(n * 0.7)
        n_val = int(n * 0.2)

        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train : n_train + n_val]
        test_ids = all_ids[n_train + n_val :]

        self.print_stats("train", train_ids, annotations)
        self.print_stats("val", val_ids, annotations)
        self.print_stats("test", test_ids, annotations)

        if split == "train":
            return train_ids
        elif split == "val":
            return val_ids
        elif split == "test":
            return test_ids
        else:
            raise ValueError(f"Unknown split name: {split}")

    def filtering_valid_img_ids(self, annotations: List[Dict[str, Any]]) -> Set[int]:
        """
        プレーヤー（category_id=2）が2人いる画像だけを有効とする。
        """
        player_count_per_image = defaultdict(int)

        for ann in annotations:
            if ann.get("category_id") == 2:
                player_count_per_image[ann["image_id"]] += 1

        valid_image_ids = {
            img_id for img_id, count in player_count_per_image.items() if count == 2
        }

        return valid_image_ids

    @staticmethod
    def print_stats(split_name: str, ids: List[int], annotations: List[Dict[str, Any]]):
        split_anns = [
            ann
            for ann in annotations
            if ann.get("image_id") in ids and ann.get("category_id") == 2
        ]
        num_images = len(set(ann["image_id"] for ann in split_anns))
        total_anns = len(split_anns)
        print(
            f"[{split_name.upper()}] Images: {num_images}, Total Annotations: {total_anns}"
        )

    def getitem_from_original_path(self, idx):
        coco = self.coco
        img_id = self.ids[idx]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        original_path = img_info.get(
            "original_path", img_info.get("file_name")
        )  # フォールバック
        path = os.path.join(self.img_folder, original_path)

        img = Image.open(path).convert("RGB")
        return img, anns


if __name__ == "__main__":
    train_transform = prepare_transform()

    dataset = CocoDetection(
        img_folder=r"data\ball\images",
        annotation_file=r"data\ball\coco_annotations_ball_ranged.json",
        cat_id_map={2: 0},
        use_original_path=True,
        split="train",
        transform=train_transform,
    )

