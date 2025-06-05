import json
import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from torch.utils.data import Dataset

from src.court.dataset.court_dataset import CourtDataset


class PseudoLabeledCourtDataset(CourtDataset):
    """
    擬似ラベルをサポートするためのCourtDatasetの拡張クラス。
    オリジナルのラベル付きデータセットに擬似ラベルを追加し、
    擬似ラベルには学習時に重みを適用できるようにします。
    """

    def __init__(
        self,
        annotation_path: str,
        image_root: str,
        input_size: Tuple[int, int] = (360, 640),
        heatmap_size: Tuple[int, int] = (360, 640),
        default_num_keypoints: int = 15,
        transform=None,
        sigma: float = 2,
        is_each_keypoint: bool = True,
    ):
        """
        初期化

        Parameters
        ----------
        annotation_path : COCOフォーマットアノテーションファイル
        image_root : 画像ファイルのルートディレクトリ
        input_size : 入力画像サイズ (H, W)
        heatmap_size : ヒートマップサイズ (H, W)
        default_num_keypoints : デフォルトのキーポイント数
        transform : Albumentations変換
        sigma : ガウスカーネルの標準偏差
        is_each_keypoint : 各キーポイントに個別のヒートマップを生成するかどうか
        """
        super().__init__(
            annotation_path=annotation_path,
            image_root=image_root,
            input_size=input_size,
            heatmap_size=heatmap_size,
            default_num_keypoints=default_num_keypoints,
            transform=transform,
            sigma=sigma,
            is_each_keypoint=is_each_keypoint,
        )
        
        # 擬似ラベル関連の属性
        self.pseudo_labels = {}  # image_path -> annotation
        self.pseudo_weight = 1.0  # 擬似ラベルの重み
        self.has_pseudo_labels = False
        
        # 元のデータのインデックスマッピングを作成
        self.path_to_idx = {example["file_name"]: i for i, example in enumerate(self.data)}

    def add_pseudo_labels(self, pseudo_label_path: Union[str, Path], weight: float = 0.5):
        """
        擬似ラベルを追加する

        Parameters
        ----------
        pseudo_label_path : 擬似ラベルのCOCOフォーマットファイル
        weight : 擬似ラベルの重み (0.0 ~ 1.0)
        """
        pseudo_label_path = Path(pseudo_label_path)
        with open(pseudo_label_path, "r", encoding="utf-8") as f:
            pseudo_data = json.load(f)

        # 擬似ラベルデータの処理
        self.pseudo_labels = {}
        
        # 画像IDからファイルパスへのマッピングを作成
        if isinstance(pseudo_data, list):
            # リスト形式のJSONデータ
            for item in pseudo_data:
                file_name = item.get("file_name")
                if file_name and file_name in self.path_to_idx:
                    self.pseudo_labels[file_name] = {
                        "keypoints": item.get("keypoints", []),
                        "is_pseudo": True,
                        "score": item.get("score", 0.5)
                    }
        else:
            # COCO形式のJSONデータ
            image_id_to_path = {}
            for img in pseudo_data.get("images", []):
                image_id_to_path[img["id"]] = img.get("file_name")
            
            for ann in pseudo_data.get("annotations", []):
                if ann.get("category_id") == 4:  # コートカテゴリ
                    img_id = ann["image_id"]
                    file_name = image_id_to_path.get(img_id)
                    
                    if file_name and file_name in self.path_to_idx:
                        self.pseudo_labels[file_name] = {
                            "keypoints": ann.get("keypoints", []),
                            "is_pseudo": True,
                            "score": ann.get("score", 0.5),
                            "num_keypoints": len(ann.get("keypoints", [])) // 3
                        }

        self.pseudo_weight = max(0.0, min(weight, 1.0))
        self.has_pseudo_labels = True
        print(f"Added {len(self.pseudo_labels)} pseudo labels with weight {self.pseudo_weight}")

    def __getitem__(self, idx):
        """
        データセットからアイテムを取得する

        Parameters
        ----------
        idx : インデックス

        Returns
        -------
        arged_image : 変換後の画像テンソル
        heatmaps : ヒートマップテンソル
        is_pseudo : 擬似ラベルかどうかのフラグ
        """
        example = self.data[idx]
        file_name = example["file_name"]
        
        # 擬似ラベルの確認
        is_pseudo = False
        if self.has_pseudo_labels and file_name in self.pseudo_labels:
            is_pseudo = True
            # 擬似ラベルでサンプルを更新
            example_copy = copy.deepcopy(example)
            pseudo_data = self.pseudo_labels[file_name]
            example_copy["keypoints"] = pseudo_data["keypoints"]
            example_copy["num_keypoints"] = pseudo_data.get(
                "num_keypoints", self.default_num_keypoints
            )
            example = example_copy

        # 画像読み込み
        image_path = os.path.join(self.image_root, file_name)
        image = self.load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image. image_path: {image_path}")

        # キーポイントの準備
        keypoints = torch.tensor(example["keypoints"], dtype=torch.float32).view(-1, 3)
        default_num_keypoints = example.get("num_keypoints", self.default_num_keypoints)

        if keypoints.size(0) != default_num_keypoints:
            raise ValueError(
                f"Keypoint count mismatch before augmentation.\n"
                f"Actual: {keypoints.size(0)}\n"
                f"Expected: {default_num_keypoints}\n"
            )

        # データ拡張
        image = np.array(image)
        argumented = self.transform(image=image, keypoints=keypoints)
        arged_image = argumented["image"]
        arged_keypoints = torch.tensor(argumented["keypoints"])

        if arged_keypoints.size(0) != default_num_keypoints:
            raise ValueError(
                f"Keypoint count mismatch after augmentation.\n"
                f"After: {arged_keypoints.size(0)}\n"
                f"Expected: {default_num_keypoints}\n"
            )

        # 画面外のキーポイントのフィルタリング
        self.filtering_outside_screen_keypoints(arged_keypoints, self.input_size)

        # ヒートマップサイズに合わせてスケーリング
        scaled_keypoints = self.scaling_to_heatmap(
            arged_keypoints, self.input_size, self.heatmap_size
        )

        # ヒートマップ生成
        if self.is_each_keypoint:
            heatmaps = torch.zeros(
                (len(scaled_keypoints), *self.heatmap_size), dtype=torch.float32
            )
            for i, (x, y, v) in enumerate(scaled_keypoints):
                if v > 0:
                    self.draw_each_gaussian(
                        heatmaps[i], (x.item(), y.item()), sigma=self.sigma
                    )
        else:
            heatmaps = torch.zeros((1, *self.heatmap_size), dtype=torch.float32)
            for i, (x, y, v) in enumerate(scaled_keypoints):
                if v > 0:
                    self.draw_each_gaussian(
                        heatmaps[0], (x.item(), y.item()), sigma=self.sigma
                    )

        return arged_image, heatmaps, torch.tensor([is_pseudo], dtype=torch.bool) 