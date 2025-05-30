import json
import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from src.ball.dataset.seq_key_dataset import SequenceKeypointDataset


class PseudoLabeledSequenceDataset(SequenceKeypointDataset):
    """
    擬似ラベルをサポートするためのSequenceKeypointDatasetの拡張クラス。
    オリジナルのラベル付きデータセットに擬似ラベルを追加し、
    擬似ラベルには学習時に重みを適用できるようにします。
    """

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        T: int,
        input_size: List[int],
        heatmap_size: List[int],
        transform,
        split: str = "train",
        use_group_split: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        input_type: str = "stack",
        output_type: str = "all",
        skip_frames_range: tuple = (1, 5),
    ):
        """
        初期化

        Parameters
        ----------
        annotation_file : COCOフォーマットアノテーションファイル
        image_root : 画像ファイルのルートディレクトリ
        T : シーケンス長
        input_size : 入力画像サイズ [H, W]
        heatmap_size : ヒートマップサイズ [H, W]
        transform : Albumentations変換
        split : データセット分割 "train" / "val" / "test"
        use_group_split : クリップ単位でデータ分割するかどうか
        train_ratio : トレーニングセット比率
        val_ratio : 検証セット比率
        seed : 乱数シード
        input_type : 入力フォーマット "cat" / "stack"
        output_type : 出力フォーマット "all" / "last"
        skip_frames_range : スキップするフレーム範囲
        """
        super().__init__(
            annotation_file=annotation_file,
            image_root=image_root,
            T=T,
            input_size=input_size,
            heatmap_size=heatmap_size,
            transform=transform,
            split=split,
            use_group_split=use_group_split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            input_type=input_type,
            output_type=output_type,
            skip_frames_range=skip_frames_range,
        )
        
        # 擬似ラベル関連の属性
        self.pseudo_labels = {}  # image_id -> annotation
        self.pseudo_weight = 1.0  # 擬似ラベルの重み
        self.has_pseudo_labels = False

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

        # 擬似ラベルの抽出とマッピング
        self.pseudo_labels = {}
        for ann in pseudo_data["annotations"]:
            if ann.get("category_id") == 1:  # ボールカテゴリ
                img_id = ann["image_id"]
                # キーポイント形式に変換 [x, y, v]
                x, y, w, h = ann["bbox"]
                center_x = x + w / 2
                center_y = y + h / 2
                score = ann.get("score", 0.5)
                
                self.pseudo_labels[img_id] = {
                    "keypoints": [center_x, center_y, 1.0],  # 可視性は1.0に設定
                    "score": score,
                    "is_pseudo": True
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
        input_tensor : 入力テンソル
        heatmap_tensor : ヒートマップテンソル
        visibility_tensor : 可視性テンソル
        is_pseudo : 擬似ラベルかどうかのフラグ
        """
        # 親クラスの__getitem__を呼び出して基本データを取得
        input_tensor, heatmap_tensor, visibility_tensor = super().__getitem__(idx)
        
        # 擬似ラベルかどうかのフラグを追加
        clip_key, start = self.windows[idx]
        ids_sorted = sorted(self.clip_groups[clip_key])
        
        if self.output_type == "all":
            # 各フレームが擬似ラベルかどうかを確認
            is_pseudo = torch.zeros(self.T, dtype=torch.bool)
            for i, img_id in enumerate(ids_sorted[start:start+self.T]):
                if img_id in self.pseudo_labels:
                    is_pseudo[i] = True
        else:  # "last"
            # 最後のフレームが擬似ラベルかどうかを確認
            last_img_id = ids_sorted[start+self.T-1]
            is_pseudo = torch.tensor([last_img_id in self.pseudo_labels], dtype=torch.bool)
        
        return input_tensor, heatmap_tensor, visibility_tensor, is_pseudo

    def _get_annotation(self, img_id):
        """
        画像IDに対応するアノテーションを取得する
        擬似ラベルが利用可能な場合はそれを優先
        
        Parameters
        ----------
        img_id : 画像ID
        
        Returns
        -------
        annotation : アノテーション辞書
        """
        if self.has_pseudo_labels and img_id in self.pseudo_labels:
            return self.pseudo_labels[img_id]
        
        return self.anns_by_image.get(img_id) 