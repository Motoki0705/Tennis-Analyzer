import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
import albumentations as A
import time
from typing import Optional, Callable, Tuple, Union
from pathlib import Path
import copy
from torch.utils.data import DataLoader

from src.ball.utils.visualize import visualize_img_with_heatmap
from src.ball.utils.heatmap import generate_gaussian_heatmap


class KeypointDataset(Dataset):
    def __init__(
            self,
            annotation_path: Union[str, Path] = "data/annotation_jsons",
            image_root: Union[str, Path] = "data/images",
            transform: Optional[Callable] = None,
            input_size: Tuple[int, int] = (512, 512),
            heatmap_size: Tuple[int, int] = (512, 512),
            input_type: str = "cat",  # "cat" or "stack"
            output_type: str = "all",  # "all" or "last"
    ):
        """
        KeypointDataset:
        ----------------
        動画フレームとキーポイントアノテーションからなるデータセットクラス。

        - 複数フレームの画像を読み込み、指定の前処理を適用。
        - 1フレームごとにキーポイントからガウスヒートマップを生成。
        - 入力フォーマット（cat/stack）および出力ヒートマップ（all frames / last frame）を柔軟に切り替え可能。

        Args:
            annotation_path (str or Path): 
                アノテーションファイルのパス（JSON形式、COCO風）。
            image_root (str or Path): 
                画像ファイルのルートディレクトリ。
            transform (Callable, optional): 
                Albumentationsのトランスフォーム。デフォルトはResize＋Normalize。
            input_size (Tuple[int, int], optional): 
                入力画像サイズ（H, W）。デフォルト: (512, 512)。
            heatmap_size (Tuple[int, int], optional): 
                出力ヒートマップサイズ（H, W）。デフォルト: (512, 512)。
            input_type (str, optional): 
                入力フォーマット:
                    - "cat": 複数フレームをチャネル方向に結合 [C*T, H, W]。
                    - "stack": 複数フレームを時間軸方向に積む [T, C, H, W]。
                デフォルト: "cat"。
            output_type (str, optional): 
                出力ヒートマップフォーマット:
                    - "all": 全フレームのヒートマップ [T, H, W]。
                    - "last": 最終フレームのみのヒートマップ [H, W]。
                デフォルト: "all"。

        Returns:
            input_tensor (torch.Tensor): 
                入力テンソル。
                - input_type="cat": [C*T, H, W]
                - input_type="stack": [T, C, H, W]
            heatmap_tensor (torch.Tensor): 
                ヒートマップテンソル。
                - output_type="all": [T, H, W]
                - output_type="last": [H, W]
        """
        assert input_type in {"cat", "stack"}, "input_type must be 'cat' or 'stack'"
        assert output_type in {"all", "last"}, "output_type must be 'all' or 'last'"
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.image_root = image_root
        self.input_type = input_type
        self.output_type = output_type

        # トランスフォーム設定
        self.transform = transform if transform is not None else A.ReplayCompose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(),
            A.pytorch.ToTensorV2()
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))

        # アノテーション読み込み
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"{annotation_path}が見つかりませんでした。")
        with open(annotation_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"{annotation_path}が開けませんでした。\n詳細：{str(e)}")

        self.samples = []
        self.video_paths = {v["id"]: v["path"] for v in data["videos"]}
        self.annotations = {ann["image_id"]: ann for ann in data["annotations"]}

        for img in data["images"]:
            video_path = self.video_paths[img["video_id"]]
            full_paths = [os.path.normpath(os.path.join(video_path, fname)) for fname in img["file_names"]]
            self.samples.append({
                "frames": full_paths,
                "label": self.annotations[img["id"]]
            })

    def __len__(self):
        return len(self.samples)

    def clip_keypoints(self, keypoints, image_shape):
        """
        キーポイント座標を画像内にクリップするユーティリティ関数。

        Args:
            keypoints (tuple): (x, y) のキーポイント座標。
            image_shape (tuple): 画像サイズ (H, W, C)。

        Returns:
            List[tuple]: クリップ後の [(x, y)] キーポイントリスト。
        """
        h, w = image_shape[:2]
        x, y = keypoints[0], keypoints[1]
        x = min(max(x, 0), w - 1e-3)
        y = min(max(y, 0), h - 1e-3)
        return [(x, y)]

    def __getitem__(self, idx):
        """
        指定インデックスのデータを取得。

        - 各フレームを読み込み、トランスフォームを適用。
        - 各フレームごとにキーポイントを変換してヒートマップを生成。
        - 入力テンソルとヒートマップテンソルを返す。

        Args:
            idx (int): サンプルインデックス。

        Returns:
            input_tensor (torch.Tensor): 
                入力テンソル（[C*T, H, W] または [T, C, H, W]）。
            heatmap_tensor (torch.Tensor): 
                ヒートマップテンソル（[T, H, W] または [H, W]）。
        """
        ...
        sample = self.samples[idx]
        frame_paths = sample["frames"]
        label = copy.deepcopy(sample["label"])
        x, y, visibility = label["keypoints"]
        keypoints = [(x, y)]

        frames = []
        heatmaps = []

        for i, path in enumerate(frame_paths):
            try:
                image_pil = Image.open(path).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"ファイルが存在しません: {path}")

            image_np = np.array(image_pil)
            img_h, img_w = image_np.shape[:2]

            # キーポイントのclip（範囲外エラー対策）
            if keypoints[0][0] >= img_w or keypoints[0][1] >= img_h:
                print(f"keypoints is out of range: {keypoints}")
                keypoints = self.clip_keypoints(keypoints[0], image_np.shape)

            if i == 0:
                augmented = self.transform(image=image_np, keypoints=keypoints)
                transform_replay = augmented["replay"]
            else:
                augmented = A.ReplayCompose.replay(transform_replay, image=image_np, keypoints=keypoints)

            image_aug = augmented["image"]
            keypoints_aug = augmented["keypoints"]
            frames.append(image_aug)

            # heatmap 作成（各フレーム用）
            if keypoints_aug:
                label["keypoints"] = list(keypoints_aug[0]) + [visibility]
            else:
                label["keypoints"] = None

            heatmap = generate_gaussian_heatmap(
                raw_label=label,
                input_size=self.input_size,
                output_size=self.heatmap_size,
            )
            heatmaps.append(heatmap)

        # 入力作成
        if self.input_type == "cat":
            input_tensor = torch.cat(frames, dim=0)  # [C*T, H, W]
        else:
            input_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]

        # 出力作成
        if self.output_type == "all":
            # [T, H, W]
            heatmap_tensor = torch.cat(heatmaps, dim=0)  # [(T, 1, H, W)] -> cat → [T, H, W]
        else:
            # 最後のフレームだけ
            heatmap_tensor = heatmaps[-1].squeeze(0)  # [1, H, W] -> [H, W]

        return input_tensor, heatmap_tensor
    

if __name__ == '__main__':
    input_size = (512, 512)
    heatmap_size = (512, 512)

    transform = A.ReplayCompose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.ColorJitter(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    dataset = KeypointDataset(
        annotation_path=r"data\ball\coco_annotations_all.json",
        image_root=r"data\ball\images",
        transform=transform,
        input_size=input_size,
        heatmap_size=heatmap_size,
        input_type="cat",    # "cat" または "stack"
        output_type="all",   # "all" または "last"
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for frames, heatmaps in dataloader:
        print(f"frames.shape: {frames.shape}")
        print(f"heatmaps.shape: {heatmaps.shape}")
        # 例えば: frames: [B, C×T, H, W] / heatmaps: [B, T, H, W]（allモード時）

        # 可視化（オプション）
        visualize_img_with_heatmap(frames, heatmaps, is_cat_frames=True)
        break
