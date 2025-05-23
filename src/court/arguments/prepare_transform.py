from typing import List, Tuple, Union, Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2


def prepare_transform(
    input_size: Union[List[int], Tuple[int, int]], is_train: bool = True
) -> Callable:
    """
    コート検出用の画像変換パイプラインを準備します。

    Args:
        input_size: 入力画像サイズ (H, W)
        is_train: 学習用の変換を含めるかどうか

    Returns:
        albumentations の Compose オブジェクト
    """
    # 基本変換（テスト・推論用）
    transforms = []

    if is_train:
        # 学習時のデータ拡張
        transforms.extend([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            ], p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                A.MotionBlur(blur_limit=(3, 5), p=0.4),
            ], p=0.5),
            A.RandomResizedCrop(
                height=input_size[0],
                width=input_size[1],
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.7,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,
                p=0.7,
            ),
        ])
    else:
        # テスト・推論時はリサイズのみ
        transforms.append(
            A.Resize(height=input_size[0], width=input_size[1], p=1.0),
        )

    # 共通の最終変換
    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    return A.Compose(transforms) 