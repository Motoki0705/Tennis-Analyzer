import albumentations as A


def prepare_transform():
    """
    プレーヤー検出用の画像変換を準備する。
    
    Returns:
        A.ReplayCompose: albumentations の変換オブジェクト
    """
    bbox_params = A.BboxParams(
        format="coco", 
        label_fields=["category_id"], 
        min_visibility=0.1,  # より寛容な可視性設定
        min_area=1.0,        # 最小面積設定
        clip=True            # bbox をクリップする
    )

    train_transform = A.ReplayCompose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),      # より保守的なスケール設定
                translate_percent=(0.02, 0.02),  # より小さな移動
                rotate=5,                # より小さな回転
                shear=3,                 # より小さなせん断変形
                p=0.5,                   # 確率を下げる
                keep_ratio=True,         # アスペクト比保持
                mode=0,                  # border mode
            ),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.2),  # より保守的
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.2),
        ],
        bbox_params=bbox_params,
    )

    return train_transform
