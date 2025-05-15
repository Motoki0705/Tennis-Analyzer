import albumentations as A

def prepare_transform():
    bbox_params = A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.5, filter_invalid_bboxes=True)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),            # 拡大・縮小
            translate_percent=(0.05, 0.05), # 平行移動
            rotate=10,                  # 回転（±10度）
            shear=5,                    # シアー（±5度）
            p=0.7
        ),
        A.Perspective(
            scale=(0.05, 0.1), keep_size=True, p=0.3
        ),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1, p=0.7
        ),
        A.Blur(blur_limit=3, p=0.3),  # または A.GaussianBlur(blur_limit=3, p=0.3)
    ], bbox_params=bbox_params)
    
    return transform