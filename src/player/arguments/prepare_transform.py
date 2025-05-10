import albumentations as A

def prepare_transform():
    bbox_params = A.BboxParams(format='coco', label_fields=['category_id'], min_visibility=0.3)

    train_transform = A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.05, 0.05),
            rotate=10,
            shear=5,
            p=0.7
        ),
        A.Perspective(
            scale=(0.05, 0.1), keep_size=True, p=0.3
        ),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1, p=0.7
        ),
        A.Blur(blur_limit=3, p=0.3),
    ], bbox_params=bbox_params)

    return train_transform
