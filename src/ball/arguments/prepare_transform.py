import albumentations as A

def prepare_transform(input_size):
    keypoint_params = A.KeypointParams(format='xy', remove_invisible=True)

    train_transform = A.ReplayCompose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        A.pytorch.ToTensorV2(),
    ], keypoint_params=keypoint_params)

    val_test_transform = A.ReplayCompose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(),
        A.pytorch.ToTensorV2(),
    ], keypoint_params=keypoint_params)

    return train_transform, val_test_transform