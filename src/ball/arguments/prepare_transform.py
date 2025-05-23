import albumentations as A


def prepare_transform(input_size, is_train=True):
    """
    学習用と評価用の画像変換を準備します。

    Args:
        input_size (list): 入力画像サイズ [height, width]
        is_train (bool): 学習用変換を返すかどうか

    Returns:
        A.ReplayCompose: 画像変換オブジェクト
    """
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=True)

    if is_train:
        transform = A.ReplayCompose(
            [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )
    else:
        transform = A.ReplayCompose(
            [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ],
            keypoint_params=keypoint_params,
        )

    return transform
