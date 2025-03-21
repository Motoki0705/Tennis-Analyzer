from torchvision import transforms

def prepare_transforms(resize_shape):
    # ImageNetでよく用いられる正規化パラメータを使用（必要に応じて変更してください）
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # トレーニング時の変換:
    # ※ジオメトリ変換（例: 平行移動、回転など）は避け、
    #    色調補正やグレースケール変換など、画像の内容自体は変化させるもののみを選んでいます
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
        transforms.Resize(resize_shape)
    ])
    
    # 検証・テスト時の変換:
    # 指定どおり、ToTensor, Normalize, Resize のみを適用
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Resize(resize_shape)
    ])
    
    return train_transform, val_test_transform
