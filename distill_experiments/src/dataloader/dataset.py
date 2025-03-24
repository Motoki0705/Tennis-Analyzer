from torchvision import datasets, transforms

# 訓練データ用のtransform (データ拡張あり)
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # ランダムに±10度回転
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ランダムな平行移動
    transforms.ToTensor(),          # Tensor型に変換（0～1に正規化される）
    transforms.Normalize((0.1307,), (0.3081,))  # MNISTの標準的な平均と標準偏差で正規化
])

# 検証（テスト）データ用のtransform（データ拡張なし）
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# データセット作成
train_dataset = datasets.MNIST(root='data', train=True, transform=train_transform, download=True)
val_dataset = datasets.MNIST(root='data', train=False, transform=val_transform, download=True)
