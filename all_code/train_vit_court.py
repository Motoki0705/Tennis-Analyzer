from types import SimpleNamespace
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from TennisCourtDetector.src.dataset.court_dataset import CustomCourtDataset
from TennisCourtDetector.src.models.vit_court import ViTCourt
import os
from TennisCourtDetector.src.trainer.court_trainer import Trainer

# 引数管理
args = SimpleNamespace(
    input_size=(224, 224),  # ViTに合わせて調整（224推奨）
    output_size=(56, 56),
    batch_size=4,
    num_epochs=20,
    learning_rate=1e-4,
    train_path="./data/converted_train.json",
    val_path="./data/converted_val.json",
    image_root="./data/images",
    save_dir="checkpoints",
    freeze_epochs=3,  # ViTを最初にfreezeするエポック数
    device="cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(args.save_dir, exist_ok=True)

# 前処理 (ViT標準の正規化に合わせている)
train_transforms = transforms.Compose([
    transforms.Resize(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# データセットと DataLoader
train_dataset = CustomCourtDataset(
    json_path=args.train_path,
    image_root=args.image_root,
    input_size=args.input_size,
    output_size=args.output_size,
    transforms=train_transforms
)
val_dataset = CustomCourtDataset(
    json_path=args.val_path,
    image_root=args.image_root,
    input_size=args.input_size,
    output_size=args.output_size,
    transforms=train_transforms
)
# モデルと学習設定
model = ViTCourt(num_keypoints=15).to(args.device)

trainer = Trainer(model, train_dataset, val_dataset, args)
trainer.train()