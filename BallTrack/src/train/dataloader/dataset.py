import os
import csv

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from BallTrack.src.train.utils.heatmap import generate_heatmap
from BallTrack.src.tests.dataset_tester import DatasetVisualizer

class TennisDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', train_ratio=0.7, val_ratio=0.15):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        print('dataset processing')
        
        for game in sorted(os.listdir(self.root_dir)):
            game_path = os.path.join(self.root_dir, game)
            if not os.path.isdir(game_path):
                continue
            for clip in sorted(os.listdir(game_path)):
                clip_path = os.path.join(game_path, clip)
                if not os.path.isdir(clip_path):
                    continue
                image_files = sorted([f for f in os.listdir(clip_path) if f.endswith(".jpg")])
                if len(image_files) < 3:
                    continue

                label_file = os.path.join(root_dir, game, clip, "Label.csv")
                if not os.path.exists(label_file):
                    print(f"ラベルファイルが見つかりません: {label_file}")
                    continue
                
                labels = {}
                with open(label_file, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:                            
                        filename = row[0]
                        # 欠損値は仮のデータを渡す。学習に用いない。
                        if int(row[1]) == 0: 
                            labels[filename] = {
                            "visibility": int(row[1]),
                            "x": None,
                            "y": None,
                            "status": None
                        }
                        else:
                            labels[filename] = {
                                "visibility": int(row[1]),
                                "x": float(row[2]),
                                "y": float(row[3]),
                                "status": int(row[4])
                            }
                for i in range(1, len(image_files) - 1):
                    mid_file = image_files[i]
                    if mid_file not in labels:
                        continue
                    if labels[mid_file]['visibility'] == 0:
                        continue
                    sample = {
                        "prev": os.path.join(clip_path, image_files[i - 1]),
                        "curr": os.path.join(clip_path, mid_file),
                        "next": os.path.join(clip_path, image_files[i + 1]),
                        "label": labels[mid_file]
                    }
                    self.samples.append(sample)
        
        # split to train, val and test
        num_samples = len(self.samples)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        num_train_samples = int(num_samples * train_ratio)
        num_val_samples = int(num_samples * val_ratio)

        train_indices = indices[:num_train_samples]
        val_indices = indices[num_train_samples:num_train_samples + num_val_samples]
        test_indices = indices[num_train_samples * num_val_samples:]

        if mode == 'train':
            self.selected_indices = train_indices
        
        elif mode == 'val':
            self.selected_indices = val_indices

        elif mode == 'test':
            self.selected_indices = test_indices

        else:
            raise ValueError("mode should be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        real_idx = self.selected_indices[idx]
        sample = self.samples[real_idx]
        prev_img = Image.open(sample["prev"]).convert("RGB")
        curr_img = Image.open(sample["curr"]).convert("RGB")
        next_img = Image.open(sample["next"]).convert("RGB")
        
        if self.transform:
            prev_img = self.transform(prev_img)
            curr_img = self.transform(curr_img)
            next_img = self.transform(next_img)
            
        input_tensor = torch.cat([prev_img, curr_img, next_img], dim=0)
        target = generate_heatmap(sample["label"], output_size=(512, 512))
        
        return input_tensor, target

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = TennisDataset(root_dir=r'TrackNet/datasets/Tennis', transform=transform)
    
    # testerにはデータ拡張を使用する場合と使用しない場合でどちらも行う
    tester = DatasetVisualizer(dataset=dataset, delay=0.1)
    tester.visualize()
    