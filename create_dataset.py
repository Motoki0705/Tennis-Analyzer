import numpy as np
import os
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pickle
import csv
import pandas as pd
from config import Config

class FrameDataset(Dataset):
    def __init__(self, file_directory, transformer=None, config: Config=None, train=False):
        self.config = config
        self.transformer = transformer
        seane_dirs = self.load_seanedirs(file_directory)
        
        #frame_pathsを取得
        if not os.path.exists('frame_paths.pkl'):
            frame_paths = self.load_filepaths(seane_dirs, ".jpg")
            with open('frame_paths.pkl', 'wb') as file:
                pickle.dump(frame_paths, file)
        
        with open('frame_paths.pkl', 'rb') as file:
            self.frame_paths = pickle.load(file)
        
        #csv_pathsを取得
        self.csv_paths = self.load_filepaths(seane_dirs, ".csv")
        
        self.frame_paths, self.label = self.commit_frame_label(self.frame_paths, self.csv_paths)
        
        if train:
            self.frame_paths, self.label = self.shrink_outplay(self.frame_paths, self.label)
            self.label_state, self.label_event = self.get_dummies_label(self.label)
        
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        label_state = torch.tensor(self.label_state.iloc[idx].values, dtype=torch.float32)
        label_event = torch.tensor(self.label_event.iloc[idx].values, dtype=torch.float32)
        
        frame = Image.open(frame_path).convert('RGB')
        
        if self.transformer:
            frame = self.transformer(frame)
        
        return frame, label_state, label_event
            
            
    def load_seanedirs(self,file_directory):
        seane_dirs = [os.path.join(file_directory, f) for f in os.listdir(file_directory) if os.path.isdir(os.path.join(file_directory, f))]
        sorted_seane_dirs = sorted(
            seane_dirs,
            key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group())
            )
        
        return sorted_seane_dirs

    def load_filepaths(self, file_directories, extension):
        file_paths = []
        for file_directory in file_directories:
            files = [os.path.join(file_directory, f) for f in os.listdir(file_directory) if f.endswith(extension)]
            sorted_files = sorted(
                files,
                key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group())
            )
            file_paths.append(sorted_files)
            
        return file_paths

    def commit_frame_label(self, frame_paths, label_paths):
        commited_frame_paths = []
        commited_labels = []
        for idx, frames in enumerate(frame_paths):
            index_list = [int(re.search(r"\d+", frame).group()) for frame in frames]
            label = pd.read_csv(label_paths[idx][0])
            commited_label = label.iloc[index_list]
            commited_frame_paths.extend(frames)
            commited_labels.append(commited_label)

        return commited_frame_paths, pd.concat(commited_labels, ignore_index=True)
    
    def shrink_outplay(self, frame_paths, label):
        inplay_index = (label[label['state'] == 'inplay'].index).tolist()
        add_index = []
        pre_idx = 0
        for idx in inplay_index:
            if pre_idx + 1 == idx:
                continue
            add_index.extend(range(pre_idx + 1, pre_idx + self.config.outplay_range + 1))
            add_index.extend(range(idx - self.config.outplay_range, idx))
            pre_idx = idx
        inplay_index.extend(add_index)
        shrinked_outplay_index = sorted(list(set(inplay_index)))
        
        frame_paths = [frame_paths[idx] for idx in shrinked_outplay_index]
        label = label.iloc[shrinked_outplay_index]
        
        return frame_paths, label
    
    def get_dummies_label(self, label):
        state_categories = self.config.states
        event_categories = self.config.events

        label_state = pd.get_dummies(label['state']).reindex(columns=state_categories, fill_value=0).astype(int)
        label_event = pd.get_dummies(label['event']).reindex(columns=event_categories, fill_value=0).astype(int)
        return label_state, label_event
    
if __name__ == '__main__':
    config = Config()
    transformer = transforms.Compose([
        transforms.Resize(config.input_frame_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = FrameDataset(r'C:\Users\kamim\Desktop\vscode\tennis_analyzer\frames', transformer, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    for frame, label_state, label_event in dataloader:
        print(frame.shape)
        print(label_state.shape)
        print(label_event.shape)
        break