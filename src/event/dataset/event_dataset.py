import json
import os
from typing import Tuple, Optional

import albumentations as A
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class EventDataset(Dataset):
    def __init__(self,
        annotation_path: str,
        image_root: str,
    ):
        self.annotation_path = annotation_path
        self.image_root = image_root
    
    def __len__(self):
        return len(self.annotation_path)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        annotation = self.annotation_path[idx]
        image_path = os.path.join(self.image_root, annotation["image_path"])
        image = self.load_image(image_path)
        

