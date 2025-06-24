import os
import cv2
import torch
import random
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import pandas as pd
import numpy as np

class CasiaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((120, 120)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vid_path = os.path.join(self.root_dir, row['video_path'])
        cap = cv2.VideoCapture(vid_path)
        ret, frame = cap.read()
        cap.release()
        label = 1 if row['label'] != 'real' else 0  # spoof:1, real:0
        if not ret:
            frame = np.zeros((120, 120, 3), dtype=np.uint8)
        img = self.transform(frame)
        return img, label

class BalancedSampler(Sampler):
    def __init__(self, labels, batch_size, spoof_ratio=0.5):
        self.indices = list(range(len(labels)))
        self.labels = labels
        self.batch_size = batch_size
        self.spoof_ratio = spoof_ratio

    def __iter__(self):
        spoof = [i for i in self.indices if self.labels[i] == 1]
        real = [i for i in self.indices if self.labels[i] == 0]
        min_len = min(len(spoof), len(real))
        batch_indices = []
        for _ in range(min_len * 2 // self.batch_size):
            batch = random.sample(spoof, self.batch_size // 2) + \
                    random.sample(real, self.batch_size // 2)
            random.shuffle(batch)
            batch_indices.extend(batch)
        return iter(batch_indices)

    def __len__(self):
        return len(self.indices)
