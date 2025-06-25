import os
import cv2
import torch
import random
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import pandas as pd
import numpy as np

class CasiaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((120, 120)),
            transforms.ToTensor()
        ])
        # Duyệt folder real và spoof
        for label_name, label in [('real', 0), ('spoof', 1)]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                img_path = os.path.join(folder, fname)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((120, 120, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
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
