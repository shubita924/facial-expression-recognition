import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SPLIT_MAP = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}

class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split="train"):
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == SPLIT_MAP[split]]
        self.labels = df["emotion"].values
        self.pixels = df["pixels"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = np.array(self.pixels[idx].split(), dtype=np.float32).reshape(1, 48, 48) / 255.0
        return torch.from_numpy(img), int(self.labels[idx])
