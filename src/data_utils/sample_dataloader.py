import os
import sys
import time
import hydra
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from data_utils.sample_utils import sample_data


class SampleDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.dataset = []
        for _ in tqdm(range(num_samples), desc="Sampling"):
            self.dataset.append(sample_data())
        self.dataset = np.array(self.dataset, dtype=np.float32)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def create_dataloader(cfg):
    train_dataset = SampleDataset(
        num_samples=cfg.num_samples
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        shuffle=True
    )
    return train_dataloader
