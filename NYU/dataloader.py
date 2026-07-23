from torch.utils.data import Dataset
import torch
import numpy as np


def process_domain(domain, max_len=75):
    domain = str(domain).lower()[-max_len:]
    indices = [ord(c) if ord(c) < 128 else 0 for c in domain]
    pad_len = max_len - len(indices)
    return [0] * pad_len + indices


class DGA_Dataset(Dataset):
    def __init__(self, df, domain_col='domain', label_col='label', max_len=75):
        self.x = df[domain_col].tolist()
        self.y = df[label_col].values
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = process_domain(self.x[idx], self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)
