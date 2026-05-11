import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class DatasetProcessor(Dataset) :
    def __init__(self, df, domain_col='domain', label_col='label', max_len=75, voca_size=128):
        self.domain_col = domain_col
        self.label_col = label_col
        self.df = df[[self.domain_col, self.label_col]].reset_index(drop=True)
        self.max_len = max_len
        self.voca_size = voca_size

    def domain_to_indices(self, domain) :
        """도메인 문자열 → 인덱스 시퀀스"""
        domain = domain.lower()
        indices = [ord(ch) if ord(ch) < self.voca_size else 0 for ch in domain]

        # zero padding
        if len(indices) < self.max_len:
            pad_len = self.max_len - len(indices)
            indices = [0] * pad_len + indices
        else:
            indices = indices[:self.max_len]
        return np.array(indices, dtype=np.int64)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        domain = self.df.iloc[idx][self.domain_col]
        label = self.df.iloc[idx][self.label_col]
        X = self.domain_to_indices(domain)
        y = np.float32(label)
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

if __name__ == '__main__':

    df = pl.read_parquet('./dataset/T24_dga_sampled_train.parquet')
    dataset = DatasetProcessor(df)
    print(dataset.df)

    X, y = dataset[1]

    print("도메인 원본:", df.row(1)[0])
    print("전처리된 X (길이):", X.shape)
    print("전처리된 X (앞 20개):", X[:20].tolist())  # 앞 20개만 출력
    print("전처리된 X (뒤 20개):", X[-20:].tolist())
    print("라벨 y:", y.item())