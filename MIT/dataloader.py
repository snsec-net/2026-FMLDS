from torch.utils.data import Dataset
import torch
import numpy as np

class DGA_Dataset(Dataset):
    def __init__(self, df, domain_col='domain', label_col='label'):
        self.x = df[domain_col].tolist() # 리스트의 리스트 형태: [[0, 0, ..., 97, 98], ...]
        self.y = df[label_col].values    # numpy 배열 형태: [0, 1, 0, ...]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # x는 파이썬 리스트이므로 torch.tensor로 변환 (Embedding layer를 위해 long 타입)
        x_tensor = torch.tensor(self.x[idx], dtype=torch.long)
        
        # y는 스칼라 값이므로 torch.tensor로 변환 (BCELoss를 위해 float32 타입)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return x_tensor, y_tensor