from torch.utils.data import Dataset
import torch
import numpy as np

def process_domain(domain):
        # 소문자 변환 및 뒤에서 75자만 자르기
        domain = str(domain).lower()[-75:]
        
        # ASCII 정수로 변환
        indices = [ord(c) for c in domain]
        
        # 0으로 패딩 (75자리 맞추기)
        pad_len = 75 - len(indices)
        return [0] * pad_len + indices

class DGA_Dataset(Dataset):
    def __init__(self, df, domain_col='domain', label_col='label'):
        self.x = df[domain_col].tolist() # 리스트의 리스트 형태: [[0, 0, ..., 97, 98], ...]
        self.y = df[label_col].values    # numpy 배열 형태: [0, 1, 0, ...]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raw_domain = self.x[idx]
        processed_list = process_domain(raw_domain)
        
        x_tensor = torch.tensor(processed_list, dtype=torch.long)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return x_tensor, y_tensor