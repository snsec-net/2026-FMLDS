"""HMT dataloader and bigram dictionary builder (pandas-friendly)."""
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset


class BigramDictionaryBuilder:
    def __init__(self, domains, max_voca_size=2000):
        self.domains = domains
        self.max_voca_size = max_voca_size

    def build(self):
        all_bigrams = []
        for d in self.domains:
            d = str(d).lower()
            if len(d) < 2:
                continue
            all_bigrams.extend(d[i:i+2] for i in range(len(d) - 1))
        cnt = Counter(all_bigrams)
        top = cnt.most_common(self.max_voca_size)
        return {bg: i + 1 for i, (bg, _) in enumerate(top)}


class HMTDataset(Dataset):
    def __init__(self, df, bigram_to_index, domain_col='domain', label_col='label',
                 max_len_char=49, voca_size_char=256, max_len_bigram=48):
        self.x = df[domain_col].tolist()
        self.y = df[label_col].values.astype(np.float32)
        self.max_len_char = max_len_char
        self.voca_size_char = voca_size_char
        self.max_len_bigram = max_len_bigram
        self.bigram_to_index = bigram_to_index
        self.voca_size_bigram = len(bigram_to_index) + 2
        self.oov = self.voca_size_bigram - 1

    def __len__(self):
        return len(self.y)

    def _encode(self, domain):
        d = str(domain).lower()
        ic = [ord(c) if ord(c) < self.voca_size_char else 0 for c in d]
        if len(ic) < self.max_len_char:
            ic = [0] * (self.max_len_char - len(ic)) + ic
        else:
            ic = ic[:self.max_len_char]

        if len(d) >= 2:
            bigrams = [d[i:i+2] for i in range(len(d) - 1)]
        else:
            bigrams = []
        ib = [self.bigram_to_index.get(b, self.oov) for b in bigrams]
        if len(ib) < self.max_len_bigram:
            ib = [0] * (self.max_len_bigram - len(ib)) + ib
        else:
            ib = ib[:self.max_len_bigram]
        return np.array(ic, dtype=np.int64), np.array(ib, dtype=np.int64)

    def __getitem__(self, idx):
        xc, xb = self._encode(self.x[idx])
        return (torch.tensor(xc, dtype=torch.long),
                torch.tensor(xb, dtype=torch.long),
                torch.tensor(self.y[idx], dtype=torch.float32))
