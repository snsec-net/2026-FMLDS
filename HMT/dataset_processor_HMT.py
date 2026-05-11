import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

class BigramDictionaryBuilder :
    def __init__(self, df, domain_col='domain', max_voca_size = 2000) :
        self.df = df
        self.domain_col = domain_col
        self.max_voca_size = max_voca_size
        self.bigram_to_index: dict[str, int] = {}

    def extract_all_bigrams(self):
        counter = Counter()
        # all_bigrams = []

        for domain in self.df[self.domain_col] :
            domain = domain.lower()
            if len(domain) < 2 :
                continue
            # bigrams = [domain[i:i+2] for i in range(len(domain) - 1)]
            counter.update(domain[i:i+2] for i in range(len(domain)-1))
            # all_bigrams.extend(bigrams)
        
        return counter
    
    def build_dictionary(self) :
        bigram_counts = self.extract_all_bigrams()

        # 상위 2000개 선택
        top_bigrams = bigram_counts.most_common(self.max_voca_size)
        # print("Selected top bigrams:", len(top_bigrams))

        bigram_to_index = {}
        for index, (bigram, _) in enumerate(top_bigrams, start=1):
            bigram_to_index[bigram] = index

        self.bigram_to_index = bigram_to_index

        return bigram_to_index


class DatasetProcessor(Dataset) :
    def __init__(self, df, domain_col='domain', label_col='label', max_len_char=49, voca_size_char=256, 
                 max_len_bigram=48, bigram_to_index = None):
        self.domain_col = domain_col
        self.label_col = label_col
        self.df = df[[self.domain_col, self.label_col]].reset_index(drop=True)
        self.max_len_char = max_len_char
        self.max_len_bigram = max_len_bigram
        self.voca_size_char = voca_size_char
        self.bigram_to_index = bigram_to_index
        if bigram_to_index is None:
            raise ValueError("Bigram requires 'bigram_to_index' dictionary.")
        self.voca_size_bigram = len(bigram_to_index) + 2
        self.oov_index = self.voca_size_bigram - 1

    def domain_to_bigrams(self, domain):
        """도메인 문자열을 바이그램 문자열 시퀀스로 변환"""
        domain = domain.lower()
        if len(domain) < 2:
             return []
        
        bigrams = [domain[i:i+2] for i in range(len(domain) - 1)]
        return bigrams
        
    def domain_to_indices(self, domain) :
        """도메인 문자열 → 인덱스 시퀀스"""
        domain = domain.lower()

        # 문자 수준 -> 아스키 인덱스 사용
        indices_char = [ord(ch) if ord(ch) < self.voca_size_char else 0 for ch in domain]
        # zero padding
        if len(indices_char) < self.max_len_char:
            pad_len = self.max_len_char - len(indices_char)
            indices_char = [0] * pad_len + indices_char
        else:
            indices_char = indices_char[:self.max_len_char]

        # bigram 수준 -> 외부 딕셔너리 사용
        bigrams = self.domain_to_bigrams(domain)
        # OOV (Out-of-Vocabulary) 처리를 위한 인덱스 처리
        indices_bigram = [self.bigram_to_index.get(bigram, self.oov_index) for bigram in bigrams] #OOV는 패딩과 별도 처리 해야함 
        # zero padding
        if len(indices_bigram) < self.max_len_bigram:
            pad_len = self.max_len_bigram - len(indices_bigram)
            indices_bigram = [0] * pad_len + indices_bigram
        else:
            indices_bigram = indices_bigram[:self.max_len_bigram]

        return np.array(indices_char, dtype=np.int64), np.array(indices_bigram, dtype=np.int64)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        domain = self.df.loc[idx, self.domain_col]
        label = self.df.loc[idx, self.label_col]
        X_char, X_bigram = self.domain_to_indices(domain)

        if self.label_col == 'label' :
            y = np.float32(label)
            return torch.tensor(X_char, dtype=torch.long), torch.tensor(X_bigram, dtype=torch.long), torch.tensor(y, dtype=torch.float32)
        else : 
            y = np.int32(label)
            return torch.tensor(X_char, dtype=torch.long), torch.tensor(X_bigram, dtype=torch.long), torch.tensor(y, dtype=torch.long)


if __name__ == '__main__':

    df = pd.read_parquet('./dataset/T24_dga.parquet')

    dic_builder = BigramDictionaryBuilder(df)
    bigram_dict = dic_builder.build_dictionary()
    print('dict count:',len(bigram_dict)) 
    dataset = DatasetProcessor(df, label_col='label', bigram_to_index=bigram_dict)

    X_char, X_bigram, y = dataset[0]

    print("도메인 원본:", df.loc[0, "domain"])
    print("전처리된 X char 길이:", X_char.shape)
    print("전처리된 X bigram 길이:", X_bigram.shape)
    print("전처리된 X char:", X_char[-20:].tolist())
    print("전처리된 X bigram:", X_bigram[-20:].tolist())
    print("라벨 y:", y.item())