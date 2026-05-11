# reproduction of character level based detection of DGA Domain Names's MIT model
# originally implemented in TensorFlow/Keras, re-implemented here in PyTorch

import torch
import torch.nn as nn
from torchinfo import summary

class MIT(nn.Module):
    def __init__(self):
        super(MIT, self).__init__()# 사용자 정의 클래스 MIT 클래스의 생성자가, 본인인 MIT 클래스의 부모 클래스인 nn.Module의 초기화 메서드를 호출하여 초기화한다.
        # 임베딩 층의 입력으로 사용하기 위해서 입력 시퀀스의 각 단어들은 모두 정수 인코딩이 되어있어야 한다. # embedding.py에서 import torch 이전까지가 정수 인코딩을 하는 전처리 과정이다.
        self.embedding = nn.Embedding(num_embeddings=128, embedding_dim=128) # 전자: ASCII code가 0~127이므로 128 # 후자: 임베딩할 차원이 128
        self.conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dense = nn.Linear(in_features=64, out_features=1)

        # weights initialization (Xavier)
        torch.nn.init.xavier_normal_(self.embedding.weight)
        torch.nn.init.xavier_normal_(self.conv.weight)
        torch.nn.init.xavier_normal_(self.lstm.weight_ih_l0) # AttributeError: 'LSTM' object has no attribute 'weight'
        torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, x):
        # x shape: (batch_size, 75)
        
        x = self.embedding(x) # x shape: (batch_size, 75, 128)
        
        x = x.permute(0, 2, 1) # Conv1d는 (batch_size, channels, sequence_length) 형태를 요구하므로 차원 변경 # x shape: (batch_size, 128, 75)
        x = self.conv(x) # x shape: (batch_size, 128, 75)
        
        x = self.max_pool(x) # x shape: (batch_size, 128, 38)
        
        x = x.permute(0, 2, 1) # LSTM은 (batch_size, sequence_length, features) 형태를 요구하므로 차원 변경 # x shape: (batch_size, 38, 128)
        _, (h_n, _) = self.lstm(x) # outputs, (hidden state, cell state) = lstm(inputs)
        x = h_n.squeeze(0) # 텐서의 차원 중 크기가 1인 차원을 제거 # x shape: (batch_size, 64)
        x = torch.sigmoid(self.dense(x)) # x shape: (batch_size, 1)
        
        return x


if __name__ == '__main__':
    model = MIT() # 모델 인스턴스화
    summary(model, input_size=(100,75), dtypes=[torch.long], depth=5) # input_size는 (batch_size, sequence_length) 형태로 지정

