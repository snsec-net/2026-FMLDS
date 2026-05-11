import torch
from torch import nn
import torch.nn.functional as F
import math
from torchinfo import summary

class Embedding(nn.Module):
    def __init__(self,V1, V2, d_model) :
        super().__init__()
        self.char_embedding = nn.Embedding(V1, d_model, padding_idx=0)
        self.bigram_embedding = nn.Embedding(V2, d_model, padding_idx=0)

        # 초기화
        torch.nn.init.xavier_normal_(self.char_embedding.weight)
        torch.nn.init.xavier_normal_(self.bigram_embedding.weight)

    def forward(self, char_input, bigram_input) :
        return self.char_embedding(char_input), self.bigram_embedding(bigram_input)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        P_E = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        _2i = torch.arange(0, d_model, step= 2, dtype=torch.float)

        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model)) 

        P_E[:, 0::2] = torch.sin(pos * div_term)
        P_E[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', P_E.unsqueeze(0))
        
    def forward(self, x) :
        seq_len = x.size(1)

        pe_slice = self.pe[:, :seq_len, :]

        return x + pe_slice

class CNNFFN(nn.Module) :
    def __init__(self, d_model, input_cnn, k, n_c, padding):
        super().__init__()
        self.cnn = nn.Conv1d(input_cnn, n_c, kernel_size=k, stride=1, padding=padding)
        self.ffn = nn.Linear(n_c, d_model)

        # 초기화
        torch.nn.init.xavier_normal_(self.cnn.weight)
        torch.nn.init.xavier_normal_(self.ffn.weight)

    def forward(self, x) :
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = F.relu(x)

        x = x.transpose(1, 2)
        x = self.ffn(x)

        return x
    
class ModifiedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, k, n_c) :
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cnn_ffn = CNNFFN(d_model, d_model, k, n_c, padding='same')
        self.norm2 = nn.LayerNorm(d_model)

        # 초기화
        # torch.nn.init.xavier_normal_(self.mha.in_proj_weight)

    def forward(self, x) :
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.cnn_ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class HybridModifiedTransformer(nn.Module) :
    def __init__(self, num_classes = 1, d_model = 128, V1 = 256, V2 = 2000, L1 = 49, L2 = 48, n_heads = 8, 
                 k = 3, n_c_char = 256, n_c_bigram = 64, n_transformer_layers = 12) :
        super().__init__()
        self.embedding = Embedding(V1, V2, d_model)
        self.pos_encoder_char = PositionalEncoding(d_model, L1)
        self.pos_encoder_bigram = PositionalEncoding(d_model, L2)

        # Character-level 경로
        self.char_transformer_layers = nn.ModuleList([
            ModifiedTransformerBlock(d_model, n_heads, k, n_c_char)
            for _ in range(n_transformer_layers)
        ])

        # Bigram-level 경로
        self.bigram_block = CNNFFN(d_model, d_model, k, n_c_bigram, padding=0)

        self.linear = nn.Linear(d_model * 2, num_classes)
        # self.final_activation = nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)

    def forward(self, char_input, bigram_input) :

        # 임베딩 및 PE 적용
        E1_raw, E2_raw = self.embedding(char_input, bigram_input)
        H_char = self.pos_encoder_char(E1_raw)
        H_bigram = self.pos_encoder_bigram(E2_raw)

        # Character-level 경로
        for layer in self.char_transformer_layers :
            H_char = layer(H_char)

        # Bigram-level 경로
        H_bigram = self.bigram_block(H_bigram)

        # 특징 선택 및 Concatenation
        # char_feature = H_char[:, -1, :]
        char_feature = H_char[:, 0, :]
        bigram_feature = H_bigram.max(dim=1).values # max pooling
        # bigram_feature = H_bigram.mean(dim=1) # mean pooling

        feature_vector = torch.cat((char_feature, bigram_feature), dim=1)

        output = self.linear(feature_vector)

        # return self.final_activation(output)
        return output


if __name__ == '__main__':

    # model = HybridModifiedTransformer(128, 256, 2000, 49, 48)
    # summary(model, input_size=[(1, 49), (1,48)], dtypes=[torch.long, torch.long])

    # model = CNNFFN(128, 128, 3, 256, 'same')
    # summary(model, input_size=(1, 49, 128), dtypes=[torch.float])

    model = HybridModifiedTransformer(num_classes=1, n_transformer_layers=6)
    summary(model, input_size=[(1, 49), (1,48)], dtypes=[torch.long, torch.long])