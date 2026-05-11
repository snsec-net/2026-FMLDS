import torch
from torch import nn
from torchinfo import summary

class NYU_og(nn.Module) :
    def __init__(self, n_classes, input_length, input_dim, conv_kernels, fc_neurons) :
        super(NYU_og, self).__init__()

        dimension = int((input_length - 96) / 27 * conv_kernels)

        self.NYU_og = nn.Sequential(
            nn.Conv1d(input_dim, conv_kernels, kernel_size=7, padding=0), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(conv_kernels, conv_kernels, kernel_size=7, padding=0), nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(conv_kernels, conv_kernels, kernel_size=3, padding=0), nn.ReLU(),
            nn.Conv1d(conv_kernels, conv_kernels, kernel_size=3, padding=0), nn.ReLU(),
            nn.Conv1d(conv_kernels, conv_kernels, kernel_size=3, padding=0), nn.ReLU(),
            nn.Conv1d(conv_kernels, conv_kernels, kernel_size=7, padding=0), nn.ReLU(),
            nn.MaxPool1d(3)
        )

        self.fc1 = nn.Sequential(nn.Linear(dimension, fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(fc_neurons, fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(fc_neurons, n_classes)

        # 가중치 초기화
        if conv_kernels == 256 and fc_neurons == 1024 :
            self._create_weights(mean=0.0, std=0.05)
        elif conv_kernels == 1024 and fc_neurons == 2048 :
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean, std) :
        for module in self.modules() :
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) :
                module.weight.data.normal_(mean, std)

    def forward(self, x) :
        x = self.NYU_og(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class NYU_DGA(nn.Module) :
    def __init__(self, n_classes=1, voca_size=128, embedding_dim=128, conv_kernels=128, fc_neurons=64) :
        super().__init__()

        self.embedding = nn.Embedding(voca_size, embedding_dim)

        # Conv1
        self.conv1 = nn.Conv1d(embedding_dim, conv_kernels, kernel_size=3, padding=1)
        self.thresholded_relu1 = nn.Threshold(threshold=1e-6, value=0)
        self.pool1 = nn.MaxPool1d(2, padding=1) # 128, 75

        # Conv2
        self.conv2 = nn.Conv1d(conv_kernels, conv_kernels, kernel_size=2, padding='same')
        self.thresholded_relu2 = nn.Threshold(threshold=1e-6, value=0)
        self.pool2 = nn.MaxPool1d(2)

        # Fully connected
        self.dimension = conv_kernels * 19
        self.fc = nn.Linear(self.dimension, fc_neurons)
        self.thresholded_relu3 = nn.Threshold(threshold=1e-6, value=0)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(64, n_classes)

        # weights initialization (Xavier)
        torch.nn.init.xavier_normal_(self.embedding.weight)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.xavier_normal_(self.out.weight)
        
        # self.theta = 1e-6

    # def thresholded_relu(self, x) :
    #     return torch.clamp(x - self.theta, min=0)
    
    def forward(self, x) :
        x = self.embedding(x)
        x = x.transpose(1,2)

        x = self.conv1(x)
        x = self.thresholded_relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.thresholded_relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.thresholded_relu3(x)
        x = self.dropout(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x
    

if __name__ == '__main__':
    
    # NYUlarge = NYU_og(2, 1014, 70, 1024, 2048)
    # print(NYUlarge)

    model = NYU_DGA(1)
    summary(model, input_size=(100, 75), dtypes=[torch.long])