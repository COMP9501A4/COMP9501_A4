from unicodedata import bidirectional
import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(self, channel_num, kernal_size, drop_out, task, n_vocabulary, max_length, embedding_wight = None, embedding_size=50):
        super(TextCNN, self).__init__()
        self.task = task
        self.embedding_layer = nn.Embedding(n_vocabulary, embedding_size)
        if embedding_wight is not None:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_wight))
            # self.embedding_layer.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, channel_num, (kernal, embedding_size)),
                nn.LeakyReLU(),
                nn.MaxPool2d((max_length - kernal + 1, 1))
            )
            for kernal in kernal_size
        ])
        self.flatten = nn.Flatten()
        self.drop_out = nn.Dropout(drop_out)
        self.fc = nn.Linear(channel_num * len(kernal_size), 5)
        if self.task == 'regression':
            self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.unsqueeze(1)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.flatten(x)
        x = self.drop_out(x)
        x = self.fc(x)
        if self.task == 'classification':
            x = F.softmax(x, dim=1)
        elif self.task == 'regression':
            x = F.relu(x)
            x = self.fc2(x)
            x = x.squeeze(1)
        return x 

class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, task, n_vocabulary, embedding_wight = None, embedding_size=50):
        super(RNN, self).__init__()
        self.task = task
        self.embedding_layer = nn.Embedding(n_vocabulary, embedding_size)
        if embedding_wight is not None:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_wight))
            self.embedding_layer.weight.requires_grad = False
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 5)
    
    def forward(self, x):
        x = self.embedding_layer(x)
        x, _ = self.lstm(x)
        x = F.relu(self.fc(x))
        x = F.avg_pool2d(x, (x.size(1), 1)).squeeze(1)
        out = self.fc2(x)
        return out
