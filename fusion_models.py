from torch import nn
import torch
from torch.nn import ModuleList

from model import GRUClassifier, EnsembleModel


class IntermediateFusion(nn.Module):
    def __init__(self, models, gru_dim=32, num_gru_layers=1, hidden_size=16):
        super().__init__()
        self.models = ModuleList(models)
        total_dim = sum([model.gru_dim for model in models])
        self.proj = nn.Linear(total_dim, gru_dim)
        self.gru_mm = nn.GRU(gru_dim, gru_dim, num_layers=num_gru_layers, batch_first=True,
                             bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(gru_dim, hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, Xs, return_hidden=False):
        # Assume that the two sequences are given as tuples. This is crucial for compatibility with the methods implemented above
        Xs = [X.float().to(self.linear1.weight.device) for X in Xs]
        gru_res = [self.models[i].features(X)[1] for i, X in enumerate(Xs)]
        x = torch.cat(gru_res, dim=2)
        x = self.proj(x)
        x = self.activation(x)
        x, _ = self.gru_mm(x)
        x = x[-1, :]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x.squeeze()


class IntermediateFusion2(nn.Module):
    def __init__(self, models: list[GRUClassifier | EnsembleModel], gru_mm_dim, num_gru_mm_layers, hidden_size):
        super().__init__()
        self.gru1 = models[0]
        self.gru2 = models[1]
        self.proj = nn.Linear(models[0].gru_dim + models[1].gru_dim, gru_mm_dim)
        self.gru_mm = nn.GRU(gru_mm_dim, gru_mm_dim, num_layers=num_gru_mm_layers, batch_first=True,
                             bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(gru_mm_dim, hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, X):
        # Assume that the two sequences are given as tuples. This is crucial for compatibility with the methods implemented above
        X1, X2 = X
        x1 = X1.float()
        x2 = X2.float()
        x1 = self.gru1(x1, return_hidden=True)
        x2 = self.gru2(x2, return_hidden=True)
        x = torch.cat([x1, x2], dim=2)
        x = self.proj(x)
        x = self.activation(x)
        x, _ = self.gru_mm(x)
        x = x[:, -1]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x.squeeze()
