import torch.nn as nn
import torch
from torch.nn import ModuleList
class LateFusion(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.modules = ModuleList(models)

    def forward(self, X, lengths=None):
        sum_res = torch.zeros(len(self.modules))
        for module in self.modules:
            sum_res = torch.add(sum_res, module(X))
        sum_res /= len(self.modules)
        return sum_res

class IntermediateFusion(nn.Module):
    def __init__(self, models, gru_dim = 32, num_gru_layers = 1, hidden_size = 16):
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

    def forward(self, Xs):
        # Assume that the two sequences are given as tuples. This is crucial for compatibility with the methods implemented above
        Xs = [X.float().to(self.linear1.weight.device) for X in Xs]
        gru_res = [self.models[i].features(X)[1] for i,X in enumerate(Xs)]
        x = torch.cat(gru_res, dim=2)
        x = self.proj(x)
        x = self.activation(x)
        x, _ = self.gru_mm(x)
        x = x[-1,:]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x.squeeze()