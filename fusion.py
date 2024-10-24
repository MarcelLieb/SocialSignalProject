import torch
from torch import nn


class IFModel(nn.Module):

    def __init__(self, input_dim1, input_dim2, gru1_dim=32, gru2_dim=32, num_gru_layers1=1,
                 num_gru_layers2=1, input_mm=64, gru_mm_dim=32, num_gru_mm_layers=1, hidden_size=16):
        super(IFModel, self).__init__()
        self.gru1 = nn.GRU(input_dim1, gru1_dim, num_layers=num_gru_layers1, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_dim2, gru2_dim, num_layers=num_gru_layers2, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(gru1_dim + gru2_dim, gru_mm_dim)
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
        x1, _ = self.gru1(x1)
        x2, _ = self.gru2(x2)
        x = torch.cat([x1, x2], dim=2)
        x = self.proj(x)
        x = self.activation(x)
        x, _ = self.gru_mm(x)
        x = x[:, -1]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x