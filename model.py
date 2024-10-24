import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, gru_dim=32, num_gru_layers=2, hidden_size=16, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_dim, num_layers=num_gru_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.classification_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(gru_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, X, lengths=None):
        _, h_n = self.gru(X)  # NL, h_out
        h_n = h_n[-1, :]
        logits = self.classification_head(h_n)
        return logits.squeeze(-1)