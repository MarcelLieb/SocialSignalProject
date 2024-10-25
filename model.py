import os

import torch.nn as nn
import torch

from dataset import DATA_DIR


def load_checkpoint(features, count=1):
    checkpoint_path = os.path.join(DATA_DIR, "day4", 'model_checkpoints', features)
    checkpoints = os.listdir(checkpoint_path)
    checkpoints.sort(reverse=True)
    checkpoints = checkpoints[:count]

    models = []
    for checkpoint in checkpoints:
        path = os.path.join(checkpoint_path, checkpoint)
        data = torch.load(path)
        model = GRUClassifier(
            **data["settings"]
        )
        model.load_state_dict(data["model"])
        models.append(model)

    if len(models) == 1:
        return models[0]
    return models


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, gru_dim=32, num_gru_layers=2, hidden_size=16, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.features = nn.GRU(input_size=input_dim, hidden_size=gru_dim, num_layers=num_gru_layers,
                               batch_first=True, bidirectional=bidirectional)
        self.gru_dim = gru_dim * (2 if bidirectional else 1)
        self.classification_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.gru_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, X, return_hidden=False):
        X = X.to(self.features.all_weights[0][0].device)
        x, _ = self.features(X)
        if return_hidden:
            return x
        x = x[:, -1]
        logits = self.classification_head(x)
        return logits.squeeze(-1)


class EnsembleModel(nn.Module):
    def __init__(self, models: list[nn.Module], hidden_en_dim=16):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.projs = nn.ModuleList([nn.Linear(model.gru_dim, hidden_en_dim) for model in models]) if isinstance(models[0], GRUClassifier) else [None for _ in range(len(models))]
        self.gru_dim = hidden_en_dim

    def forward(self, X, return_hidden=False):
        out = []
        for module, proj in zip(self.models, self.projs):
            if return_hidden:
                out.append(
                    proj(
                        module(X, return_hidden=True)
                    )
                )
            else:
                out.append(module(X))
        return torch.stack(out).mean(dim=0)
