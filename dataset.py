import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from functools import lru_cache

DATA_DIR = '../../data'
FEATURES_DIR = os.path.join(DATA_DIR, 'features')


# speed up loading time drastically
@lru_cache(maxsize=128)
def load_csv(file):
    return pd.read_csv(file)


@lru_cache(maxsize=2)
def load_gs():
    return load_csv(os.path.join(DATA_DIR, 'gs.csv'))


def load_unimodal_data(label_df, features, undersample_negative=None):
    feature_dir = f'{FEATURES_DIR}/{features}'
    X = []
    y = []
    ids = []
    current_segment = None
    feature_df = None
    if not (undersample_negative is None):
        rng = np.random.default_rng(42)
    for _, row in tqdm(label_df.iterrows()):
        if not (undersample_negative is None) and row.humor == 0:
            if rng.random() <= (1 - undersample_negative):
                continue
        if current_segment != row.segment:
            current_segment = row.segment
            feature_df = load_csv(f'{feature_dir}/{row.coach}/{row.segment}.csv')
        seg_feature_df = feature_df[(feature_df.timestamp >= row.start) & (feature_df.timestamp < row.end)]
        seg_features = seg_feature_df.iloc[:, 2:].values
        seg_features = np.pad(seg_features, ((0, 4 - seg_features.shape[0]), (0, 0)))
        seg_features = np.expand_dims(seg_features, 0)
        X.append(seg_features)
        y.append(row.humor)
        ids.append(row.ID)
    X = np.vstack(X).astype(np.float64)
    y = np.array(y)
    return X, y, ids

@lru_cache(maxsize=128)
def load_dataset(split, features, undersample_negative=None):
    gs_df = load_gs()
    X, y, ids = load_unimodal_data(gs_df[gs_df.partition == split], features,
                                                     undersample_negative=undersample_negative)
    return X, y, ids


class CustomDS(Dataset):

    def __init__(self, X, y, ids, device: str = 'cpu'):
        super(CustomDS, self).__init__()
        self.device = device
        self.X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.y = torch.from_numpy(y.astype(np.float32)).to(self.device)
        self.ids = ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.ids[item]


class CustomDSSeparate(Dataset):

    def __init__(self, X1, X2, y, ids, device: str = 'cpu'):
        super(CustomDSSeparate, self).__init__()
        assert X1.shape[0] == X2.shape[0]
        self.X1 = torch.from_numpy(X1).to(device)
        self.X2 = torch.from_numpy(X2).to(device)
        self.y = torch.tensor(y)
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return (self.X1[item], self.X2[item]), self.y[item], self.ids[item]


def custom_collate_fn(batch):
    x1s = [torch.unsqueeze(b[0][0], 0) for b in batch]
    x2s = [torch.unsqueeze(b[0][1], 0) for b in batch]
    ys = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    x1s = torch.cat(x1s)
    x2s = torch.cat(x2s)
    ys = torch.tensor(ys)
    return (x1s, x2s), ys, ids