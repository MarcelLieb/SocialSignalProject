import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import load_dataset, DATA_DIR, CustomDS
from model import GRUClassifier
from train import get_predictions


def main():
    X, y, ids = load_dataset("test", "faus")
    test_ds = CustomDS(X, y, ids)
    test_loader = DataLoader(test_ds, batch_size=6, shuffle=False)
    model = GRUClassifier(
        input_dim=X.shape[-1],
        gru_dim=139,
        num_gru_layers=4,
        hidden_size=10
    )
    model.load_state_dict(torch.load(f'{DATA_DIR}/day4/model_checkpoints/faus_gru.pt', weights_only=True))
    pred = (get_predictions(model, test_loader) > 0).astype(int)
    submission_df = pd.DataFrame({'ID': ids, 'humor': pred})
    submission_df.to_csv('faus_parameter_test2.csv', index=False)


if __name__ == '__main__':
    main()