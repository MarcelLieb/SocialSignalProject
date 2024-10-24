import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import datetime

from dataset import load_dataset, CustomDS
from model import EnsembleModel, load_checkpoint
from train import get_predictions, train, DEVICE


def main():
    features = "bert32"

    models = load_checkpoint(features, count=5)
    ensemble_model = EnsembleModel(models).to(DEVICE)

    train_X, train_y, train_ids = load_dataset("train", features, undersample_negative=0.1)
    dev_X, dev_y, dev_ids = load_dataset("devel", features)

    train_ds = CustomDS(train_X, train_y, train_ids, device=DEVICE)
    dev_ds = CustomDS(dev_X, dev_y, dev_ids, device=DEVICE)
    train_ds = ConcatDataset([train_ds, dev_ds])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=4, shuffle=False)

    pos_weight = torch.sum(torch.tensor(train_y) == 0).float() / torch.sum(
        torch.tensor(train_y) == 1).float()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(lr=1e-5, params=ensemble_model.parameters(), weight_decay=1e-4)

    ensemble_model, uar = train(ensemble_model, train_loader, dev_loader, loss_fn, 4, 2, optimizer)

    X, y, ids = load_dataset("test", features)

    test_ds = CustomDS(X, y, ids)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    pred = (get_predictions(ensemble_model, test_loader) > 0.5).astype(int)

    submission_df = pd.DataFrame({'ID': ids, 'humor': pred})
    # Get current time
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    submission_df.to_csv(f"{features}_ensemble_{now}.csv", index=False)


if __name__ == '__main__':
    main()
