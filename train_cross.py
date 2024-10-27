import os

import optuna
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from dataset import load_dataset, CustomDS, DATA_DIR
from model import GRUClassifier, EnsembleModel
from train import DEVICE, train, MIN_SAVE_SCORE


def main(
        k_folds=5,
        features='faus',
        batch_size=4,
        lr=0.005,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        undersample_negative=0.2,
        gru_dim=64,
        num_gru_layers=2,
        hidden_size=16,
        bidirectional=True,
        num_epochs=15,
        patience=2,
        trial: optuna.Trial=None,
):
    torch.manual_seed(42)
    train_X, train_y, train_ids = load_dataset("train", features, undersample_negative=undersample_negative)
    dev_X, dev_y, dev_ids = load_dataset("devel", features)

    train_ds = CustomDS(train_X, train_y, train_ids, device=DEVICE)
    dev_ds = CustomDS(dev_X, dev_y, dev_ids, device=DEVICE)

    concat_ds = ConcatDataset([train_ds, dev_ds])

    scores = []
    models = []
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_idx, dev_idx) in enumerate(kfold.split(concat_ds)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_idx)

        train_loader = DataLoader(concat_ds, batch_size=batch_size, sampler=train_subsampler)
        dev_loader = DataLoader(concat_ds, batch_size=batch_size, sampler=dev_subsampler)

        model = GRUClassifier(input_dim=train_X.shape[-1], gru_dim=gru_dim, num_gru_layers=num_gru_layers,
                              hidden_size=hidden_size, bidirectional=bidirectional)
        model = model.to(DEVICE)
        pos_weight = torch.sum(torch.tensor(train_y) == 0).float() / torch.sum(
            torch.tensor(train_y) == 1).float()

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(lr=lr, params=model.parameters(), betas=betas, weight_decay=weight_decay)
        print(f"Training fold {fold + 1}/{k_folds}")
        model, best_uar, _ = train(
            model, train_loader, dev_loader, loss_fn,
            num_epochs=num_epochs, patience=patience, optimizer=optimizer,
            trial=trial,
        )
        models.append(model)
        scores.append(best_uar)

    final_score = sum(scores) / len(scores)
    model = EnsembleModel(models).cpu()
    print(f"Final score: {final_score}")

    save_path = f'{DATA_DIR}/day4/model_checkpoints'
    directory = os.path.join(save_path, f'{features}')
    os.makedirs(directory, exist_ok=True)

    if final_score > MIN_SAVE_SCORE:
        torch.save({
            "model": model.state_dict(),
            "settings": {
                "input_dim": train_X.shape[-1],
                "gru_dim": gru_dim,
                "num_gru_layers": num_gru_layers,
                "hidden_size": hidden_size,
                "bidirectional": bidirectional
            }
        }, os.path.join(directory, f'gru_{int(best_uar*10_000)}.pt'))

    return model, final_score


if __name__ == '__main__':
    main()
