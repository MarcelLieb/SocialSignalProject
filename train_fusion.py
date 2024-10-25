import datetime

import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import load_dataset, CustomDSSeparate, custom_collate_fn
from fusion_models import IntermediateFusion2
from model import GRUClassifier, load_checkpoint, EnsembleModel
from train import train, DEVICE, get_predictions


def main(
        features=("vit_face", "bert32"),
        batch_size=4,
        lr=0.005,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        undersample_negative=0.2,
        gru_mm_dim=64,
        num_gru_layers=2,
        hidden_size=16,
        hidden_size_en=16,
        num_epochs=10,
        patience=2,
        ensemble_count=0,
        trial: optuna.Trial = None
):
    train_X_1, train_y_1, train_ids_1 = load_dataset("train", features[0], undersample_negative)
    train_X_2, train_y_2, train_ids_2 = load_dataset("train", features[1], undersample_negative)

    dev_X_1, dev_y_1, dev_ids_1 = load_dataset("devel", features[0])
    dev_X_2, dev_y_2, dev_ids_2 = load_dataset("devel", features[1])

    train_ds_if = CustomDSSeparate(train_X_1, train_X_2, train_y_1, train_ids_1, device=DEVICE)
    dev_ds_if = CustomDSSeparate(dev_X_1, dev_X_2, dev_y_1, dev_ids_1, device=DEVICE)

    train_loader_if = DataLoader(train_ds_if, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader_if = DataLoader(dev_ds_if, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    model_1 = load_checkpoint(features[0], count=ensemble_count)
    if ensemble_count == 1:
        model_1 = [model_1]
    if len(model_1) == 0:
        model_1 = [GRUClassifier(input_dim=train_X_1.shape[-1])]

    model_2 = load_checkpoint(features[1], count=ensemble_count)
    if ensemble_count == 1:
        model_2 = [model_2]
    if len(model_2) == 0:
        model_2 = [GRUClassifier(input_dim=train_X_2.shape[-1])]
    model_1 = EnsembleModel(model_1, hidden_size_en)
    model_2 = EnsembleModel(model_2, hidden_size_en)

    model_if = IntermediateFusion2([model_1, model_2], gru_mm_dim, num_gru_layers, hidden_size)
    model_if = model_if.to(DEVICE)

    train_labels_sep = train_loader_if.dataset.y
    pos_weight = torch.sum(train_labels_sep == 0).float() / torch.sum(train_labels_sep == 1).float()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(lr=lr, params=model_if.parameters(), weight_decay=weight_decay, betas=betas)

    best_m_if, best_uar_if = train(
        model_if,
        train_loader_if,
        dev_loader_if,
        loss_fn,
        num_epochs,
        patience,
        optimizer,
        trial=trial,
    )

    if best_uar_if > 0.8:
        test_X_1, test_y_1, test_ids_1 = load_dataset("test", features[0])
        test_X_2, test_y_2, test_ids_2 = load_dataset("test", features[1])

        test_ds_if = CustomDSSeparate(test_X_1, test_X_2, test_y_1, test_ids_1, device=DEVICE)
        test_loader_if = DataLoader(test_ds_if, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

        pred = (get_predictions(best_m_if, test_loader_if) > 0.5).astype(int)
        submission_df = pd.DataFrame({'ID': test_ids_1, 'humor': pred})
        # Get current time
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        submission_df.to_csv(f"Fusion_{features[0]}_{features[1]}_{now}_{int(best_uar_if*10_000)}.csv", index=False)

    return best_m_if, best_uar_if


if __name__ == '__main__':
    main()
