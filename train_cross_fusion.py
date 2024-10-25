import os
from datetime import datetime

import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from dataset import load_dataset, CustomDS, DATA_DIR, CustomDSSeparate
from fusion_models import IntermediateFusion2
from model import GRUClassifier, EnsembleModel, load_checkpoint
from train import DEVICE, train, MIN_SAVE_SCORE, get_predictions


def main(
        k_folds=5,
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
    torch.manual_seed(42)
    train_X_1, train_y_1, train_ids_1 = load_dataset("train", features[0], undersample_negative)
    train_X_2, train_y_2, train_ids_2 = load_dataset("train", features[1], undersample_negative)

    dev_X_1, dev_y_1, dev_ids_1 = load_dataset("devel", features[0])
    dev_X_2, dev_y_2, dev_ids_2 = load_dataset("devel", features[1])

    train_ds_if = CustomDSSeparate(train_X_1, train_X_2, train_y_1, train_ids_1, device=DEVICE)
    dev_ds_if = CustomDSSeparate(dev_X_1, dev_X_2, dev_y_1, dev_ids_1, device=DEVICE)
    concat_ds = ConcatDataset([train_ds_if, dev_ds_if])

    scores = []
    models = []
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_idx, dev_idx) in enumerate(kfold.split(concat_ds)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_idx)

        train_loader = DataLoader(concat_ds, batch_size=batch_size, sampler=train_subsampler)
        dev_loader = DataLoader(concat_ds, batch_size=batch_size, sampler=dev_subsampler)

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

        pos_weight = torch.sum(torch.tensor(train_y_1) == 0).float() / torch.sum(
                torch.tensor(train_y_1) == 1).float()

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(lr=lr, params=model_if.parameters(), betas=betas, weight_decay=weight_decay)
        print(f"Training fold {fold + 1}/{k_folds}")
        model, best_uar = train(
            model_if, train_loader, dev_loader, loss_fn,
            num_epochs=num_epochs, patience=patience, optimizer=optimizer,
            trial=trial,
        )
        models.append(model)
        scores.append(best_uar)

    final_score = sum(scores) / len(scores)
    model = EnsembleModel(models).cpu()

    save_path = f'{DATA_DIR}/day4/model_checkpoints'
    directory = os.path.join(save_path, f'{features}')
    os.makedirs(directory, exist_ok=True)

    if final_score > MIN_SAVE_SCORE:
        test_X_1, test_y_1, test_ids_1 = load_dataset("test", features[0])
        test_X_2, test_y_2, test_ids_2 = load_dataset("test", features[1])

        test_ds_if = CustomDSSeparate(test_X_1, test_X_2, test_y_1, test_ids_1, device=DEVICE)
        test_loader_if = DataLoader(test_ds_if, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

        pred = (get_predictions(model, test_loader_if) > 0.5).astype(int)
        submission_df = pd.DataFrame({'ID': test_ids_1, 'humor': pred})
        # Get current time
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        submission_df.to_csv(f"Fusion_{features[0]}_{features[1]}_{now}_{int(best_uar_if * 10_000)}.csv", index=False)

    return model, final_score


if __name__ == '__main__':
    main()
