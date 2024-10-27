import os
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold, GroupKFold, LeavePGroupsOut, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from dataset import load_dataset, CustomDS, DATA_DIR, CustomDSSeparate, custom_collate_fn
from fusion_models import IntermediateFusion2
from model import GRUClassifier, EnsembleModel, load_checkpoint
from train import DEVICE, train, MIN_SAVE_SCORE, get_predictions


def main(
        k_folds=3,
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
    train_X_1, train_y_1, train_ids_1 = load_dataset("train", features[0])
    train_X_2, train_y_2, train_ids_2 = load_dataset("train", features[1])

    dev_X_1, dev_y_1, dev_ids_1 = load_dataset("devel", features[0])
    dev_X_2, dev_y_2, dev_ids_2 = load_dataset("devel", features[1])

    y = torch.concatenate([torch.tensor(train_y_1), torch.tensor(dev_y_1)])

    train_ds = CustomDSSeparate(train_X_1, train_X_2, train_y_1, train_ids_1, device=DEVICE)
    dev_ds = CustomDSSeparate(dev_X_1, dev_X_2, dev_y_1, dev_ids_1, device=DEVICE)
    concat_ds = ConcatDataset([train_ds, dev_ds])

    ids = train_ids_1 + dev_ids_1
    coaches = [iden.split("_")[0] for iden in ids]
    unique_coaches = list(set(coaches))
    coach_group = [unique_coaches.index(coach) for coach in coaches]

    scores = []
    models = []
    epochs = []
    kfold = LeavePGroupsOut(n_groups=6)

    for fold, (train_idx, dev_idx) in enumerate(kfold.split(concat_ds, y, groups=coach_group)):
        train_labels = [(i, y[i]) for i in train_idx]
        train_labels = np.array(train_labels).astype(int)
        sub_index = train_labels[train_labels[:, 1] == 0][:, 0]
        pos_index = train_labels[train_labels[:, 1] == 1][:, 0]
        gen = np.random.default_rng(42)
        sub_index = gen.choice(sub_index, int(undersample_negative * len(sub_index)), replace=False)
        train_idx = np.concatenate([sub_index, pos_index])

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

        pos_weight = torch.sum(y[train_idx] == 0).float() / torch.sum(
                y[train_idx] == 1).float()

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(lr=lr, params=model_if.parameters(), betas=betas, weight_decay=weight_decay)
        print(f"Training fold {fold + 1}/{kfold.get_n_splits(groups=coach_group)}")
        model, best_uar, epoch = train(
            model_if, train_loader, dev_loader, loss_fn,
            num_epochs=num_epochs, patience=patience, optimizer=optimizer,
            trial=trial,
        )
        models.append(model)
        scores.append(best_uar)
        epochs.append(epoch + 1)
        if trial is not None:
            trial.report(sum(scores) / len(scores), fold)
        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned()

    final_score = sum(scores) / len(scores)
    model = EnsembleModel(models).cpu()

    save_path = f'{DATA_DIR}/day4/model_checkpoints'
    directory = os.path.join(save_path, f'{features}')
    os.makedirs(directory, exist_ok=True)
    print(f"Final UAR: {final_score}")

    if final_score > MIN_SAVE_SCORE:
        """
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

        sub_index = np.where(np.array(train_y_1) == 0)[0]
        pos_index = np.where(np.array(train_y_1) == 1)[0]
        gen = np.random.default_rng(42)
        sub_index = gen.choice(sub_index, int(undersample_negative * len(sub_index)), replace=False)
        train_idx = np.concatenate([sub_index, pos_index])

        sub_index = np.where(np.array(dev_y_1) == 0)[0]
        pos_index = np.where(np.array(dev_y_1) == 1)[0]
        gen = np.random.default_rng(42)
        sub_index = gen.choice(sub_index, int(undersample_negative * len(sub_index)), replace=False)
        dev_idx = np.concatenate([sub_index, pos_index])

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_idx)

        train_loader = DataLoader(concat_ds, batch_size=batch_size, sampler=train_subsampler)
        dev_loader = DataLoader(concat_ds, batch_size=batch_size, sampler=dev_subsampler)
        pos_weight = torch.sum(y == 0).float() / torch.sum(
            y == 1).float()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(lr=lr, params=model_if.parameters(), betas=betas, weight_decay=weight_decay)

        final_epochs = int(round(sum(epochs) / len(epochs)))
        model, best_uar, _ = train(
            model_if, train_loader, dev_loader, loss_fn,
            num_epochs=final_epochs, patience=patience, optimizer=optimizer,
            trial=trial,
        )
        """
        full_loader = DataLoader(concat_ds, batch_size=batch_size, shuffle=False)
        pred = (get_predictions(model, full_loader) > 0.5).astype(int)
        uar = recall_score(y, pred, average='macro')
        print(f'UAR on full dataset: {uar}')

        test_X_1, test_y_1, test_ids_1 = load_dataset("test", features[0])
        test_X_2, test_y_2, test_ids_2 = load_dataset("test", features[1])

        test_ds_if = CustomDSSeparate(test_X_1, test_X_2, test_y_1, test_ids_1, device=DEVICE)
        test_loader_if = DataLoader(test_ds_if, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

        pred = (get_predictions(model, test_loader_if) > 0.5).astype(int)
        submission_df = pd.DataFrame({'ID': test_ids_1, 'humor': pred})
        # Get current time
        now = datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        submission_df.to_csv(f"Fusion_{features[0]}_{features[1]}_{now}_{int(final_score * 10_000)}.csv", index=False)

    return model, final_score


if __name__ == '__main__':
    main()
