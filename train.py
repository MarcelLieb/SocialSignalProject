import os

import numpy as np
import torch
from sklearn.metrics import recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_unimodal_data, CustomDS, DATA_DIR, load_dataset, custom_collate_fn, CustomDSSeparate
from fusion_models import IntermediateFusion
from dataset import CustomDS, DATA_DIR, load_dataset
from model import GRUClassifier
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_SAVE_SCORE = 0.74


def get_predictions(model, data_loader):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            X, _, _ = batch
            pred = model(X).squeeze().cpu().detach().numpy().tolist()
            if isinstance(pred, float):
                pred = [pred]
            predictions.extend(pred)
    return np.array(predictions)


def train(model, train_loader, dev_loader, loss_fn, num_epochs, patience, optimizer):
    es_counter = 0
    best_uar = -1
    best_state_dict = None
    dev_labels = dev_loader.dataset.y.cpu()
    losses = []

    for epoch in range(num_epochs):
        print(f'Training epoch {epoch + 1}')
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            X, y, _ = batch
            y = y.to(DEVICE)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Loss: {np.round(np.mean(losses), 4)}')
        # eval 
        print('Evaluation')
        dev_preds = get_predictions(model, dev_loader)
        # binarize dev_preds 
        dev_preds = (dev_preds > 0.5).astype(np.int32)
        dev_uar = recall_score(dev_labels, dev_preds, average='macro')
        print(f'UAR: {dev_uar}')
        if dev_uar > best_uar:
            es_counter = 0
            best_uar = dev_uar
            best_state_dict = model.state_dict()
            # torch.save(model.state_dict(), model_cp_file)
        else:
            es_counter += 1
            if es_counter > patience:
                print('Early stopping.')
                break
    model.load_state_dict(best_state_dict)
    return model, best_uar

def main(
        features=['faus','egemaps'],
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
):
    train_X = []
    dev_X = []
    train_y = None
    train_ids = None
    dev_y = None
    dev_ids = None
    for feature in features:
        t_train_X, train_y, train_ids = load_dataset("train", feature, undersample_negative=undersample_negative)
        t_dev_X, dev_y, dev_ids = load_dataset("devel", feature)
        train_X.append(t_train_X)
        dev_X.append(t_dev_X)

    train_ds = CustomDSSeparate(train_X, train_y, train_ids, device=DEVICE)
    dev_ds = CustomDSSeparate(dev_X, dev_y, dev_ids, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    models = [GRUClassifier(input_dim=train_X[i].shape[-1], gru_dim=gru_dim, num_gru_layers=num_gru_layers,
                          hidden_size=hidden_size, bidirectional=bidirectional) for i in range(len(features))]
    for i in range(len(models)):
        models[i] = models[i].to(DEVICE)
    model = IntermediateFusion(models)
    model = model.to(DEVICE)
    pos_weight = torch.sum(torch.tensor(train_y) == 0).float() / torch.sum(
        torch.tensor(train_y) == 1).float()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(lr=lr, params=model.parameters(), betas=betas, weight_decay=weight_decay)

    model, best_uar = train(
        model, train_loader, dev_loader, loss_fn,
        num_epochs=num_epochs, patience=patience, optimizer=optimizer
    )
    save_path = f'{DATA_DIR}/day4/model_checkpoints'
    directory = os.path.join(save_path, f'{features}')
    os.makedirs(directory, exist_ok=True)

    if best_uar > MIN_SAVE_SCORE:
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

    return model, best_uar


if __name__ == '__main__':
    main()


