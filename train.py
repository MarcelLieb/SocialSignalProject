import os

import numpy as np
import torch
from sklearn.metrics import recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_unimodal_data, CustomDS, DATA_DIR, load_dataset
from model import GRUClassifier
from dataset import load_csv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def train(model, train_loader, dev_loader, loss_fn, num_epochs, patience, optimizer,
          model_cp_file=f'{DATA_DIR}/day2/model_checkpoints/fau_gru.pt'):
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
            X = X.to(DEVICE)
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
            torch.save(model.state_dict(), model_cp_file)
        else:
            es_counter += 1
            if es_counter > patience:
                print('Early stopping.')
                break
    model.load_state_dict(torch.load(model_cp_file, weights_only=True))
    return model, best_uar

def main(
        features='xhubert_raw',
        batch_size=4,
        lr=0.005,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        undersample_negative=0.1,
        gru_dim=32,
        num_gru_layers=2,
        hidden_size=16,
        bidirectional=True,
        num_epochs=10,
        patience=1,
):
    train_X, train_y, train_ids = load_dataset("train", features, undersample_negative=undersample_negative)
    dev_X, dev_y, dev_ids = load_dataset("devel", features)

    train_ds = CustomDS(train_X, train_y, train_ids, device=DEVICE)
    dev_ds = CustomDS(dev_X, dev_y, dev_ids, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    model = GRUClassifier(input_dim=train_X.shape[-1], gru_dim=gru_dim, num_gru_layers=num_gru_layers, hidden_size=hidden_size, bidirectional=bidirectional)
    model = model.to(DEVICE)
    pos_weight = torch.sum(torch.tensor(train_y) == 0).float() / torch.sum(
        torch.tensor(train_y) == 1).float()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(lr=lr, params=model.parameters(), betas=betas, weight_decay=weight_decay)

    model, best_uar = train(model, train_loader, dev_loader, loss_fn, num_epochs=num_epochs, patience=patience, optimizer=optimizer)

    return model, best_uar


if __name__ == '__main__':
    main()


