import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import load_dataset, CustomDSSeparate, custom_collate_fn
from fusion_models import IntermediateFusion2
from model import GRUClassifier
from train import train, DEVICE


def main():
    train_X_video, train_y_video, train_ids_video = load_dataset("train", "vit_face")
    train_X_audio, train_y_audio, train_ids_audio = load_dataset("train", "electra")
    train_X_text, train_y_text, train_ids_text = load_dataset("train", "bert32")

    dev_X_video, dev_y_video, dev_ids_video = load_dataset("devel", "vit_face")
    # dev_X_audio, dev_y_audio, dev_ids_audio = load_dataset("devel", "electra")
    dev_X_text, dev_y_text, dev_ids_text = load_dataset("devel", "bert32")

    test_X_video, test_y_video, test_ids_video = load_dataset("test", "vit_face")
    # test_X_audio, test_y_audio, test_ids_audio = load_dataset("test", "electra")
    test_X_text, test_y_text, test_ids_text = load_dataset("test", "bert32")

    train_ds_if = CustomDSSeparate(train_X_video, train_X_text, train_y_video, train_ids_video, device=DEVICE)
    dev_ds_if = CustomDSSeparate(dev_X_video, dev_X_text, dev_y_video, dev_ids_video, device=DEVICE)
    test_ds_if = CustomDSSeparate(test_X_video, test_X_text, test_y_video, test_ids_video, device=DEVICE)

    train_loader_if = DataLoader(train_ds_if, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    dev_loader_if = DataLoader(dev_ds_if, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    # test_loader_if = DataLoader(test_ds_if, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    model_video = GRUClassifier(input_dim=train_X_video.shape[-1], gru_dim=64, num_gru_layers=2, hidden_size=16, bidirectional=False)
    # model_audio = GRUClassifier(input_dim=train_X_audio.shape[-1], gru_dim=64, num_gru_layers=2, hidden_size=16, bidirectional=False)
    model_text = GRUClassifier(input_dim=train_X_text.shape[-1], gru_dim=64, num_gru_layers=2, hidden_size=16, bidirectional=False)

    model_if = IntermediateFusion2([model_video, model_text], 32, 1, 16)

    train_labels_sep = train_loader_if.dataset.y
    pos_weight = torch.sum(torch.tensor(train_labels_sep) == 0).float() / torch.sum(
        torch.tensor(train_labels_sep) == 1).float()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(lr=0.0005, params=model_if.parameters())

    best_m_if, best_uar_if = train(
        model_if,
        train_loader_if,
        dev_loader_if,
        loss_fn,
        10,
        2,
        optimizer
    )


if __name__ == '__main__':
    main()
