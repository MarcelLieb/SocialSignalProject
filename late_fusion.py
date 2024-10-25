from train import main as train_function, get_predictions, train
import numpy as np
import os
import torch
from sklearn.metrics import recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from dataset import CustomDS, DATA_DIR, load_dataset
from model import GRUClassifier
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_SAVE_SCORE = 0.74
'''
Extra klasse für late Fusion Tests
'''



'''
Late fusion durch mehrheitsvoting
'''
def late_fusion(predictions):
    num_pred = predictions.__len__()
    new_pred = []
    for i in range(predictions[0][0].__len__()):
        score = 0
        for number in range(num_pred):
            score += predictions[number][0][i]*(predictions[number][1]-0.5)
        score = (score > 0).astype(int)
        #score = 1 if score > 0 else np.round(score/num_pred)
        new_pred.append(score)
    return new_pred
'''
kleine modifikationen und tests bei der ausführung
'''
def main():
    m = [
        main2(features='faus', batch_size=6, lr=0.00014408848065597334, betas=(0.12892714768277658, 0.7582463568636874),
              weight_decay=1.7164724740648662e-10, undersample_negative=0.2, gru_dim=139, num_gru_layers=4,
              hidden_size=10, bidirectional=False, num_epochs=15, patience=2, normal=True),
        main2(features='xhubert_raw', batch_size=32, lr=3.1562325330190156e-05,
              betas=(0.46835399559322194, 0.46835399559322194), weight_decay=9.201883367771638e-05,
              undersample_negative=0.45, gru_dim=251, num_gru_layers=4, hidden_size=215, bidirectional=False,
              num_epochs=15, patience=2, normal=True),
        main2(features='vit_face', batch_size=23, lr=0.0004169459777972417,
              betas=(0.1777221814757832, 0.714318394120134), weight_decay=9.992630759984277e-10,
              undersample_negative=0.3, gru_dim=87, num_gru_layers=4, hidden_size=159, bidirectional=False,
              num_epochs=15, patience=2, normal=True),
        #main2(features='vit_face', batch_size=9, lr=9.996875379471902e-05,
         #     betas=(0.4578019959891491, 0.8653615831633331), weight_decay=	1.6306672650179436e-08,
          #    undersample_negative=0.3, gru_dim=82, num_gru_layers=5, hidden_size=154, bidirectional=False,
           #   num_epochs=15, patience=2, normal=True),
        main2(features='vit_face', batch_size=22, lr=0.000163626740268518,
              betas=(0.30045638338013136, 0.7493999757153151), weight_decay=0.0071193145290880284,
              undersample_negative=0.4, gru_dim=153, num_gru_layers=5, hidden_size=180, bidirectional=True,
              num_epochs=15, patience=2, normal=True),
        main2(features='bert32', batch_size=1, lr=0.00016365005132096136, betas=(0.136277726703837, 0.27071843516073035),
              weight_decay=2.025735473109792e-05, undersample_negative=0.1, gru_dim=253, num_gru_layers=6,
              hidden_size=189, bidirectional=False, num_epochs=15, patience=2, normal=True)
    ]
    #m.append(main2(features='self_wav', batch_size=4, lr=0.005, betas=(0.9, 0.999), weight_decay=0.01, undersample_negative=0.1, gru_dim=32, num_gru_layers=2, hidden_size=16, bidirectional=False, num_epochs=15, patience=1, normal = True))
    #m.append(main2(features='self_wav', batch_size=4, lr=0.005, betas=(0.9, 0.999), weight_decay=0.01, undersample_negative=0.1, gru_dim=32, num_gru_layers=2, hidden_size=16, bidirectional=False, num_epochs=15, patience=1, normal = True))
    #m.append(main2(features='self_wav', batch_size=4, lr=0.005, betas=(0.9, 0.999), weight_decay=0.01, undersample_negative=0.1, gru_dim=32, num_gru_layers=2, hidden_size=16, bidirectional=False, num_epochs=15, patience=1, normal = True))


    features = ['faus', 'xhubert_raw', 'vit_face', 'vit_face', 'bert32']
    #models = [main2(features=feat, normal=False) for feat in features]
    #[models.append(main2(features=feat, normal=True)) for feat in features]
    #print(models.__len__())
    list_pred = []

    i = 0
    ids = []
    for model_ret, best_uar in m:
        # load test
        test_X, test_y, test_ids = load_dataset("test", features[i%len(features)])
        ids = test_ids
        i+=1
        train_ds = CustomDS(test_X, test_y, test_ids, device=DEVICE)
        test_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        list_pred.append((get_predictions(model_ret, test_loader), best_uar))
    new_pred_late = late_fusion(list_pred)
    dic = {'ID': ids, 'humor': new_pred_late}
    df_n_p = pd.DataFrame(dic)
    df_n_p.to_csv("latetest.csv", index=False)

'''
ausführung des standarttrainings
'''
def main2(
        features='self_wav',
        batch_size=4,
        lr=0.005,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        undersample_negative=0.1,
        gru_dim=32,
        num_gru_layers=2,
        hidden_size=16,
        bidirectional=False,
        num_epochs=15,
        patience=2,
        normal = True
):
    insert = "train" if normal else "devel"
    train_X, train_y, train_ids = load_dataset(insert, features, undersample_negative=undersample_negative)
    insert = "devel" if normal else "train"
    dev_X, dev_y, dev_ids = load_dataset(insert, features)

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