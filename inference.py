import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import datetime

from dataset import load_dataset, DATA_DIR, CustomDS
from model import EnsembleModel
from model import GRUClassifier
from train import get_predictions


def main():
    features = "vit_face"

    checkpoint_path = os.path.join(DATA_DIR, "day4", 'model_checkpoints', features)
    checkpoints = os.listdir(checkpoint_path)
    checkpoints.sort(reverse=True)
    checkpoints = checkpoints[:5]

    models = []
    for checkpoint in checkpoints:
        path = os.path.join(checkpoint_path, checkpoint)
        data = torch.load(path)
        model = GRUClassifier(
            **data["settings"]
        )
        model.load_state_dict(data["model"])
        models.append(model)
    ensemble_model = EnsembleModel(models)

    X, y, ids = load_dataset("test", features)

    test_ds = CustomDS(X, y, ids)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    pred = (get_predictions(ensemble_model, test_loader) > 0).astype(int)

    submission_df = pd.DataFrame({'ID': ids, 'humor': pred})
    # Get current time
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    submission_df.to_csv(f"{features}_ensemble_{now}.csv", index=False)


if __name__ == '__main__':
    main()