import optuna
import torch

from train import main as train_model

FEATURES = "self_wav"
DATABASE = "NO"


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_int("batch_size", 1, 32)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    betas = (trial.suggest_float("beta1", 0.1, 0.9), trial.suggest_float("beta2", 0.1, 0.9))
    weight_decay = trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True)
    undersample_negative = trial.suggest_float("undersample_negative", 0.05, 1.0, step=0.05)
    gru_dim = trial.suggest_int("gru_dim", 4, 256)
    num_gru_layers = trial.suggest_int("num_gru_layers", 1, 6)
    hidden_size = trial.suggest_int("hidden_size", 4, 256)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    model, score = train_model(
        features=FEATURES,
        batch_size=batch_size,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        undersample_negative=undersample_negative,
        gru_dim=gru_dim,
        num_gru_layers=num_gru_layers,
        hidden_size=hidden_size,
        bidirectional=bidirectional,
        num_epochs=10,
        patience=1,
    )

    return score


def main():
    study = optuna.create_study(
        study_name=FEATURES,
        storage=DATABASE,
        load_if_exists=True,
        directions=["maximize"],
        sampler=optuna.samplers.TPESampler(multivariate=True),

    )

    study.enqueue_trial({
        "batch_size": 4,
        "lr": 0.005,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 1e-12,
        "undersample_negative": 0.1,
        "gru_dim": 32,
        "num_gru_layers": 2,
        "hidden_size": 16,
        "bidirectional": False,
    }, skip_if_exists=True)

    study.optimize(objective, n_trials=100, catch=(torch.cuda.OutOfMemoryError, RuntimeError), gc_after_trial=True)


if __name__ == "__main__":
    main()
