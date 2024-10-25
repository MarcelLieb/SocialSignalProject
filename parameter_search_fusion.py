import optuna
import torch

from train_cross_fusion import main as train_model

FEATURES = ("vit_face", "bert32")
DATABASE = ""


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_int("batch_size", 2, 32)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    betas = (trial.suggest_float("beta1", 0.1, 0.9), trial.suggest_float("beta2", 0.1, 0.9))
    weight_decay = trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True)
    undersample_negative = trial.suggest_float("undersample_negative", 0.1, 1.0, step=0.05)
    gru_dim = trial.suggest_int("gru_dim", 4, 256)
    num_gru_layers = trial.suggest_int("num_gru_layers", 1, 6)
    hidden_size = trial.suggest_int("hidden_size", 4, 256)
    use_checkpoint = trial.suggest_categorical("use_checkpoint", [True, False])
    model, score = train_model(
        features=FEATURES,
        batch_size=batch_size,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        undersample_negative=undersample_negative,
        num_gru_layers=num_gru_layers,
        gru_mm_dim=gru_dim,
        hidden_size=hidden_size,
        num_epochs=10,
        patience=1,
        ensemble_count=0,
        trial=trial
    )

    return score


def main():
    study = optuna.create_study(
        study_name=f"fusion_{FEATURES[0]}_{FEATURES[1]}",
        storage=DATABASE,
        load_if_exists=True,
        directions=["maximize"],
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
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

    study.enqueue_trial({
        "batch_size": 4,
        "lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 1e-12,
        "undersample_negative": 0.1,
        "gru_dim": 256,
        "num_gru_layers": 2,
        "hidden_size": 32,
        "bidirectional": False,

    }, skip_if_exists=True)

    study.optimize(objective, n_trials=100, catch=(torch.cuda.OutOfMemoryError, RuntimeError), gc_after_trial=True)


if __name__ == "__main__":
    main()
