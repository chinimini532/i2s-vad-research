"""
src/training/optuna_tune.py

Optuna hyperparameter search for all four models.

What this does:
  Automatically tries different combinations of:
  - Learning rate
  - Batch size
  - Dropout rate
  - Number of conv filters (CNN1D)
  - Embedding dim (Transformer)

  For each combination it trains briefly (10 epochs)
  and records validation accuracy. After N trials it
  picks the best combination and saves it.

Best hyperparameters saved to:
  outputs/{EXPERIMENT}/stats/best_params.json

After finding best params, train.py uses them
automatically if best_params.json exists.

Run AFTER preprocessing and splitting.
Run BEFORE full training.

Usage:
  python src/training/optuna_tune.py
  (change EXPERIMENT in config.py first)
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import (
    CFG, EXPERIMENT, SPLITS, STATS_DIR, SEED, print_config
)
from src.models.cnn1d           import CNN1D
from src.models.wavenet_small   import WaveNetSmall
from src.models.ecapa_vad       import ECAPAVAD
from src.models.transformer_vad import TransformerVAD

# ─── Optuna config ────────────────────────────────────────────────────────────
N_TRIALS       = 30    # number of hyperparameter combinations to try
TUNE_EPOCHS    = 10    # epochs per trial (short, just to compare)
TUNE_PATIENCE  = 3     # early stopping during tuning


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    X_train = torch.tensor(
        np.load(str(SPLITS / "X_train.npy")), dtype=torch.float32)
    y_train = torch.tensor(
        np.load(str(SPLITS / "y_train.npy")), dtype=torch.long)
    X_val   = torch.tensor(
        np.load(str(SPLITS / "X_val.npy")),   dtype=torch.float32)
    y_val   = torch.tensor(
        np.load(str(SPLITS / "y_val.npy")),   dtype=torch.long)
    return X_train, y_train, X_val, y_val


def quick_train(model, X_train, y_train, X_val, y_val,
                lr, batch_size, device):
    """Train for TUNE_EPOCHS and return best val accuracy."""
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds     = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(TUNE_EPOCHS):
        # train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        # validate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_val), 512):
                X_b = X_val[i:i+512].to(device)
                p   = model(X_b).argmax(1).cpu()
                all_preds.append(p)
        preds   = torch.cat(all_preds)
        val_acc = (preds == y_val).float().mean().item()

        if val_acc > best_val_acc + 1e-4:
            best_val_acc   = val_acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= TUNE_PATIENCE:
                break

    return best_val_acc


def tune_model(model_name, X_train, y_train, X_val, y_val, device):
    """Run Optuna search for one model."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Installing optuna...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "optuna", "-q"], check=True)
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n  Tuning {model_name} ({N_TRIALS} trials)...")

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        dropout    = trial.suggest_float("dropout", 0.1, 0.5)

        # build model with trial params
        if model_name == "CNN1D":
            model = CNN1D(num_classes=2)
            # patch dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = dropout

        elif model_name == "WaveNetSmall":
            model = WaveNetSmall(num_classes=2)
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = dropout

        elif model_name == "ECAPAVAD":
            model = ECAPAVAD(num_classes=2)
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = dropout

        elif model_name == "TransformerVAD":
            embed_dim = trial.suggest_categorical(
                "embed_dim", [32, 64, 128])
            num_heads = trial.suggest_categorical(
                "num_heads", [2, 4])
            # ensure embed_dim divisible by num_heads
            if embed_dim % num_heads != 0:
                embed_dim = num_heads * (embed_dim // num_heads)
            model = TransformerVAD(
                num_classes=2,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        val_acc = quick_train(
            model, X_train, y_train, X_val, y_val,
            lr=lr, batch_size=batch_size, device=device
        )
        return val_acc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    best["best_val_acc"] = study.best_value
    print(f"  Best val acc : {study.best_value:.4f}")
    print(f"  Best params  : {best}")
    return best


def main():
    print_config()
    print(f"\n  Optuna hyperparameter search")
    print(f"  Trials per model : {N_TRIALS}")
    print(f"  Epochs per trial : {TUNE_EPOCHS}")

    torch.manual_seed(SEED)
    device = get_device()
    print(f"  Device : {device}")

    X_train, y_train, X_val, y_val = load_data()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    models_to_tune = [
        "CNN1D",
        "WaveNetSmall",
        "ECAPAVAD",
        "TransformerVAD",
    ]

    all_best_params = {}

    for model_name in models_to_tune:
        best = tune_model(
            model_name, X_train, y_train, X_val, y_val, device)
        all_best_params[model_name] = best

    # save
    params_path = STATS_DIR / "best_params.json"
    with open(str(params_path), "w") as f:
        json.dump(all_best_params, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Optuna search complete")
    print(f"  Best params saved to: {params_path}")
    print(f"{'='*55}")

    for name, params in all_best_params.items():
        print(f"\n  {name}:")
        for k, v in params.items():
            print(f"    {k}: {v}")

    print(f"\nNext: python src/training/train.py")
    print(f"train.py will automatically use these best params.")


if __name__ == "__main__":
    main()