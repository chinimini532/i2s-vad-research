"""
src/training/train.py

Trains all four models for the current experiment.
Reads config from src/training/config.py
Saves models and stats to outputs/{EXPERIMENT}/

To run a different experiment:
  1. Change EXPERIMENT in src/training/config.py
  2. Run python src/data/preprocess.py
  3. Run python src/data/split.py
  4. Run python src/training/train.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import (
    CFG, EXPERIMENT, SPLITS, MODELS_DIR, STATS_DIR, SEED,
    print_config
)
from src.models.cnn1d           import CNN1D
from src.models.wavenet_small   import WaveNetSmall
from src.models.ecapa_vad       import ECAPAVAD
from src.models.transformer_vad import TransformerVAD


# ─── Device ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  CPU (no GPU detected)")
    return device


# ─── Data ─────────────────────────────────────────────────────────────────────
def load_splits():
    print("\nLoading splits...")
    X_train = np.load(str(SPLITS / "X_train.npy"))
    y_train = np.load(str(SPLITS / "y_train.npy"))
    X_val   = np.load(str(SPLITS / "X_val.npy"))
    y_val   = np.load(str(SPLITS / "y_val.npy"))
    X_test  = np.load(str(SPLITS / "X_test.npy"))
    y_test  = np.load(str(SPLITS / "y_test.npy"))

    print(f"  Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test.shape}")

    to_t = lambda x: torch.tensor(x, dtype=torch.float32)
    to_l = lambda y: torch.tensor(y, dtype=torch.long)

    return (to_t(X_train), to_l(y_train),
            to_t(X_val),   to_l(y_val),
            to_t(X_test),  to_l(y_test))


def make_loaders(X_train, y_train, X_val, y_val):
    bs = CFG["batch_size"]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=bs, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ─── Train / validate one epoch ───────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = total_correct = total_n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            if train:
                loss.backward()
                optimizer.step()
            preds         = logits.argmax(1)
            total_loss   += loss.item() * len(y_b)
            total_correct += (preds == y_b).sum().item()
            total_n      += len(y_b)

    return total_loss / total_n, total_correct / total_n


# ─── Train one model ──────────────────────────────────────────────────────────
def train_model(name, model, train_loader, val_loader, device):
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}")
    print(f"  Model      : {name}")
    print(f"  Parameters : {n_params:,}")
    print(f"{'='*55}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience_count = 0
    save_path      = MODELS_DIR / f"{name}_best.pt"
    history        = {"train_loss": [], "train_acc": [],
                      "val_loss":   [], "val_acc":   []}
    start = time.time()

    for epoch in range(1, CFG["max_epochs"] + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True)
        vl_loss, vl_acc = run_epoch(
            model, val_loader,   optimizer, criterion, device, train=False)

        scheduler.step(vl_loss)

        history["train_loss"].append(round(tr_loss, 6))
        history["train_acc"].append(round(tr_acc,   4))
        history["val_loss"].append(round(vl_loss,   6))
        history["val_acc"].append(round(vl_acc,     4))

        improved = vl_loss < (best_val_loss - 1e-4)
        marker   = " *" if improved else ""
        print(f"  Epoch {epoch:3d}/{CFG['max_epochs']} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {vl_loss:.4f}/{vl_acc:.4f}{marker}")

        if improved:
            best_val_loss  = vl_loss
            best_val_acc   = vl_acc
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_name":  name,
                "experiment":  EXPERIMENT,
                "model_state": model.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
                "n_params":    n_params,
                "history":     history,
                "cfg":         CFG,
            }, str(save_path))
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print(f"\n  Early stop at epoch {epoch}")
                break

    elapsed = time.time() - start
    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Best val acc  : {best_val_acc:.4f}")
    print(f"  Time          : {elapsed:.1f}s")
    print(f"  Saved         : {save_path}")

    # save history
    hist_path = STATS_DIR / f"{name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "experiment":      EXPERIMENT,
        "model":           name,
        "n_params":        n_params,
        "best_val_loss":   round(best_val_loss, 6),
        "best_val_acc":    round(best_val_acc,  4),
        "epochs_trained":  len(history["train_loss"]),
        "training_time_s": round(elapsed, 1),
        "use_alaw":        CFG["use_alaw"],
        "use_musan":       CFG["use_musan"],
        "fraction":        CFG["fraction"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print_config()
    torch.manual_seed(SEED)
    device = get_device()

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)

    # save test set for evaluation notebook
    with open(str(STATS_DIR / "test_info.json"), "w") as f:
        json.dump({
            "experiment":    EXPERIMENT,
            "X_test_shape":  list(X_test.shape),
            "y_test_shape":  list(y_test.shape),
            "use_alaw":      CFG["use_alaw"],
            "use_musan":     CFG["use_musan"],
        }, f, indent=2)

    models = [
        ("CNN1D",          CNN1D(num_classes=2)),
        ("WaveNetSmall",   WaveNetSmall(num_classes=2)),
        ("ECAPAVAD",       ECAPAVAD(num_classes=2)),
        ("TransformerVAD", TransformerVAD(num_classes=2)),
    ]

    all_stats = []
    for name, model in models:
        stats = train_model(name, model, train_loader, val_loader, device)
        all_stats.append(stats)

    df = pd.DataFrame(all_stats)
    csv_path = STATS_DIR / "training_stats.csv"
    df.to_csv(str(csv_path), index=False)

    print(f"\n{'='*55}")
    print(f"  All models trained  |  Experiment: {EXPERIMENT}")
    print(f"{'='*55}")
    print(df[["model", "n_params", "best_val_acc",
              "epochs_trained", "training_time_s"]].to_string(index=False))
    print(f"\n  Stats : {csv_path}")
    print(f"  Next  : open notebooks/evaluation.ipynb")


if __name__ == "__main__":
    main()