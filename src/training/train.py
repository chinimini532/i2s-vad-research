"""
src/training/train.py

What this file does (in simple words):
---------------------------------------
Loads the train/val splits from data/splits/
Trains all three models one by one:
  1. CNN1D
  2. WaveNetSmall
  3. ECAPAVAD

For each model:
  - Trains for up to MAX_EPOCHS epochs
  - Checks val loss after every epoch
  - Saves the BEST version of the model (lowest val loss)
  - Stops early if val loss stops improving (early stopping)
  - Saves training stats to outputs/stats/

After all three models finish:
  - Saves a combined stats CSV with all results
  - You can share this CSV for paper writing

Saved files:
  outputs/models/cnn1d_best.pt
  outputs/models/wavenet_best.pt
  outputs/models/ecapa_best.pt
  outputs/stats/all_training_stats.csv
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

# add project root to path so imports work
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.cnn1d        import CNN1D
from src.models.wavenet_small import WaveNetSmall
from src.models.ecapa_vad    import ECAPAVAD

# ─── Configuration ────────────────────────────────────────────────────────────
MAX_EPOCHS    = 50       # maximum epochs per model
BATCH_SIZE    = 64       # samples per batch
LEARNING_RATE = 1e-3     # initial learning rate
PATIENCE      = 7        # early stopping patience (epochs without improvement)
MIN_DELTA     = 1e-4     # minimum improvement to count as progress
SEED          = 42

# ─── Paths ────────────────────────────────────────────────────────────────────
SPLITS      = ROOT / "data"    / "splits"
MODELS_DIR  = ROOT / "outputs" / "models"
STATS_DIR   = ROOT / "outputs" / "stats"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Device setup ─────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  Using CPU (no GPU detected)")
    return device


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_splits() -> tuple:
    """Load train/val/test numpy arrays and convert to tensors."""
    print("\nLoading data splits...")

    X_train = np.load(str(SPLITS / "X_train.npy"))
    y_train = np.load(str(SPLITS / "y_train.npy"))
    X_val   = np.load(str(SPLITS / "X_val.npy"))
    y_val   = np.load(str(SPLITS / "y_val.npy"))
    X_test  = np.load(str(SPLITS / "X_test.npy"))
    y_test  = np.load(str(SPLITS / "y_test.npy"))

    print(f"  Train : {X_train.shape}  labels: {y_train.shape}")
    print(f"  Val   : {X_val.shape}    labels: {y_val.shape}")
    print(f"  Test  : {X_test.shape}   labels: {y_test.shape}")

    # convert to float32 tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loaders(X_train, y_train, X_val, y_val) -> tuple:
    """Wrap tensors in DataLoaders for batched training."""
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader


# ─── Training one epoch ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion,
                    device) -> tuple:
    """Run one full pass over the training data."""
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds          = logits.argmax(dim=1)
        total_loss    += loss.item() * len(y_batch)
        total_correct += (preds == y_batch).sum().item()
        total_samples += len(y_batch)

    avg_loss = total_loss    / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ─── Validation one epoch ─────────────────────────────────────────────────────
def validate(model, loader, criterion, device) -> tuple:
    """Run one full pass over the validation data."""
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            preds          = logits.argmax(dim=1)
            total_loss    += loss.item() * len(y_batch)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

    avg_loss = total_loss    / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ─── Train one model ──────────────────────────────────────────────────────────
def train_model(model_name: str, model: nn.Module,
                train_loader, val_loader,
                device) -> dict:
    """
    Full training loop for one model.
    Returns a dict of training stats for this model.
    """
    print(f"\n{'='*55}")
    print(f"  Training: {model_name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Max epochs: {MAX_EPOCHS}  |  Patience: {PATIENCE}")
    print(f"{'='*55}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # learning rate scheduler: reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience_count = 0
    save_path      = MODELS_DIR / f"{model_name}_best.pt"

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc,   4))
        history["val_loss"].append(round(val_loss,     6))
        history["val_acc"].append(round(val_acc,       4))

        # ── check if this is the best model so far ────────────────────
        improved = val_loss < (best_val_loss - MIN_DELTA)
        marker   = " *" if improved else ""

        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}{marker}")

        if improved:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            patience_count = 0

            # save best model checkpoint
            torch.save({
                "epoch":          epoch,
                "model_name":     model_name,
                "model_state":    model.state_dict(),
                "val_loss":       val_loss,
                "val_acc":        val_acc,
                "n_params":       n_params,
                "history":        history,
            }, str(save_path))

        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    total_time = time.time() - start_time

    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Best val acc  : {best_val_acc:.4f}")
    print(f"  Training time : {total_time:.1f}s")
    print(f"  Saved to      : {save_path}")

    # ── stats dict for this model ──────────────────────────────────────
    stats = {
        "model":           model_name,
        "n_params":        n_params,
        "best_val_loss":   round(best_val_loss, 6),
        "best_val_acc":    round(best_val_acc,  4),
        "epochs_trained":  len(history["train_loss"]),
        "training_time_s": round(total_time, 1),
        "save_path":       str(save_path),
    }

    # also save history separately
    history_path = STATS_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved : {history_path}")

    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  I2S VAD Research — Model Training")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Max epochs : {MAX_EPOCHS}")
    print(f"  Patience   : {PATIENCE}")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    device = get_device()

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)

    # save test set info for evaluation notebook
    test_info = {
        "X_test_shape": list(X_test.shape),
        "y_test_shape": list(y_test.shape),
    }
    with open(str(STATS_DIR / "test_info.json"), "w") as f:
        json.dump(test_info, f)

    # define models to train
    models = [
        ("cnn1d",    CNN1D(num_classes=2)),
        ("wavenet",  WaveNetSmall(num_classes=2)),
        ("ecapa",    ECAPAVAD(num_classes=2)),
    ]

    all_stats = []

    for model_name, model in models:
        stats = train_model(
            model_name, model,
            train_loader, val_loader,
            device,
        )
        all_stats.append(stats)

    # ── save combined stats CSV ────────────────────────────────────────
    df = pd.DataFrame(all_stats)
    csv_path = STATS_DIR / "all_training_stats.csv"
    df.to_csv(str(csv_path), index=False)

    print(f"\n{'='*55}")
    print(f"  All models trained successfully")
    print(f"{'='*55}")
    print(df.to_string(index=False))
    print(f"\n  Stats saved to : {csv_path}")
    print(f"  Next step      : open notebooks/evaluation.ipynb")


if __name__ == "__main__":
    main()