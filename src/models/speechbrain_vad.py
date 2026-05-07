"""
src/models/speechbrain_vad.py

Fine-tunes SpeechBrain's pretrained ECAPA-TDNN for VAD.

What this does:
  1. Loads pretrained ECAPA-TDNN from SpeechBrain
     (trained on VoxCeleb for speaker recognition)
  2. Freezes the feature extractor layers
  3. Adds a small VAD classification head
  4. Fine-tunes on our A-law processed data (exp3)
  5. Saves model to outputs/exp3_alaw_musan/models/

Why this matters for the paper:
  Shows that even a strong pretrained model benefits
  from hardware-aware fine-tuning on A-law data.

Saves:
  outputs/exp3_alaw_musan/models/SpeechBrainVAD_best.pt
  outputs/exp3_alaw_musan/stats/SpeechBrainVAD_history.json
"""

import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# use exp3 config for fine-tuning
import importlib
import src.training.config as cfg_module

# force exp3
cfg_module.EXPERIMENT  = "exp3_alaw_musan"
from src.training.config import SPLITS, MODELS_DIR, STATS_DIR, SEED

MODELS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# training config
MAX_EPOCHS  = 30
BATCH_SIZE  = 64
LR          = 0.0001    # lower LR for fine-tuning pretrained model
PATIENCE    = 7


class SpeechBrainVAD(nn.Module):
    """
    Pretrained ECAPA-TDNN encoder + VAD classification head.

    The encoder is frozen (pretrained weights kept fixed).
    Only the classification head is trained.
    This is called transfer learning.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # load pretrained ECAPA-TDNN from SpeechBrain
        print("  Loading pretrained SpeechBrain ECAPA-TDNN...")
        from speechbrain.pretrained import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},   # load on CPU first
        )
        print("  Pretrained model loaded.")

        # freeze encoder weights - we don't update these
        for param in self.encoder.parameters():
            param.requires_grad = False

        # the encoder outputs 192-dim embeddings
        # add a small classification head on top
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 256) raw audio windows at 8kHz
        """
        # SpeechBrain encoder expects (batch, time) at 16kHz
        # upsample from 8kHz to 16kHz
        from scipy.signal import resample_poly
        x_np = x.cpu().numpy()
        upsampled = []
        for w in x_np:
            up = resample_poly(w, 2, 1).astype(np.float32)
            upsampled.append(up)
        x_16k = torch.tensor(
            np.array(upsampled), dtype=torch.float32
        ).to(next(self.classifier.parameters()).device)

        # get embeddings from pretrained encoder
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(x_16k)
            # embeddings shape: (batch, 1, 192)
            embeddings = embeddings.squeeze(1)   # (batch, 192)

        # classify
        return self.classifier(embeddings)


def count_parameters(model: nn.Module) -> int:
    # only count trainable params (classification head only)
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)


def load_data():
    print("\nLoading exp3 splits...")
    X_train = np.load(str(SPLITS / "X_train.npy"))
    y_train = np.load(str(SPLITS / "y_train.npy"))
    X_val   = np.load(str(SPLITS / "X_val.npy"))
    y_val   = np.load(str(SPLITS / "y_val.npy"))
    X_test  = np.load(str(SPLITS / "X_test.npy"))
    y_test  = np.load(str(SPLITS / "y_test.npy"))

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    to_t = lambda x: torch.tensor(x, dtype=torch.float32)
    to_l = lambda y: torch.tensor(y, dtype=torch.long)

    return (to_t(X_train), to_l(y_train),
            to_t(X_val),   to_l(y_val),
            to_t(X_test),  to_l(y_test))


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


def main():
    print(f"\n{'='*55}")
    print(f"  SpeechBrain ECAPA-TDNN Fine-tuning")
    print(f"  Experiment: exp3_alaw_musan")
    print(f"{'='*55}")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # use smaller subset for fine-tuning to save time
    # 50K samples is enough for fine-tuning
    MAX_FINETUNE = 50_000
    if len(X_train) > MAX_FINETUNE:
        idx     = torch.randperm(len(X_train))[:MAX_FINETUNE]
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Using {MAX_FINETUNE:,} samples for fine-tuning")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # build model
    model    = SpeechBrainVAD(num_classes=2).to(device)
    n_params = count_parameters(model)
    print(f"  Trainable params: {n_params:,} (classification head only)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False)

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience_count = 0
    save_path      = MODELS_DIR / "SpeechBrainVAD_best.pt"
    history        = {"train_loss": [], "train_acc": [],
                      "val_loss":   [], "val_acc":   []}
    start = time.time()

    print(f"\n  Training classification head...")
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, True)
        vl_loss, vl_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, False)

        scheduler.step(vl_loss)

        history["train_loss"].append(round(tr_loss, 6))
        history["train_acc"].append(round(tr_acc,   4))
        history["val_loss"].append(round(vl_loss,   6))
        history["val_acc"].append(round(vl_acc,     4))

        improved = vl_loss < (best_val_loss - 1e-4)
        marker   = " *" if improved else ""
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {vl_loss:.4f}/{vl_acc:.4f}{marker}")

        if improved:
            best_val_loss  = vl_loss
            best_val_acc   = vl_acc
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_name":  "SpeechBrainVAD",
                "experiment":  "exp3_alaw_musan",
                "model_state": model.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
                "n_params":    n_params,
                "history":     history,
            }, str(save_path))
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch}")
                break

    elapsed = time.time() - start

    # save history
    hist_path = STATS_DIR / "SpeechBrainVAD_history.json"
    with open(str(hist_path), "w") as f:
        json.dump(history, f, indent=2)

    # update training stats CSV
    stats = {
        "experiment":      "exp3_alaw_musan",
        "model":           "SpeechBrainVAD",
        "n_params":        n_params,
        "best_val_loss":   round(best_val_loss, 6),
        "best_val_acc":    round(best_val_acc,  4),
        "epochs_trained":  len(history["train_loss"]),
        "training_time_s": round(elapsed, 1),
        "use_alaw":        True,
        "use_musan":       True,
        "fraction":        1.0,
        "lr":              LR,
        "batch_size":      BATCH_SIZE,
    }

    csv_path = STATS_DIR / "training_stats.csv"
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
        # remove old SpeechBrainVAD row if exists
        df = df[df["model"] != "SpeechBrainVAD"]
        df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
    else:
        df = pd.DataFrame([stats])
    df.to_csv(str(csv_path), index=False)

    print(f"\n{'='*55}")
    print(f"  SpeechBrain fine-tuning complete")
    print(f"  Best val acc  : {best_val_acc:.4f}")
    print(f"  Training time : {elapsed:.1f}s")
    print(f"  Saved         : {save_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()