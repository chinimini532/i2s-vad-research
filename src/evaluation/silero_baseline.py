"""
src/evaluation/silero_baseline.py

Runs Silero VAD on the test sets from exp2 and exp3.
Silero is a pretrained VAD model — no training needed.
We just run inference and compare against our trained models.

What this proves:
  Silero trained on clean 16kHz data performs worse on
  A-law distorted 8kHz telephony data than our hardware-aware
  trained models. That gap is our paper's contribution.

Saves results to:
  outputs/silero_baseline/silero_results.csv
  outputs/silero_baseline/silero_comparison.pdf
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# output dir
SILERO_DIR = ROOT / "outputs" / "silero_baseline"
SILERO_DIR.mkdir(parents=True, exist_ok=True)

# window size matches our training
WINDOW_SIZE = 256
TARGET_SR   = 8000
SILERO_SR   = 16000   # Silero expects 16kHz


# ─── Load Silero VAD ──────────────────────────────────────────────────────────
def load_silero():
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True,
    )
    model.eval()
    print("  Silero VAD loaded successfully.")
    return model, utils


# ─── Resample windows to 16kHz for Silero ─────────────────────────────────────
def upsample_to_16k(windows: np.ndarray) -> np.ndarray:
    """
    Our windows are 256 samples at 8kHz = 32ms.
    Silero expects 16kHz so we upsample to 512 samples.
    The A-law distortion pattern remains after resampling.
    """
    common = gcd(TARGET_SR, SILERO_SR)
    up     = SILERO_SR // common
    down   = TARGET_SR // common
    upsampled = []
    for w in windows:
        w_up = resample_poly(w, up, down).astype(np.float32)
        upsampled.append(w_up)
    return np.array(upsampled, dtype=np.float32)


# ─── Run Silero on windows ────────────────────────────────────────────────────
def run_silero(model, windows_16k: np.ndarray,
               threshold: float = 0.5) -> np.ndarray:
    """
    Run Silero VAD on each window.
    Returns binary predictions: 1=speech, 0=noise
    """
    preds = []
    model.reset_states()

    with torch.no_grad():
        for w in windows_16k:
            # Silero expects (1, samples) tensor
            t      = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
            prob   = model(t, SILERO_SR).item()
            preds.append(1 if prob >= threshold else 0)

    return np.array(preds, dtype=np.int64)


# ─── Evaluate on one experiment test set ─────────────────────────────────────
def evaluate_experiment(silero_model, experiment: str) -> dict:
    """
    Load test set from one experiment, run Silero, compute metrics.
    """
    splits_dir = ROOT / "data" / "splits" / experiment
    x_path     = splits_dir / "X_test.npy"
    y_path     = splits_dir / "y_test.npy"

    if not x_path.exists():
        print(f"  [skip] {experiment} test set not found.")
        return None

    print(f"\n  Loading test set: {experiment}")
    X_test = np.load(str(x_path))
    y_test = np.load(str(y_path))
    print(f"  Test set: {X_test.shape}  "
          f"speech={y_test.sum()}  noise={(y_test==0).sum()}")

    # upsample to 16kHz for Silero
    print("  Upsampling to 16kHz...")
    X_16k = upsample_to_16k(X_test)

    # run Silero
    print("  Running Silero VAD inference...")
    preds = run_silero(silero_model, X_16k)

    # metrics
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec  = recall_score(y_test, preds, average='macro', zero_division=0)
    f1   = f1_score(y_test, preds, average='macro', zero_division=0)
    f1pc = f1_score(y_test, preds, average=None, zero_division=0)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 macro  : {f1:.4f}")
    print(f"  F1 noise  : {float(f1pc[0]):.4f}")
    print(f"  F1 speech : {float(f1pc[1]):.4f}")

    return {
        "experiment":  experiment,
        "model":       "SileroVAD",
        "accuracy":    round(acc,  4),
        "precision":   round(prec, 4),
        "recall":      round(rec,  4),
        "f1_macro":    round(f1,   4),
        "f1_noise":    round(float(f1pc[0]), 4),
        "f1_speech":   round(float(f1pc[1]), 4),
        "latency_ms":  "N/A",
        "size_kb":     "~1800",
        "n_params":    "~1M",
        "y_test":      y_test,
        "preds":       preds,
    }


# ─── Confusion matrix figure ──────────────────────────────────────────────────
def plot_confusion_matrices(results: list):
    valid  = [r for r in results if r is not None]
    n      = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Silero VAD — Confusion Matrices per Experiment",
                 fontsize=13, fontweight='bold')

    for ax, r in zip(axes, valid):
        cm = confusion_matrix(r["y_test"], r["preds"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Noise', 'Speech'],
                    yticklabels=['Noise', 'Speech'],
                    ax=ax, cbar=False)
        ax.set_title(f'{r["experiment"]}\n'
                     f'Acc={r["accuracy"]:.4f}  F1={r["f1_macro"]:.4f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    path = SILERO_DIR / "silero_confusion_matrices.pdf"
    plt.savefig(str(path), bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  Silero VAD Baseline Evaluation")
    print(f"{'='*55}")

    # load Silero
    silero_model, _ = load_silero()

    # evaluate on all available experiment test sets
    experiments = [
        "exp1_alaw_synthetic",
        "exp2_clean_musan",
        "exp3_alaw_musan",
    ]

    results = []
    for exp in experiments:
        r = evaluate_experiment(silero_model, exp)
        if r is not None:
            results.append(r)

    if not results:
        print("No experiment test sets found. Run training first.")
        return

    # save CSV (without y_test and preds columns)
    csv_data = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("y_test", "preds")}
        csv_data.append(row)

    df       = pd.DataFrame(csv_data)
    csv_path = SILERO_DIR / "silero_results.csv"
    df.to_csv(str(csv_path), index=False)

    print(f"\n{'='*55}")
    print(f"  Silero VAD Results")
    print(f"{'='*55}")
    print(df[["experiment", "accuracy", "f1_macro",
              "f1_noise", "f1_speech"]].to_string(index=False))
    print(f"\nSaved: {csv_path}")

    # confusion matrices
    plot_confusion_matrices(results)

    print(f"\n{'='*55}")
    print(f"  Done. Results saved to outputs/silero_baseline/")
    print(f"  Next: run notebooks/evaluation.ipynb for")
    print(f"  cross-experiment comparison figure.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()