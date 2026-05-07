"""
src/evaluation/silero_baseline.py

Runs Silero VAD on exp2 and exp3 test sets.
No interactive plots - all figures saved silently to PDF.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend - no popup windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SILERO_DIR = ROOT / "outputs" / "silero_baseline"
SILERO_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 256
TARGET_SR   = 8000
SILERO_SR   = 16000


def load_silero():
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True,
    )
    model.eval()
    print("  Silero VAD loaded.")
    return model, utils


def upsample_to_16k(windows: np.ndarray) -> np.ndarray:
    common = gcd(TARGET_SR, SILERO_SR)
    up     = SILERO_SR // common
    down   = TARGET_SR // common
    out = []
    for w in windows:
        out.append(resample_poly(w, up, down).astype(np.float32))
    return np.array(out, dtype=np.float32)


def run_silero(model, windows_16k: np.ndarray,
               threshold: float = 0.5) -> np.ndarray:
    preds = []
    model.reset_states()
    with torch.no_grad():
        for w in windows_16k:
            t    = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
            prob = model(t, SILERO_SR).item()
            preds.append(1 if prob >= threshold else 0)
    return np.array(preds, dtype=np.int64)


def evaluate_experiment(silero_model, experiment: str) -> dict:
    splits_dir = ROOT / "data" / "splits" / experiment
    x_path     = splits_dir / "X_test.npy"
    y_path     = splits_dir / "y_test.npy"

    if not x_path.exists():
        print(f"  [skip] {experiment} test set not found.")
        return None

    print(f"\n  Evaluating on: {experiment}")
    X_test = np.load(str(x_path))
    y_test = np.load(str(y_path))
    print(f"  Test set: {X_test.shape}")

    print("  Upsampling to 16kHz...")
    X_16k = upsample_to_16k(X_test)

    print("  Running Silero inference...")
    preds = run_silero(silero_model, X_16k)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec  = recall_score(y_test, preds, average='macro', zero_division=0)
    f1   = f1_score(y_test, preds, average='macro', zero_division=0)
    f1pc = f1_score(y_test, preds, average=None, zero_division=0)

    print(f"  Accuracy : {acc:.4f}  F1: {f1:.4f}")

    return {
        "experiment": experiment,
        "model":      "SileroVAD",
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_macro":   round(f1,   4),
        "f1_noise":   round(float(f1pc[0]), 4),
        "f1_speech":  round(float(f1pc[1]), 4),
        "latency_ms": "N/A",
        "size_kb":    "~1800",
        "n_params":   "~1M",
        "y_test":     y_test,
        "preds":      preds,
    }


def plot_confusion_matrices(results):
    valid = [r for r in results if r is not None]
    n     = len(valid)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Silero VAD — Confusion Matrices",
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
    plt.close()   # close silently - no popup
    print(f"  Saved: {path}")


def main():
    print(f"\n{'='*55}")
    print(f"  Silero VAD Baseline Evaluation")
    print(f"{'='*55}")

    silero_model, _ = load_silero()

    experiments = ["exp2_clean_musan", "exp3_alaw_musan"]
    results     = []

    for exp in experiments:
        r = evaluate_experiment(silero_model, exp)
        if r is not None:
            results.append(r)

    if not results:
        print("No test sets found.")
        return

    # save CSV
    csv_data = [{k: v for k, v in r.items()
                 if k not in ("y_test", "preds")}
                for r in results]
    df       = pd.DataFrame(csv_data)
    csv_path = SILERO_DIR / "silero_results.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nSaved: {csv_path}")
    print(df[["experiment", "accuracy", "f1_macro"]].to_string(index=False))

    plot_confusion_matrices(results)

    print(f"\n{'='*55}")
    print(f"  Silero evaluation complete.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()