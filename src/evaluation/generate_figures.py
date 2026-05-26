"""
src/evaluation/generate_figures.py

Generates all publication-quality figures for Paper 1.
Run from project root:
    python src/evaluation/generate_figures.py

Output: outputs/exp3_alaw_musan/figures/ and outputs/evaluation/figures/
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Output directories ────────────────────────────────────────────────────────
EXP3_FIGS = ROOT / "outputs" / "exp3_alaw_musan" / "figures"
EVAL_FIGS  = ROOT / "outputs" / "evaluation" / "figures"
EXP3_FIGS.mkdir(parents=True, exist_ok=True)
EVAL_FIGS.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

COLORS = {
    "CNN1D":          "#1565C0",
    "WaveNetSmall":   "#2E7D32",
    "ECAPAVAD":       "#C62828",
    "TransformerVAD": "#6A1B9A",
    "Silero":         "#E65100",
    "WebRTC":         "#37474F",
}
MODEL_NAMES = ["CNN1D", "WaveNetSmall", "ECAPAVAD", "TransformerVAD"]
MODEL_LABELS = ["CNN1D", "WaveNet-Small", "ECAPA-VAD", "Transformer-VAD"]


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Training Loss Curves (exp3)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_training_loss():
    print("  Generating Fig 1: Training Loss Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Training and Validation Loss — G.711 A-law Training (exp3)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, name, label in zip(
        axes.flat, MODEL_NAMES, MODEL_LABELS
    ):
        hist_path = ROOT / "outputs" / "exp3_alaw_musan" / "stats" / f"{name}_history.json"
        with open(str(hist_path)) as f:
            h = json.load(f)

        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], color=COLORS[name],
                lw=1.8, label="Train loss")
        ax.plot(epochs, h["val_loss"], color=COLORS[name],
                lw=1.8, ls="--", label="Val loss")

        # mark best epoch
        best_ep = int(np.argmin(h["val_loss"])) + 1
        best_vl = min(h["val_loss"])
        ax.axvline(x=best_ep, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.scatter([best_ep], [best_vl], color=COLORS[name],
                   s=60, zorder=5)
        ax.annotate(f"Best\nEp.{best_ep}",
                    xy=(best_ep, best_vl),
                    xytext=(best_ep + 0.5, best_vl + 0.02),
                    fontsize=8, color="gray")

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = EXP3_FIGS / "fig1_training_loss.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Training Accuracy Curves (exp3)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_training_accuracy():
    print("  Generating Fig 2: Training Accuracy Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Training and Validation Accuracy — G.711 A-law Training (exp3)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, name, label in zip(
        axes.flat, MODEL_NAMES, MODEL_LABELS
    ):
        hist_path = ROOT / "outputs" / "exp3_alaw_musan" / "stats" / f"{name}_history.json"
        with open(str(hist_path)) as f:
            h = json.load(f)

        epochs = range(1, len(h["train_acc"]) + 1)
        ax.plot(epochs, [a * 100 for a in h["train_acc"]],
                color=COLORS[name], lw=1.8, label="Train acc")
        ax.plot(epochs, [a * 100 for a in h["val_acc"]],
                color=COLORS[name], lw=1.8, ls="--", label="Val acc")

        best_ep  = int(np.argmax(h["val_acc"])) + 1
        best_acc = max(h["val_acc"]) * 100
        ax.axvline(x=best_ep, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.annotate(f"{best_acc:.2f}%",
                    xy=(best_ep, best_acc),
                    xytext=(best_ep + 0.5, best_acc - 1.5),
                    fontsize=8, color=COLORS[name],
                    fontweight="bold")

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([90, 100.5])

    plt.tight_layout()
    path = EXP3_FIGS / "fig2_training_accuracy.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
def fig3_confusion_matrices():
    print("  Generating Fig 3: Confusion Matrices...")

    eval_df = pd.read_csv(str(ROOT / "outputs" / "evaluation" / "final_evaluation.csv"))
    exp3_df = eval_df[eval_df["condition"] == "C3_Alaw_NoAlawNorm"]

    # load test labels
    y_test = np.load(str(ROOT / "data" / "splits" / "exp3_alaw_musan" / "y_test.npy"))

    from src.models.cnn1d           import CNN1D
    from src.models.wavenet_small   import WaveNetSmall
    from src.models.ecapa_vad       import ECAPAVAD
    from src.models.transformer_vad import TransformerVAD

    model_classes = {
        "CNN1D": CNN1D, "WaveNetSmall": WaveNetSmall,
        "ECAPAVAD": ECAPAVAD, "TransformerVAD": TransformerVAD
    }

    device = torch.device("cpu")
    X_test = np.load(str(ROOT / "data" / "splits" / "exp3_alaw_musan" / "X_test.npy"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(
        "Confusion Matrices — G.711 A-law Trained Models\n(Test set: 60,000 windows)",
        fontsize=13, fontweight="bold"
    )

    for ax, name, label in zip(axes.flat, MODEL_NAMES, MODEL_LABELS):
        model_path = ROOT / "outputs" / "exp3_alaw_musan" / "models" / f"{name}_best.pt"

        if not model_path.exists():
            ax.set_title(f"{label}\n(model not found)")
            continue

        model = model_classes[name](num_classes=2)
        ckpt  = torch.load(str(model_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                b = torch.tensor(X_test[i:i+512], dtype=torch.float32)
                preds.extend(model(b).argmax(1).numpy())
        preds = np.array(preds)

        cm = confusion_matrix(y_test, preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Noise", "Speech"])
        ax.set_yticklabels(["Noise", "Speech"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(label, fontweight="bold")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]:,})",
                        ha="center", va="center",
                        color="white" if cm_pct[i,j] > 50 else "black",
                        fontsize=10, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = EVAL_FIGS / "fig3_confusion_matrices.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
def fig4_roc_curves():
    print("  Generating Fig 4: ROC Curves...")

    from src.models.cnn1d           import CNN1D
    from src.models.wavenet_small   import WaveNetSmall
    from src.models.ecapa_vad       import ECAPAVAD
    from src.models.transformer_vad import TransformerVAD

    model_classes = {
        "CNN1D": CNN1D, "WaveNetSmall": WaveNetSmall,
        "ECAPAVAD": ECAPAVAD, "TransformerVAD": TransformerVAD
    }

    device  = torch.device("cpu")
    X_test  = np.load(str(ROOT / "data" / "splits" / "exp3_alaw_musan" / "X_test.npy"))
    y_test  = np.load(str(ROOT / "data" / "splits" / "exp3_alaw_musan" / "y_test.npy"))

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, label in zip(MODEL_NAMES, MODEL_LABELS):
        model_path = ROOT / "outputs" / "exp3_alaw_musan" / "models" / f"{name}_best.pt"
        if not model_path.exists():
            continue

        model = model_classes[name](num_classes=2)
        ckpt  = torch.load(str(model_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        probs = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                b = torch.tensor(X_test[i:i+512], dtype=torch.float32)
                p = torch.softmax(model(b), dim=1)[:, 1]
                probs.extend(p.numpy())
        probs = np.array(probs)

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc     = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=COLORS[name], lw=2,
                label=f"{label} (AUC = {roc_auc:.4f})")

    # Silero placeholder
    ax.plot([0, 1], [0, 1], "k--", lw=1.5,
            label="Silero VAD / WebRTC VAD (AUC ≈ 0.50)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — G.711 A-law Trained Models\nvs Off-the-Shelf Baselines",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = EVAL_FIGS / "fig4_roc_curves.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════
def fig5_pr_curves():
    print("  Generating Fig 5: Precision-Recall Curves...")

    from src.models.cnn1d           import CNN1D
    from src.models.wavenet_small   import WaveNetSmall
    from src.models.ecapa_vad       import ECAPAVAD
    from src.models.transformer_vad import TransformerVAD

    model_classes = {
        "CNN1D": CNN1D, "WaveNetSmall": WaveNetSmall,
        "ECAPAVAD": ECAPAVAD, "TransformerVAD": TransformerVAD
    }

    device = torch.device("cpu")
    X_test = np.load(str(ROOT / "data" / "splits" / "exp3_alaw_musan" / "X_test.npy"))
    y_test = np.load(str(ROOT / "data" / "splits" / "exp3_alaw_musan" / "y_test.npy"))

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, label in zip(MODEL_NAMES, MODEL_LABELS):
        model_path = ROOT / "outputs" / "exp3_alaw_musan" / "models" / f"{name}_best.pt"
        if not model_path.exists():
            continue

        model = model_classes[name](num_classes=2)
        ckpt  = torch.load(str(model_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        probs = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                b = torch.tensor(X_test[i:i+512], dtype=torch.float32)
                p = torch.softmax(model(b), dim=1)[:, 1]
                probs.extend(p.numpy())
        probs = np.array(probs)

        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap           = average_precision_score(y_test, probs)

        ax.plot(rec, prec, color=COLORS[name], lw=2,
                label=f"{label} (AP = {ap:.4f})")

    ax.axhline(y=0.5, color="k", ls="--", lw=1.5,
               label="Silero VAD / WebRTC VAD (AP ≈ 0.50)")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — G.711 A-law Trained Models",
                 fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = EVAL_FIGS / "fig5_pr_curves.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Accuracy Comparison (our models vs baselines)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_accuracy_comparison():
    print("  Generating Fig 6: Accuracy Comparison...")

    # Results from paper
    data = {
        "model":     MODEL_LABELS + ["Silero VAD", "WebRTC VAD"],
        "exp2_acc":  [97.18, 99.52, 99.63, 97.67, 49.90, 49.50],
        "exp3_acc":  [98.54, 99.57, 99.65, 97.85, 49.90, 49.50],
    }
    df = pd.DataFrame(data)

    x     = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width/2, df["exp2_acc"], width,
                   label="Clean PCM Training (Baseline)",
                   color="#90CAF9", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, df["exp3_acc"], width,
                   label="G.711 A-law Training (Proposed)",
                   color="#1565C0", edgecolor="black", linewidth=0.5)

    # value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=8, rotation=45)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=8, rotation=45, fontweight="bold")

    # random guessing line
    ax.axhline(y=50, color="red", ls="--", lw=1.5, alpha=0.7,
               label="Random guessing (50%)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(
        "Test Accuracy Comparison: Proposed G.711-Compliant Training\nvs "
        "Clean PCM Baseline and Off-the-Shelf VAD Systems",
        fontweight="bold"
    )
    ax.legend(loc="lower right")
    ax.set_ylim([0, 108])
    ax.grid(True, axis="y", alpha=0.3)

    # separator line between our models and baselines
    ax.axvline(x=3.5, color="gray", ls=":", lw=1.5, alpha=0.7)
    ax.text(3.55, 5, "Off-the-shelf\nbaselines",
            fontsize=9, color="gray", va="bottom")

    path = EVAL_FIGS / "fig6_accuracy_comparison.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — F1 Score Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig7_f1_comparison():
    print("  Generating Fig 7: F1 Score Comparison...")

    data = {
        "model":    MODEL_LABELS + ["Silero VAD", "WebRTC VAD"],
        "exp2_f1":  [0.9718, 0.9952, 0.9963, 0.9767, 0.4114, 0.3333],
        "exp3_f1":  [0.9854, 0.9957, 0.9965, 0.9857, 0.4114, 0.3333],
    }
    df = pd.DataFrame(data)

    x     = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - width/2, df["exp2_f1"], width,
           label="Clean PCM Training (Baseline)",
           color="#A5D6A7", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, df["exp3_f1"], width,
           label="G.711 A-law Training (Proposed)",
           color="#2E7D32", edgecolor="black", linewidth=0.5)

    for i, (f2, f3) in enumerate(zip(df["exp2_f1"], df["exp3_f1"])):
        ax.text(x[i] - width/2, f2 + 0.003, f"{f2:.4f}",
                ha="center", va="bottom", fontsize=7, rotation=45)
        ax.text(x[i] + width/2, f3 + 0.003, f"{f3:.4f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
                fontweight="bold")

    ax.axhline(y=0.5, color="red", ls="--", lw=1.5, alpha=0.7,
               label="Random guessing (F1 ≈ 0.33-0.41)")
    ax.axvline(x=3.5, color="gray", ls=":", lw=1.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("F1 Score (Macro)")
    ax.set_title("F1 Score Comparison Across Models and Training Conditions",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.08])
    ax.grid(True, axis="y", alpha=0.3)

    path = EVAL_FIGS / "fig7_f1_comparison.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — CM5 Latency Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig8_cm5_latency():
    print("  Generating Fig 8: CM5 Latency...")

    latency_data = {
        "model":   MODEL_LABELS,
        "mean_ms": [0.307, 0.764, 2.334, 0.466],
        "std_ms":  [0.011, 0.032, 0.066, 0.024],
        "size_kb": [488,   686,   960,   344],
    }
    df = pd.DataFrame(latency_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS[n] for n in MODEL_NAMES]
    bars   = ax.bar(df["model"], df["mean_ms"],
                    yerr=df["std_ms"], capsize=5,
                    color=colors, edgecolor="black",
                    linewidth=0.8, alpha=0.85,
                    error_kw={"elinewidth": 1.5})

    # value labels
    for bar, mean, std in zip(bars, df["mean_ms"], df["std_ms"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + std + 0.05,
                f"{mean:.3f}ms",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # real-time constraint line
    ax.axhline(y=32, color="red", ls="--", lw=2,
               label="Real-time constraint (32ms)")
    ax.text(3.4, 32.5, "32ms real-time limit",
            color="red", fontsize=9, ha="right")

    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title(
        "CM5 ARM Cortex-A76 Inference Latency\n"
        "(Mean ± Std over 200 runs, ONNX CPU, no GPU)",
        fontweight="bold"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim([0, 5])
    ax.set_xticklabels(df["model"], rotation=10, ha="right")

    # add model size annotation
    for i, (bar, sz) in enumerate(zip(bars, df["size_kb"])):
        ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                f"{sz}KB", ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")

    path = EVAL_FIGS / "fig8_cm5_latency.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Latency vs Accuracy (Pareto)
# ══════════════════════════════════════════════════════════════════════════════
def fig9_latency_vs_accuracy():
    print("  Generating Fig 9: Latency vs Accuracy...")

    data = {
        "model":    MODEL_LABELS + ["Silero VAD", "WebRTC VAD"],
        "accuracy": [98.54, 99.57, 99.65, 97.85, 49.90, 49.50],
        "latency":  [0.307, 0.764, 2.334, 0.466, None,  None],
        "size":     [488,   686,   960,   344,   1800,  None],
        "color":    [COLORS[n] for n in MODEL_NAMES] +
                    [COLORS["Silero"], COLORS["WebRTC"]],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 7))

    # plot our models
    our = df.dropna(subset=["latency"])
    sc  = ax.scatter(our["latency"], our["accuracy"],
                     s=[sz / 3 for sz in our["size"]],
                     c=our["color"], alpha=0.85,
                     edgecolors="black", linewidth=1.2, zorder=5)

    for _, row in our.iterrows():
        ax.annotate(row["model"],
                    xy=(row["latency"], row["accuracy"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold")

    # plot baselines as X markers
    baseline = df[df["latency"].isna() & df["size"].notna()]
    ax.scatter([0.5], [49.90], marker="X", s=200,
               color=COLORS["Silero"], zorder=5,
               label="Silero VAD (49.9%)", edgecolors="black")
    ax.scatter([0.5], [49.50], marker="X", s=200,
               color=COLORS["WebRTC"], zorder=5,
               label="WebRTC VAD (49.5%)", edgecolors="black")

    ax.axhline(y=50, color="red", ls="--", lw=1.5, alpha=0.6,
               label="Random guessing (50%)")
    ax.axvline(x=32, color="orange", ls="--", lw=1.5, alpha=0.6,
               label="Real-time limit (32ms)")

    ax.set_xlabel("Inference Latency on CM5 (ms) — log scale")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(
        "Accuracy vs Latency Tradeoff on CM5 Hardware\n"
        "(Bubble size proportional to model file size)",
        fontweight="bold"
    )
    ax.set_xscale("log")
    ax.set_xlim([0.1, 50])
    ax.set_ylim([45, 101])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # size legend
    for sz, label in [(344, "344KB"), (686, "686KB"), (960, "960KB")]:
        ax.scatter([], [], s=sz/3, color="gray", alpha=0.5,
                   label=label, edgecolors="black")
    ax.legend(loc="lower right", fontsize=9)

    path = EVAL_FIGS / "fig9_latency_vs_accuracy.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Model Size vs Accuracy
# ══════════════════════════════════════════════════════════════════════════════
def fig10_size_vs_accuracy():
    print("  Generating Fig 10: Model Size vs Accuracy...")

    data = {
        "model":    MODEL_LABELS + ["Silero VAD"],
        "accuracy": [98.54, 99.57, 99.65, 97.85, 49.90],
        "size_kb":  [488,   686,   960,   344,   1800],
        "params":   [124389, 175237, 242133, 71461, 1000000],
        "color":    [COLORS[n] for n in MODEL_NAMES] + [COLORS["Silero"]],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in df.iterrows():
        ax.scatter(row["size_kb"], row["accuracy"],
                   s=200, color=row["color"],
                   edgecolors="black", linewidth=1.2,
                   zorder=5)
        ax.annotate(
            f"{row['model']}\n({row['params']:,} params)",
            xy=(row["size_kb"], row["accuracy"]),
            xytext=(10, -15), textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray",
                            lw=0.8)
        )

    ax.axhline(y=50, color="red", ls="--", lw=1.5, alpha=0.6,
               label="Random guessing (50%)")

    ax.set_xlabel("Model File Size (KB)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(
        "Model Size vs Test Accuracy\n"
        "G.711 A-law Trained Models vs Silero VAD Baseline",
        fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2100])
    ax.set_ylim([45, 101])

    path = EVAL_FIGS / "fig10_size_vs_accuracy.pdf"
    plt.savefig(str(path))
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*55}")
    print(f"  Generating all paper figures")
    print(f"{'='*55}\n")

    # figures that need only CSV data (no model loading)
    fig1_training_loss()
    fig2_training_accuracy()
    fig6_accuracy_comparison()
    fig7_f1_comparison()
    fig8_cm5_latency()
    fig9_latency_vs_accuracy()
    fig10_size_vs_accuracy()

    # figures that need model loading (need splits data too)
    splits_exist = (
        ROOT / "data" / "splits" / "exp3_alaw_musan" / "X_test.npy"
    ).exists()

    models_exist = (
        ROOT / "outputs" / "exp3_alaw_musan" / "models" / "CNN1D_best.pt"
    ).exists()

    if splits_exist and models_exist:
        fig3_confusion_matrices()
        fig4_roc_curves()
        fig5_pr_curves()
    else:
        print("\n  [SKIP] Figs 3,4,5 need model files + test splits")
        print("         Copy data/splits/exp3_alaw_musan/ and")
        print("         outputs/exp3_alaw_musan/models/ to LG Gram first")

    print(f"\n{'='*55}")
    print(f"  Done. Check:")
    print(f"    {EXP3_FIGS}")
    print(f"    {EVAL_FIGS}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()