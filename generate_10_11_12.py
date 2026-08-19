"""
regenerate_figures_10_11_12.py

Regenerates Figure_10, Figure_11, Figure_12 using the correct
60,000-window test set.

Run from project root:
    python regenerate_figures_10_11_12.py

If the default path does not work, set XTEST_PATH and YTEST_PATH
manually at the top of this script.

Output: paper_figures/Figure_10.png
        paper_figures/Figure_11.png
        paper_figures/Figure_12.png
"""

import sys, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / 'paper_figures'
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 400

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'legend.fontsize':   8,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'savefig.format':    'png',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLORS = {
    'CNN1D':          '#1565C0',
    'WaveNetSmall':   '#2E7D32',
    'ECAPAVAD':       '#C62828',
    'TransformerVAD': '#6A1B9A',
}
LABELS = {
    'CNN1D':          'CNN1D',
    'WaveNetSmall':   'WaveNet-Small',
    'ECAPAVAD':       'ECAPA-VAD',
    'TransformerVAD': 'Transformer-VAD',
}
MODEL_NAMES = ['CNN1D', 'WaveNetSmall', 'ECAPAVAD', 'TransformerVAD']

# ── Test set paths — change these if your files are elsewhere ─────────────────
# Try these candidates in order — first one that exists will be used
TEST_SET_CANDIDATES = [
    ROOT / 'data/splits/exp3_alaw_musan/X_test.npy',
    ROOT / 'data/splits/X_test.npy',
    ROOT / 'data/X_test.npy',
    ROOT / 'outputs/exp3_alaw_musan/X_test.npy',
]
YTEST_CANDIDATES = [
    ROOT / 'data/splits/exp3_alaw_musan/y_test.npy',
    ROOT / 'data/splits/y_test.npy',
    ROOT / 'data/y_test.npy',
    ROOT / 'outputs/exp3_alaw_musan/y_test.npy',
]

MODEL_DIR = ROOT / 'outputs' / 'exp3_alaw_musan' / 'models'


def find_test_set():
    """Find X_test and y_test from candidate paths."""
    X_path, y_path = None, None
    for p in TEST_SET_CANDIDATES:
        if p.exists():
            X_path = p
            break
    for p in YTEST_CANDIDATES:
        if p.exists():
            y_path = p
            break
    if X_path is None or y_path is None:
        print("\n  [ERROR] Could not find test set files.")
        print("  Searched:")
        for p in TEST_SET_CANDIDATES:
            print(f"    {p}")
        print("\n  Set XTEST_PATH manually at the top of this script.")
        return None, None
    return X_path, y_path


def load_model_and_predict(name, X):
    import torch
    try:
        from src.models.cnn1d import CNN1D
        from src.models.wavenet_small import WaveNetSmall
        from src.models.ecapa_vad import ECAPAVAD
        from src.models.transformer_vad import TransformerVAD
        MC = {'CNN1D': CNN1D, 'WaveNetSmall': WaveNetSmall,
              'ECAPAVAD': ECAPAVAD, 'TransformerVAD': TransformerVAD}
    except ImportError as e:
        print(f'  [ERROR] {e}')
        return None, None

    pt = MODEL_DIR / f'{name}_best.pt'
    if not pt.exists():
        print(f'  [SKIP] {pt} not found')
        return None, None

    model = MC[name](num_classes=2)
    ckpt  = torch.load(str(pt), map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            b   = torch.tensor(X[i:i+512], dtype=torch.float32)
            out = model(b)
            preds.extend(out.argmax(1).numpy())
            probs.extend(torch.softmax(out, dim=1)[:, 1].numpy())

    model.cpu()
    del model
    gc.collect()
    return np.array(preds), np.array(probs)


# ══════════════════════════════════════════════════════════════════════════════
# Figure_10.png — Confusion matrices
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_10(X_test, y_test):
    print('  Generating Figure_10.png (confusion matrices) ...')
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))

    for ax, name in zip(axes.flat, MODEL_NAMES):
        preds, _ = load_model_and_predict(name, X_test)
        if preds is None:
            ax.text(0.5, 0.5, f'{LABELS[name]}\n(model not found)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8)
            continue

        cm     = confusion_matrix(y_test, preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Noise', 'Speech'], fontsize=8)
        ax.set_yticklabels(['Noise', 'Speech'], fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True',      fontsize=8)

        # model name as text inside panel — no ax.set_title
        ax.text(0.5, 1.01, LABELS[name],
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, fontweight='bold')

        for i in range(2):
            for j in range(2):
                col = 'white' if cm_pct[i, j] > 55 else 'black'
                ax.text(j, i,
                        f'{cm_pct[i,j]:.1f}%\n({cm[i,j]:,})',
                        ha='center', va='center',
                        color=col, fontsize=8, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046,
                     pad=0.04).ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = OUTPUT_DIR / 'Figure_10.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Figure_11.png — ROC curves
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_11(X_test, y_test, all_probs):
    print('  Generating Figure_11.png (ROC curves) ...')
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.5,
                label=f'{LABELS[name]} (AUC={roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2,
            label='Baselines ($\\approx$0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    out = OUTPUT_DIR / 'Figure_11.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Figure_12.png — PR curves
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_12(X_test, y_test, all_probs):
    print('  Generating Figure_12.png (PR curves) ...')
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for name, probs in all_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(rec, prec, color=COLORS[name], lw=1.5,
                label=f'{LABELS[name]} (AP={ap:.4f})')

    ax.axhline(y=0.5, color='k', ls='--', lw=1.2,
               label='Baselines ($\\approx$0.50)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    out = OUTPUT_DIR / 'Figure_12.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'\n{"="*55}')
    print('  Regenerating Figure_10, Figure_11, Figure_12')
    print(f'  Output: {OUTPUT_DIR.resolve()}  |  DPI: {DPI}')
    print(f'{"="*55}\n')

    # Find test set
    X_path, y_path = find_test_set()
    if X_path is None:
        sys.exit(1)

    print(f'  Loading test set from:')
    print(f'    X: {X_path}')
    print(f'    y: {y_path}')

    X_test = np.load(str(X_path))
    y_test = np.load(str(y_path))

    print(f'  Test set shape: X={X_test.shape}, y={y_test.shape}')

    if len(X_test) < 1000:
        print(f'\n  [WARNING] Test set has only {len(X_test)} samples.')
        print('  This is likely the wrong file — expected 60,000 windows.')
        print('  Check the path and update TEST_SET_CANDIDATES in this script.')
        ans = input('  Continue anyway? (y/n): ').strip().lower()
        if ans != 'y':
            sys.exit(1)

    print(f'\n  Running inference on all four models...')

    try:
        import torch
    except ImportError:
        print('  [ERROR] PyTorch not available. Install with: pip install torch')
        sys.exit(1)

    # Run inference once and reuse probs for Fig 11 and 12
    all_probs = {}
    for name in MODEL_NAMES:
        preds, probs = load_model_and_predict(name, X_test)
        if probs is not None:
            all_probs[name] = probs

    if not all_probs:
        print('  [ERROR] No models loaded. Check MODEL_DIR path.')
        sys.exit(1)

    gen_figure_10(X_test, y_test)
    gen_figure_11(X_test, y_test, all_probs)
    gen_figure_12(X_test, y_test, all_probs)

    print(f'\n{"="*55}')
    print('  Done. Upload to Overleaf figs/:')
    for n in [10, 11, 12]:
        p = OUTPUT_DIR / f'Figure_{n}.png'
        if p.exists():
            kb = p.stat().st_size // 1024
            print(f'    Figure_{n}.png  ({kb} KB)')
    print(f'{"="*55}\n')