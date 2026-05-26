"""
generate_all_figures.py

Generates all paper figures for the VAD domain gap paper.
Run from project root: python generate_all_figures.py

Requires:
  outputs/exp3_alaw_musan/models/      - proposed G.711 models
  outputs/exp_raw_16khz/models/        - raw 16kHz baseline models
  outputs/exp3_alaw_musan/stats/       - training history JSONs
  data/splits/exp3_alaw_musan/         - G.711 test set
  data/splits/exp_raw_16khz/           - raw 16kHz test set (own)

Outputs (all PDFs, upload to Overleaf figures/ folder):
  figures/fig1_training_loss.pdf
  figures/fig2_training_accuracy.pdf
  figures/fig3_confusion_matrices.pdf
  figures/fig4_roc_curves.pdf
  figures/fig5_pr_curves.pdf
  figures/fig6_accuracy_comparison.pdf
  figures/fig7_domain_gap.pdf
  figures/fig8_cm5_latency.pdf
  figures/fig9_latency_vs_accuracy.pdf
"""

import os, sys, json, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.labelsize':   12,
    'axes.titlesize':   12,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

COLORS = {
    'CNN1D':          '#1565C0',
    'WaveNetSmall':   '#2E7D32',
    'ECAPAVAD':       '#C62828',
    'TransformerVAD': '#6A1B9A',
    'silero':         '#F57F17',
    'webrtc':         '#BF360C',
    'raw16':          '#546E7A',
}

LABELS = {
    'CNN1D':          'CNN1D',
    'WaveNetSmall':   'WaveNet-Small',
    'ECAPAVAD':       'ECAPA-VAD',
    'TransformerVAD': 'Transformer-VAD',
}

MODEL_NAMES = ['CNN1D', 'WaveNetSmall', 'ECAPAVAD', 'TransformerVAD']


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_history(name, stats_dir):
    """Load training history JSON for one model."""
    path = Path(stats_dir) / f'{name}_history.json'
    if not path.exists():
        # try lowercase variants
        for p in Path(stats_dir).glob('*.json'):
            if name.lower() in p.stem.lower():
                path = p
                break
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_model_and_predict(name, model_dir, X, device_str='cpu'):
    """Load ONNX or PT model and return predictions and probabilities."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    # try to import model classes
    try:
        from src.models.cnn1d import CNN1D
        from src.models.wavenet_small import WaveNetSmall
        from src.models.ecapa_vad import ECAPAVAD
        from src.models.transformer_vad import TransformerVAD
        MODEL_CLASSES = {
            'CNN1D': CNN1D,
            'WaveNetSmall': WaveNetSmall,
            'ECAPAVAD': ECAPAVAD,
            'TransformerVAD': TransformerVAD,
        }
    except ImportError as e:
        print(f'  [ERROR] Cannot import model classes: {e}')
        return None, None

    pt_path = Path(model_dir) / f'{name}_best.pt'
    if not pt_path.exists():
        print(f'  [SKIP] {pt_path} not found')
        return None, None

    device = torch.device('cpu')
    ModelClass = MODEL_CLASSES[name]
    model = ModelClass(num_classes=2)
    ckpt  = torch.load(str(pt_path), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            b   = torch.tensor(X[i:i+512], dtype=torch.float32)
            out = model(b)
            preds.extend(out.argmax(1).numpy())
            probs.extend(torch.softmax(out, dim=1)[:,1].numpy())

    model.cpu()
    del model
    gc.collect()
    return np.array(preds), np.array(probs)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 & 2: Training loss and accuracy curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_curves():
    print('Generating fig1_training_loss.pdf and fig2_training_accuracy.pdf...')

    stats_dir = ROOT / 'outputs' / 'exp3_alaw_musan' / 'stats'
    if not stats_dir.exists():
        print(f'  [SKIP] {stats_dir} not found')
        return

    for metric, ylabel, fname in [
        ('loss', 'Cross-entropy loss', 'fig1_training_loss.pdf'),
        ('acc',  'Accuracy',           'fig2_training_accuracy.pdf'),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
        fig.suptitle(
            f'Training and Validation {"Loss" if metric=="loss" else "Accuracy"}'
            f' — G.711 A-law Domain-Matched Dataset',
            fontsize=12, fontweight='bold'
        )

        for ax, name in zip(axes.flat, MODEL_NAMES):
            hist = load_history(name, stats_dir)
            if hist is None:
                ax.text(0.5, 0.5, f'{LABELS[name]}\n(history not found)',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            tr_key = f'train_{metric}'
            vl_key = f'val_{metric}'
            tr = hist.get(tr_key, [])
            vl = hist.get(vl_key, [])

            if not tr:
                ax.text(0.5, 0.5, f'{LABELS[name]}\n(no data)',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            epochs = range(1, len(tr)+1)
            ax.plot(epochs, tr, color=COLORS[name], lw=2,
                    label='Train', alpha=0.9)
            ax.plot(epochs, vl, color=COLORS[name], lw=2,
                    ls='--', label='Validation', alpha=0.9)

            # mark best epoch
            if metric == 'loss':
                best_ep = int(np.argmin(vl)) + 1
            else:
                best_ep = int(np.argmax(vl)) + 1

            ax.axvline(x=best_ep, color='gray', ls=':', lw=1.2,
                       label=f'Best (ep {best_ep})')

            ax.set_title(LABELS[name], fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(str(OUT / fname))
        plt.close()
        print(f'  Saved {fname}')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Confusion matrices (proposed G.711 models)
# ══════════════════════════════════════════════════════════════════════════════

def fig_confusion_matrices():
    print('Generating fig3_confusion_matrices.pdf...')
    from sklearn.metrics import confusion_matrix

    X_test = np.load(str(ROOT / 'data/splits/exp3_alaw_musan/X_test.npy'))
    y_test = np.load(str(ROOT / 'data/splits/exp3_alaw_musan/y_test.npy'))
    model_dir = ROOT / 'outputs' / 'exp3_alaw_musan' / 'models'

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        'Confusion Matrices — Proposed G.711 A-law Trained Models\n'
        f'(Test set: {len(y_test):,} windows)',
        fontsize=12, fontweight='bold'
    )

    for ax, name in zip(axes.flat, MODEL_NAMES):
        preds, _ = load_model_and_predict(name, model_dir, X_test)
        if preds is None:
            ax.text(0.5, 0.5, f'{LABELS[name]}\n(model not found)',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        cm     = confusion_matrix(y_test, preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Noise', 'Speech'])
        ax.set_yticklabels(['Noise', 'Speech'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(LABELS[name], fontweight='bold')

        for i in range(2):
            for j in range(2):
                col = 'white' if cm_pct[i,j] > 55 else 'black'
                ax.text(j, i,
                        f'{cm_pct[i,j]:.1f}%\n({cm[i,j]:,})',
                        ha='center', va='center',
                        color=col, fontsize=10, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(str(OUT / 'fig3_confusion_matrices.pdf'))
    plt.close()
    print('  Saved fig3_confusion_matrices.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 & 5: ROC and PR curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_roc_pr():
    print('Generating fig4_roc_curves.pdf and fig5_pr_curves.pdf...')
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    X_test = np.load(str(ROOT / 'data/splits/exp3_alaw_musan/X_test.npy'))
    y_test = np.load(str(ROOT / 'data/splits/exp3_alaw_musan/y_test.npy'))
    model_dir = ROOT / 'outputs' / 'exp3_alaw_musan' / 'models'

    # collect all probs
    all_probs = {}
    for name in MODEL_NAMES:
        _, probs = load_model_and_predict(name, model_dir, X_test)
        if probs is not None:
            all_probs[name] = probs

    # Fig 4: ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[name], lw=2,
                label=f'{LABELS[name]} (AUC = {roc_auc:.4f})')

    ax.plot([0,1],[0,1], 'k--', lw=1.5,
            label='Silero VAD / WebRTC VAD (AUC $\\approx$ 0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — G.711 A-law Trained Models',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    plt.savefig(str(OUT / 'fig4_roc_curves.pdf'))
    plt.close()
    print('  Saved fig4_roc_curves.pdf')

    # Fig 5: PR
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in all_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(rec, prec, color=COLORS[name], lw=2,
                label=f'{LABELS[name]} (AP = {ap:.4f})')

    ax.axhline(y=0.5, color='k', ls='--', lw=1.5,
               label='Silero VAD / WebRTC VAD (AP $\\approx$ 0.50)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — G.711 A-law Trained Models',
                 fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    plt.savefig(str(OUT / 'fig5_pr_curves.pdf'))
    plt.close()
    print('  Saved fig5_pr_curves.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6: Accuracy comparison (all conditions)
# ══════════════════════════════════════════════════════════════════════════════

def fig_accuracy_comparison():
    print('Generating fig6_accuracy_comparison.pdf...')

    # hardcoded results
    proposed = {
        'CNN1D':          0.9755,
        'WaveNetSmall':   0.9955,
        'ECAPAVAD':       0.9965,
        'TransformerVAD': 0.9756,
    }
    raw16 = {
        'CNN1D':          0.9126,
        'WaveNetSmall':   0.9440,
        'ECAPAVAD':       0.9569,
        'TransformerVAD': 0.8853,
    }
    baselines = {
        'Silero VAD':  0.4990,
        'WebRTC VAD':  0.4950,
    }

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x      = np.arange(len(MODEL_NAMES))
    width  = 0.28
    offset = 0.0

    # proposed bars
    bars1 = ax.bar(x - width, [proposed[n] for n in MODEL_NAMES],
                   width, label='Proposed (G.711 A-law trained)',
                   color='#1565C0', alpha=0.88, edgecolor='black', lw=0.6)

    # raw 16kHz bars
    bars2 = ax.bar(x, [raw16[n] for n in MODEL_NAMES],
                   width, label='Raw 16 kHz (no codec sim)',
                   color='#546E7A', alpha=0.88, edgecolor='black', lw=0.6)

    # annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    # baseline horizontal lines
    line_colors = {'Silero VAD': '#F57F17', 'WebRTC VAD': '#BF360C'}
    line_styles = {'Silero VAD': '--', 'WebRTC VAD': ':'}
    for bname, bacc in baselines.items():
        ax.axhline(y=bacc, color=line_colors[bname],
                   ls=line_styles[bname], lw=2,
                   label=f'{bname} ({bacc:.4f})')

    ax.set_xticks(x - width/2)
    ax.set_xticklabels([LABELS[n] for n in MODEL_NAMES])
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.40, 1.03])
    ax.set_title('Test Accuracy Comparison — G.711 A-law Test Set\n'
                 '(60,000 windows, 30,000 speech / 30,000 non-speech)',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

    plt.tight_layout()
    plt.savefig(str(OUT / 'fig6_accuracy_comparison.pdf'))
    plt.close()
    print('  Saved fig6_accuracy_comparison.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7: Domain gap bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_domain_gap():
    print('Generating fig7_domain_gap.pdf...')

    own_acc  = [0.9776, 0.9882, 0.9904, 0.9553]
    alaw_acc = [0.9126, 0.9440, 0.9569, 0.8853]
    gaps     = [o - a for o, a in zip(own_acc, alaw_acc)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: own vs alaw grouped
    x     = np.arange(len(MODEL_NAMES))
    width = 0.35
    bars1 = axes[0].bar(x - width/2, own_acc, width,
                        label='Own test (16 kHz)',
                        color='#546E7A', alpha=0.88,
                        edgecolor='black', lw=0.6)
    bars2 = axes[0].bar(x + width/2, alaw_acc, width,
                        label='G.711 A-law test',
                        color='#C62828', alpha=0.88,
                        edgecolor='black', lw=0.6)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.002,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[n] for n in MODEL_NAMES], rotation=10)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0.80, 1.02])
    axes[0].set_title('Own Test vs. G.711 A-law Test Accuracy\n'
                      '(Raw 16 kHz trained models)',
                      fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.25)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

    # Right: gap bar chart
    bar_colors = [COLORS[n] for n in MODEL_NAMES]
    bars = axes[1].bar([LABELS[n] for n in MODEL_NAMES],
                       [g*100 for g in gaps],
                       color=bar_colors, alpha=0.88,
                       edgecolor='black', lw=0.6)
    for bar, g in zip(bars, gaps):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f'{g*100:.2f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    axes[1].set_ylabel('Accuracy drop ($\\Delta$, percentage points)')
    axes[1].set_title('Codec-Induced Domain Gap\n'
                      '($\\Delta$ = Own Acc $-$ A-law Acc)',
                      fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.25)
    axes[1].tick_params(axis='x', rotation=10)

    plt.tight_layout()
    plt.savefig(str(OUT / 'fig7_domain_gap.pdf'))
    plt.close()
    print('  Saved fig7_domain_gap.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8: CM5 latency
# ══════════════════════════════════════════════════════════════════════════════

def fig_cm5_latency():
    print('Generating fig8_cm5_latency.pdf...')

    latency_mean = {
        'CNN1D':          0.307,
        'WaveNetSmall':   0.764,
        'ECAPAVAD':       2.334,
        'TransformerVAD': 0.466,
    }
    latency_std = {
        'CNN1D':          0.011,
        'WaveNetSmall':   0.032,
        'ECAPAVAD':       0.066,
        'TransformerVAD': 0.024,
    }
    model_size = {
        'CNN1D':          488,
        'WaveNetSmall':   686,
        'ECAPAVAD':       960,
        'TransformerVAD': 344,
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    x      = np.arange(len(MODEL_NAMES))
    colors = [COLORS[n] for n in MODEL_NAMES]
    means  = [latency_mean[n] for n in MODEL_NAMES]
    stds   = [latency_std[n]  for n in MODEL_NAMES]
    sizes  = [model_size[n]   for n in MODEL_NAMES]

    bars = ax.bar(x, means, yerr=stds, width=0.5,
                  color=colors, alpha=0.88,
                  edgecolor='black', lw=0.6,
                  capsize=5, error_kw={'lw': 1.5})

    # annotate: latency + size
    for i, (bar, m, s, sz) in enumerate(zip(bars, means, stds, sizes)):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + s + 0.04,
                f'{m:.3f} ms\n({sz} KB)',
                ha='center', va='bottom', fontsize=9)

    # real-time constraint line
    ax.axhline(y=32, color='red', ls='--', lw=2,
               label='Real-time constraint (32 ms)')

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in MODEL_NAMES])
    ax.set_ylabel('Inference latency (ms)')
    ax.set_title('CM5 Inference Latency (ARM Cortex-A76, ONNX Runtime, CPU)\n'
                 'Mean $\\pm$ std over 200 inference runs',
                 fontweight='bold')
    ax.set_ylim([0, 5.5])
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(str(OUT / 'fig8_cm5_latency.pdf'))
    plt.close()
    print('  Saved fig8_cm5_latency.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9: Latency vs accuracy bubble chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_latency_vs_accuracy():
    print('Generating fig9_latency_vs_accuracy.pdf...')

    data = {
        'CNN1D':          {'acc': 0.9755, 'lat': 0.307, 'size': 488},
        'WaveNetSmall':   {'acc': 0.9955, 'lat': 0.764, 'size': 686},
        'ECAPAVAD':       {'acc': 0.9965, 'lat': 2.334, 'size': 960},
        'TransformerVAD': {'acc': 0.9756, 'lat': 0.466, 'size': 344},
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, d in data.items():
        bubble_size = d['size'] / 2
        ax.scatter(d['lat'], d['acc'],
                   s=bubble_size, c=COLORS[name],
                   alpha=0.85, edgecolors='black', lw=0.8,
                   zorder=3)
        # label offset to avoid overlap
        offsets = {
            'CNN1D':          (-0.08, 0.002),
            'WaveNetSmall':   (0.04, -0.003),
            'ECAPAVAD':       (0.06, 0.001),
            'TransformerVAD': (0.04, 0.002),
        }
        ox, oy = offsets[name]
        ax.annotate(
            f'{LABELS[name]}\n({d["size"]} KB)',
            xy=(d['lat'], d['acc']),
            xytext=(d['lat'] + ox, d['acc'] + oy),
            fontsize=9,
            ha='left' if ox > 0 else 'right',
        )

    # baseline line
    ax.axhline(y=0.499, color='gray', ls='--', lw=1.5,
               label='Off-the-shelf baselines ($\\approx$50%)')

    # real-time line
    ax.axvline(x=32, color='red', ls='--', lw=1.5,
               label='Real-time constraint (32 ms)')

    ax.set_xlabel('Inference latency on CM5 (ms)')
    ax.set_ylabel('Accuracy on G.711 A-law test set')
    ax.set_title('Accuracy vs. Latency Tradeoff\n'
                 '(Bubble size $\\propto$ ONNX model size in KB)',
                 fontweight='bold')
    ax.set_xlim([-0.2, 4.5])
    ax.set_ylim([0.45, 1.01])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

    plt.tight_layout()
    plt.savefig(str(OUT / 'fig9_latency_vs_accuracy.pdf'))
    plt.close()
    print('  Saved fig9_latency_vs_accuracy.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'\n{"="*55}')
    print('  Generating all paper figures')
    print(f'  Output: {OUT}')
    print(f'{"="*55}\n')

    # Figures using hardcoded stats (no models needed)
    fig_domain_gap()
    fig_cm5_latency()
    fig_latency_vs_accuracy()
    fig_accuracy_comparison()

    # Figures requiring model inference (need GPU or patience on CPU)
    print('\nGenerating figures that require model inference...')
    print('(These may take several minutes on CPU)')

    try:
        import torch
        fig_confusion_matrices()
        fig_roc_pr()
    except ImportError:
        print('  [SKIP] PyTorch not available — skipping confusion/ROC/PR')

    # Training curves (need JSON history files)
    fig_training_curves()

    print(f'\n{"="*55}')
    print('  Done. Upload all PDFs from figures/ to Overleaf figures/ folder.')
    print(f'{"="*55}\n')