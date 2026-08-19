"""
generate_all_figures_final.py

Generates Figure_8 through Figure_14 for the VAD paper.
Run from project root: python generate_all_figures_final.py

Paper sequence:
    Figure_7.png   -> fig:arch_all         [SKIP - draw.io]
    Figure_8.png   -> fig:accuracy_comparison
    Figure_9.png   -> fig:domain_gap
    Figure_10.png  -> fig:confusion_matrices
    Figure_11.png  -> fig:roc_curves
    Figure_12.png  -> fig:pr_curves
    Figure_13.png  -> fig:training_loss
    Figure_14.png  -> fig:latency
    Figure_15.png  -> fig:system_architecture [SKIP - draw.io]
"""

import os, sys, json, gc
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


def load_history(name, stats_dir):
    path = Path(stats_dir) / f'{name}_history.json'
    if not path.exists():
        for p in Path(stats_dir).glob('*.json'):
            if name.lower() in p.stem.lower():
                path = p; break
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_model_and_predict(name, model_dir, X):
    import torch
    try:
        from src.models.cnn1d import CNN1D
        from src.models.wavenet_small import WaveNetSmall
        from src.models.ecapa_vad import ECAPAVAD
        from src.models.transformer_vad import TransformerVAD
        MC = {'CNN1D':CNN1D,'WaveNetSmall':WaveNetSmall,
              'ECAPAVAD':ECAPAVAD,'TransformerVAD':TransformerVAD}
    except ImportError as e:
        print(f'  [ERROR] {e}'); return None, None
    pt = Path(model_dir) / f'{name}_best.pt'
    if not pt.exists():
        print(f'  [SKIP] {pt} not found'); return None, None
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
            probs.extend(torch.softmax(out,dim=1)[:,1].numpy())
    model.cpu(); del model; gc.collect()
    return np.array(preds), np.array(probs)


# ══════════════════════════════════════════════════════════════════════════════
# Figure_8.png — Accuracy comparison
# NO title inside figure
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_8():
    print('  Generating Figure_8.png (accuracy comparison) ...')
    proposed  = {'CNN1D':0.9755,'WaveNetSmall':0.9955,
                 'ECAPAVAD':0.9965,'TransformerVAD':0.9756}
    raw16     = {'CNN1D':0.9126,'WaveNetSmall':0.9440,
                 'ECAPAVAD':0.9569,'TransformerVAD':0.8853}
    baselines = {'Silero VAD':0.4990,'WebRTC VAD':0.4950}

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x, w = np.arange(len(MODEL_NAMES)), 0.30

    b1 = ax.bar(x-w/2, [proposed[n] for n in MODEL_NAMES], w,
                label='Proposed (G.711 A-law)',
                color='#1565C0', alpha=0.88, edgecolor='black', lw=0.6)
    b2 = ax.bar(x+w/2, [raw16[n]    for n in MODEL_NAMES], w,
                label='Raw 16 kHz (no codec sim)',
                color='#546E7A', alpha=0.88, edgecolor='black', lw=0.6)
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    lc = {'Silero VAD':'#F57F17','WebRTC VAD':'#BF360C'}
    ls = {'Silero VAD':'--',     'WebRTC VAD':':'}
    for bn, ba in baselines.items():
        ax.axhline(y=ba, color=lc[bn], ls=ls[bn], lw=1.5,
                   label=f'{bn} ({ba:.4f})')

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in MODEL_NAMES])
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.40, 1.03])
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

    plt.tight_layout()
    out = OUTPUT_DIR / 'Figure_8.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Figure_9.png — Domain gap
# NO title inside figure
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_9():
    print('  Generating Figure_9.png (domain gap) ...')
    own  = [0.9776, 0.9882, 0.9904, 0.9553]
    alaw = [0.9126, 0.9440, 0.9569, 0.8853]
    gaps = [o-a for o,a in zip(own,alaw)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    x, w = np.arange(len(MODEL_NAMES)), 0.35

    b1 = axes[0].bar(x-w/2, own,  w, label='Own test (16 kHz)',
                      color='#546E7A', alpha=0.88, edgecolor='black', lw=0.6)
    b2 = axes[0].bar(x+w/2, alaw, w, label='G.711 A-law test',
                      color='#C62828', alpha=0.88, edgecolor='black', lw=0.6)
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x()+bar.get_width()/2, h+0.002,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[n] for n in MODEL_NAMES],
                              rotation=10, fontsize=7)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0.80, 1.02])
    axes[0].legend(fontsize=7)
    axes[0].grid(True, axis='y', alpha=0.2, linewidth=0.4)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

    bars = axes[1].bar([LABELS[n] for n in MODEL_NAMES],
                       [g*100 for g in gaps],
                       color=[COLORS[n] for n in MODEL_NAMES],
                       alpha=0.88, edgecolor='black', lw=0.6)
    for bar, g in zip(bars, gaps):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.05,
                     f'{g*100:.2f} pp',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[1].set_ylabel('Accuracy drop (pp)')
    axes[1].grid(True, axis='y', alpha=0.2, linewidth=0.4)
    axes[1].tick_params(axis='x', rotation=10, labelsize=7)

    plt.tight_layout()
    out = OUTPUT_DIR / 'Figure_9.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Figure_10.png — Confusion matrices
# Subplot labels only (model name inside each panel), NO fig.suptitle
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_10():
    print('  Generating Figure_10.png (confusion matrices) ...')
    from sklearn.metrics import confusion_matrix
    X_test    = np.load(str(ROOT/'data/splits/exp3_alaw_musan/X_test.npy'))
    y_test    = np.load(str(ROOT/'data/splits/exp3_alaw_musan/y_test.npy'))
    model_dir = ROOT/'outputs'/'exp3_alaw_musan'/'models'

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    for ax, name in zip(axes.flat, MODEL_NAMES):
        preds, _ = load_model_and_predict(name, model_dir, X_test)
        if preds is None:
            ax.text(0.5,0.5,f'{LABELS[name]}\n(not found)',
                    ha='center',va='center',transform=ax.transAxes); continue
        cm     = confusion_matrix(y_test, preds)
        cm_pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Noise','Speech'], fontsize=8)
        ax.set_yticklabels(['Noise','Speech'], fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True',      fontsize=8)
        # model name as text inside panel, not as title
        ax.text(0.5, 1.01, LABELS[name],
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, fontweight='bold')
        for i in range(2):
            for j in range(2):
                col = 'white' if cm_pct[i,j]>55 else 'black'
                ax.text(j,i,f'{cm_pct[i,j]:.1f}%\n({cm[i,j]:,})',
                        ha='center',va='center',
                        color=col,fontsize=8,fontweight='bold')
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04).ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = OUTPUT_DIR/'Figure_10.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Figure_11.png and Figure_12.png — ROC and PR curves
# NO titles inside figures
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_11_12():
    print('  Generating Figure_11.png (ROC) and Figure_12.png (PR) ...')
    from sklearn.metrics import (roc_curve, auc,
                                  precision_recall_curve,
                                  average_precision_score)
    X_test    = np.load(str(ROOT/'data/splits/exp3_alaw_musan/X_test.npy'))
    y_test    = np.load(str(ROOT/'data/splits/exp3_alaw_musan/y_test.npy'))
    model_dir = ROOT/'outputs'/'exp3_alaw_musan'/'models'

    all_probs = {}
    for name in MODEL_NAMES:
        _, probs = load_model_and_predict(name, model_dir, X_test)
        if probs is not None:
            all_probs[name] = probs

    # ROC — NO title
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.5,
                label=f'{LABELS[name]} (AUC={auc(fpr,tpr):.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1.2,label='Baselines ($\\approx$0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    out = OUTPUT_DIR/'Figure_11.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')

    # PR — NO title
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for name, probs in all_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(rec, prec, color=COLORS[name], lw=1.5,
                label=f'{LABELS[name]} (AP={ap:.4f})')
    ax.axhline(y=0.5,color='k',ls='--',lw=1.2,
               label='Baselines ($\\approx$0.50)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    out = OUTPUT_DIR/'Figure_12.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Figure_13.png — Training loss curves
# Model name as text inside each panel, NO fig.suptitle
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_13():
    print('  Generating Figure_13.png (training loss) ...')
    stats_dir = ROOT/'outputs'/'exp3_alaw_musan'/'stats'
    if not stats_dir.exists():
        print(f'  [SKIP] {stats_dir} not found'); return

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharey=False)
    for ax, name in zip(axes.flat, MODEL_NAMES):
        hist = load_history(name, stats_dir)
        if hist is None:
            ax.text(0.5,0.5,f'{LABELS[name]}\n(not found)',
                    ha='center',va='center',transform=ax.transAxes,fontsize=8)
            continue
        tr = hist.get('train_loss',[])
        vl = hist.get('val_loss',[])
        if not tr:
            ax.text(0.5,0.5,f'{LABELS[name]}\n(no data)',
                    ha='center',va='center',transform=ax.transAxes,fontsize=8)
            continue
        epochs  = range(1, len(tr)+1)
        best_ep = int(np.argmin(vl))+1
        ax.plot(epochs, tr, color=COLORS[name], lw=1.5,
                label='Train', alpha=0.9)
        ax.plot(epochs, vl, color=COLORS[name], lw=1.5,
                ls='--', label='Validation', alpha=0.9)
        ax.axvline(x=best_ep, color='gray', ls=':', lw=1.0,
                   label=f'Best (ep {best_ep})')
        # model name as text, NOT ax.set_title
        ax.text(0.5, 0.97, LABELS[name],
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, fontweight='bold', color=COLORS[name])
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Loss',  fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, linewidth=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR/'Figure_13.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Figure_14.png — CM5 latency
# NO title inside figure
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_14():
    print('  Generating Figure_14.png (CM5 latency) ...')
    means = {'CNN1D':0.307,'WaveNetSmall':0.764,
             'ECAPAVAD':2.334,'TransformerVAD':0.466}
    stds  = {'CNN1D':0.011,'WaveNetSmall':0.032,
             'ECAPAVAD':0.066,'TransformerVAD':0.024}
    sizes = {'CNN1D':488,'WaveNetSmall':686,
             'ECAPAVAD':960,'TransformerVAD':344}

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(MODEL_NAMES))
    bars = ax.bar(x,
                  [means[n] for n in MODEL_NAMES],
                  yerr=[stds[n] for n in MODEL_NAMES],
                  width=0.5,
                  color=[COLORS[n] for n in MODEL_NAMES],
                  alpha=0.88, edgecolor='black', lw=0.6,
                  capsize=4, error_kw={'lw':1.2})
    for bar, n in zip(bars, MODEL_NAMES):
        m, s, sz = means[n], stds[n], sizes[n]
        ax.text(bar.get_x()+bar.get_width()/2, m+s+0.04,
                f'{m:.3f} ms\n({sz} KB)',
                ha='center', va='bottom', fontsize=8)
    ax.axhline(y=32, color='red', ls='--', lw=1.5,
               label='Real-time constraint (32 ms)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in MODEL_NAMES])
    ax.set_ylabel('Inference latency (ms)')
    ax.set_ylim([0, 5.5])
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR/'Figure_14.png'
    plt.savefig(str(out))
    plt.close()
    print(f'    Saved: {out}  ({DPI} DPI)')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'\n{"="*55}')
    print('  Generating paper figures (Figure_8 to Figure_14)')
    print(f'  Output: {OUTPUT_DIR.resolve()}  |  DPI: {DPI}')
    print(f'{"="*55}\n')

    print('--- Hardcoded results (no models needed) ---')
    gen_figure_8()
    gen_figure_9()
    gen_figure_14()

    print('\n--- Requires model inference ---')
    try:
        import torch
        gen_figure_10()
        gen_figure_11_12()
    except ImportError:
        print('  [SKIP] PyTorch not available')

    print('\n--- Training curves ---')
    gen_figure_13()

    print(f'\n{"="*55}')
    print('  Done. Files saved:')
    for n in [8,9,10,11,12,13,14]:
        p = OUTPUT_DIR/f'Figure_{n}.png'
        if p.exists():
            kb = p.stat().st_size//1024
            print(f'    Figure_{n}.png  ({kb} KB)')
    print()
    print('  SKIP (make in draw.io):')
    print('    Figure_2.png   — pipeline flowchart')
    print('    Figure_7.png   — architecture diagrams')
    print('    Figure_15.png  — VoIP system pipeline')
    print(f'{"="*55}\n')