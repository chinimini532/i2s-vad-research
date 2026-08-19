"""
Figure_7: Four VAD architectures with shared input/output
Layout:
  - Shared INPUT block at top (fan-out arrows to 4 models)
  - 4 parallel model columns with their internal layers
  - Shared OUTPUT block at bottom (fan-in arrows from 4 models)
  - Waveform icons in input/output
  - 600 DPI, CEE PASS
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path

DPI = 600
FW  = 7.2
FH  = 9.0

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      6.5,
    'savefig.dpi':    DPI,
    'savefig.bbox':   'tight',
    'savefig.format': 'png',
})

OUT = Path('paper_figures')
OUT.mkdir(exist_ok=True)

C = {
    'input':  '#37474F',
    'output': '#37474F',
    'cnn':    '#1565C0',
    'wave':   '#2E7D32',
    'ecapa':  '#B71C1C',
    'trans':  '#4A148C',
    'arrow':  '#546E7A',
    'bg':     '#FAFAFA',
}
LIGHT = {
    'cnn':   '#E3F2FD',
    'wave':  '#E8F5E9',
    'ecapa': '#FFEBEE',
    'trans': '#F3E5F5',
}

fig = plt.figure(figsize=(FW, FH), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_facecolor('white')

# ── Helper functions ──────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, lw=0.8, radius=0.008, alpha=1.0, zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f'round,pad=0,rounding_size={radius}',
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p

def label(ax, x, y, txt, fs=6.2, color='#1a1a1a', bold=False,
          ha='center', va='center', zorder=5, style='normal'):
    ax.text(x, y, txt, ha=ha, va=va, fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal',
            style=style, zorder=zorder)

def arr(ax, x1, y1, x2, y2, color='#546E7A', lw=0.9, zorder=4,
        style='->', headw=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=headw),
                zorder=zorder)

def mini_wave(ax, cx, cy, w, h, color, n=80, freq=3.0, noise=0.15):
    """Draw a small waveform icon inside a given box."""
    t   = np.linspace(0, 1, n)
    env = np.exp(-6*(t-0.5)**2)
    sig = env * np.sin(2*np.pi*freq*t)
    sig += np.random.default_rng(42).normal(0, noise, n) * env * 0.3
    sig /= np.max(np.abs(sig) + 1e-8)
    xs  = cx - w/2 + t * w
    ys  = cy + sig * h/2 * 0.75
    ax.plot(xs, ys, color=color, lw=0.7, zorder=6, alpha=0.9)

def mini_bar(ax, cx, cy, w, h, color):
    """Draw small bar-chart icon."""
    vals = [0.4, 0.9, 0.6, 1.0, 0.7, 0.5, 0.85]
    bw   = w / (len(vals)*1.6)
    gap  = w / len(vals)
    for i, v in enumerate(vals):
        bx = cx - w/2 + i*gap + gap*0.1
        bh = v * h * 0.7
        rbox(ax, bx, cy - h*0.35, bw, bh, color, color, lw=0, alpha=0.8)

def mini_grid(ax, cx, cy, w, h, color):
    """Draw small attention-grid icon."""
    n   = 4
    cw  = w / n
    ch  = h / n
    vals = np.array([[0.9,0.1,0.1,0.1],
                     [0.2,0.8,0.1,0.1],
                     [0.1,0.1,0.9,0.2],
                     [0.1,0.1,0.2,0.9]])
    for i in range(n):
        for j in range(n):
            bx = cx - w/2 + j*cw
            by = cy - h/2 + (n-1-i)*ch
            alpha = 0.15 + vals[i,j]*0.8
            rbox(ax, bx+0.001, by+0.001, cw-0.002, ch-0.002,
                 color, 'none', lw=0, alpha=alpha, zorder=5)

def stack_icon(ax, cx, cy, w, h, color):
    """Draw stacked layers icon."""
    n    = 4
    step = h / (n+1)
    for i in range(n):
        bh  = 0.007
        bw  = w * (0.5 + 0.12*i)
        by  = cy - h/2 + (i+0.8)*step
        bx  = cx - bw/2
        rbox(ax, bx, by, bw, bh, color, color, lw=0,
             alpha=0.25+i*0.18, zorder=5)

# ════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
# Y positions (top of each element)
Y_IN_TOP   = 0.945
IN_H       = 0.075
Y_IN_BOT   = Y_IN_TOP - IN_H          # 0.870

Y_FANOUT   = Y_IN_BOT - 0.012         # 0.858  fan-out arrows start

COL_TOPS   = 0.840                    # model column headers start
COL_H      = 0.595                    # total height of each column
COL_BOT    = COL_TOPS - COL_H         # 0.245

Y_FANIN    = COL_BOT - 0.012          # 0.233  fan-in arrows
OUT_H      = 0.075
Y_OUT_TOP  = Y_FANIN - 0.025          # 0.208

# X positions for 4 columns
COL_W   = 0.195
COL_GAP = 0.010
COL_XS  = [0.025 + i*(COL_W + COL_GAP) for i in range(4)]  # left edges
COL_CXS = [x + COL_W/2 for x in COL_XS]

# ════════════════════════════════════════════════════════════════════════════
# SHARED INPUT
# ════════════════════════════════════════════════════════════════════════════
IN_X  = 0.22
IN_W  = 0.56
IN_CX = IN_X + IN_W/2
IN_CY = Y_IN_TOP - IN_H/2

rbox(ax, IN_X, Y_IN_TOP - IN_H, IN_W, IN_H,
     '#ECEFF1', C['input'], lw=1.2, radius=0.012)

# waveform icon left side
mini_wave(ax, IN_X + 0.055, IN_CY, 0.08, IN_H*0.7, '#78909C')

# text
label(ax, IN_CX + 0.02, IN_CY + 0.012,
      'Shared Input', fs=7.5, bold=True, color=C['input'])
label(ax, IN_CX + 0.02, IN_CY - 0.013,
      '256-sample raw waveform window  (32 ms @ 8 kHz)',
      fs=5.8, color='#546E7A', style='italic')

# waveform icon right side
np.random.seed(7)
mini_wave(ax, IN_X + IN_W - 0.055, IN_CY, 0.08, IN_H*0.7, '#78909C')

# ════════════════════════════════════════════════════════════════════════════
# FAN-OUT ARROWS  (input → 4 column headers)
# ════════════════════════════════════════════════════════════════════════════
for cx in COL_CXS:
    arr(ax, IN_CX, Y_IN_TOP - IN_H,
        cx,     COL_TOPS,
        color=C['arrow'], lw=0.75, headw=4)

# ════════════════════════════════════════════════════════════════════════════
# MODEL COLUMNS
# ════════════════════════════════════════════════════════════════════════════
models = [
    ('CNN1D',           C['cnn'],   LIGHT['cnn']),
    ('WaveNet-Small',   C['wave'],  LIGHT['wave']),
    ('ECAPA-VAD',       C['ecapa'], LIGHT['ecapa']),
    ('Transformer-VAD', C['trans'], LIGHT['trans']),
]

def col_blocks(ax, cx, bx, bw, col_top, col_bot, color, lcolor, name):
    """Draw one model's internal block stack."""
    avail  = col_top - col_bot
    # Header bar
    hh = 0.038
    rbox(ax, bx, col_top - hh, bw, hh,
         color, color, lw=0, radius=0.008)
    label(ax, cx, col_top - hh/2, name,
          fs=6.2, bold=True, color='white')

    # Internal blocks — tailored per model
    body_top = col_top - hh - 0.008
    body_bot = col_bot + 0.010
    body_h   = body_top - body_bot

    if name == 'CNN1D':
        layers = [
            ('Conv1D ×4',      'k=3, BN+ReLU',  stack_icon,  0.20),
            ('Channels',       '1→32→64→128→256',None,        0.14),
            ('Global Avg Pool','256-dim vector', None,         0.14),
            ('FC 256→64',      'ReLU',           None,         0.14),
            ('FC 64→2',        'logits',         mini_bar,    0.14),
        ]
        note = 'RF=9 samp\n~1.1 ms'

    elif name == 'WaveNet-Small':
        layers = [
            ('Dilated Conv',   'Group1: d=1,2,4,8', stack_icon, 0.20),
            ('Dilated Conv',   'Group2: d=1,2,4,8', stack_icon, 0.20),
            ('Skip Sum',       'Residual add',  None,           0.13),
            ('Strided Downsamp','→ compact repr',None,           0.13),
            ('Global Avg Pool','→ 2-class logit',mini_bar,      0.14),
        ]
        note = 'RF=121 samp\n~15 ms'

    elif name == 'ECAPA-VAD':
        layers = [
            ('Front-End Conv','128 ch, k=5',    None,          0.14),
            ('SE-Res2Net ×3', 'SE ratio=8',     stack_icon,    0.22),
            ('SE Attention',  'Channel-wise',   mini_grid,     0.16),
            ('ASP Pooling',   'μ ⊕ σ vectors',  None,          0.14),
            ('FC 128→2',      'logits',         mini_bar,      0.14),
        ]
        note = 'Attn pool\nμ and σ'

    else:  # Transformer-VAD
        layers = [
            ('Patch Split',   '16 × 16 samples',None,          0.14),
            ('Linear Proj',   '64-dim embed',   None,          0.14),
            ('+ CLS Token',   '17 tokens total',None,          0.12),
            ('Transformer ×2','4-head attn',    mini_grid,     0.22),
            ('CLS → FC 64→2', 'logits',         mini_bar,      0.14),
        ]
        note = 'Global\nself-attn'

    # normalize heights
    total_ratio = sum(r for _, _, _, r in layers)
    gap    = 0.008
    n      = len(layers)
    usable = body_h - gap*(n-1)

    y_cur = body_top
    for i, (main, sub, icon_fn, ratio) in enumerate(layers):
        bh_i = usable * (ratio / total_ratio)
        by_i = y_cur - bh_i

        # shade alternates slightly
        shade = lcolor if i%2==0 else '#FFFFFF'
        rbox(ax, bx+0.004, by_i, bw-0.008, bh_i,
             shade, color, lw=0.5, radius=0.006, zorder=3)

        # icon in left portion if provided
        icon_w = 0.045
        text_x = cx + 0.012 if icon_fn else cx

        if icon_fn:
            icon_cx = bx + 0.008 + icon_w/2
            icon_cy = by_i + bh_i/2
            icon_fn(ax, icon_cx, icon_cy,
                    icon_w, bh_i*0.75, color)

        # main label
        label(ax, text_x, by_i + bh_i*0.62,
              main, fs=5.8, bold=True, color=color)
        label(ax, text_x, by_i + bh_i*0.28,
              sub, fs=4.9, color='#666666', style='italic')

        # down arrow between blocks
        if i < n-1:
            arr(ax, cx, by_i,
                cx, by_i - gap,
                color=color, lw=0.6, headw=3)

        y_cur = by_i - gap

    # note badge bottom right of column
    rbox(ax, bx+bw-0.072, col_bot+0.002, 0.068, 0.030,
         color, color, lw=0, radius=0.005, alpha=0.15, zorder=3)
    label(ax, bx+bw-0.038, col_bot+0.017,
          note, fs=4.5, color=color, bold=False)


for i, (name, color, lcolor) in enumerate(models):
    bx  = COL_XS[i]
    cx  = COL_CXS[i]
    col_blocks(ax, cx, bx, COL_W,
               COL_TOPS, COL_BOT,
               color, lcolor, name)

# ════════════════════════════════════════════════════════════════════════════
# FAN-IN ARROWS  (4 column bottoms → shared output)
# ════════════════════════════════════════════════════════════════════════════
OUT_X  = 0.22
OUT_W  = 0.56
OUT_CX = OUT_X + OUT_W/2
OUT_CY = Y_OUT_TOP - OUT_H/2

for cx in COL_CXS:
    arr(ax, cx,     COL_BOT,
        OUT_CX, Y_OUT_TOP,
        color=C['arrow'], lw=0.75, headw=4)

# ════════════════════════════════════════════════════════════════════════════
# SHARED OUTPUT
# ════════════════════════════════════════════════════════════════════════════
rbox(ax, OUT_X, Y_OUT_TOP - OUT_H, OUT_W, OUT_H,
     '#ECEFF1', C['output'], lw=1.2, radius=0.012)

# bar chart icon left
mini_bar(ax, OUT_X + 0.05, OUT_CY, 0.07, OUT_H*0.65, '#78909C')

label(ax, OUT_CX + 0.02, OUT_CY + 0.012,
      'Shared Output', fs=7.5, bold=True, color=C['output'])
label(ax, OUT_CX + 0.02, OUT_CY - 0.013,
      r'$\hat{y}$ = argmax(softmax($\mathbf{z}$))    ·    0 = Noise,  1 = Speech',
      fs=5.8, color='#546E7A', style='italic')

# bar chart icon right
mini_bar(ax, OUT_X + OUT_W - 0.05, OUT_CY, 0.07, OUT_H*0.65, '#78909C')

# ════════════════════════════════════════════════════════════════════════════
# PARAM LEGEND at very bottom
# ════════════════════════════════════════════════════════════════════════════
param_data = [
    ('CNN1D',           C['cnn'],   '124,389'),
    ('WaveNet-Small',   C['wave'],  '175,237'),
    ('ECAPA-VAD',       C['ecapa'], '242,133'),
    ('Transformer-VAD', C['trans'],  '71,461'),
]
y_leg = 0.060
x_start = 0.04
col_w   = 0.237
for i, (nm, col, params) in enumerate(param_data):
    x = x_start + i * col_w
    rbox(ax, x, y_leg - 0.014, col_w - 0.01, 0.028,
         col, col, lw=0, radius=0.005, alpha=0.15, zorder=2)
    label(ax, x + (col_w-0.01)/2, y_leg - 0.001,
          f'{nm}  |  {params} params',
          fs=5.2, color=col, bold=False)

# bottom note
label(ax, 0.50, 0.024,
      'All models: raw waveform input (no hand-crafted features) · float32 · ONNX Runtime on ARM Cortex-A76',
      fs=5.0, color='#888888', style='italic')

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out = OUT / 'Figure_7.drawio.svg'
fig.savefig(str(out), bbox_inches='tight',
            format='svg', facecolor='white')
plt.close()

kb = out.stat().st_size // 1024
print(f'Saved: {out}  ({kb} KB)')
print('Open Figure_7.svg in draw.io to edit.')