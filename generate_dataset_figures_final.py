"""
generate_dataset_figures_final.py

Generates Figure_1, Figure_3, Figure_4, Figure_5, Figure_6
Run from project root: python generate_dataset_figures_final.py

Paper sequence:
    Figure_1.png  -> fig:alaw_background   (A-law transfer + distribution)
    Figure_2.png  -> fig:pipeline          [SKIP - draw.io]
    Figure_3.png  -> fig:dataset_composition
    Figure_4.png  -> fig:validation        (simulation vs real RTP)
    Figure_5.png  -> fig:dataset_split
    Figure_6.png  -> fig:dataset_waveform_spectrogram
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import librosa
import random

random.seed(42)
np.random.seed(42)

DPI = 400

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "savefig.dpi":      DPI,
    "savefig.bbox":     "tight",
    "savefig.format":   "png",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

LIBRI_PATH = Path("data/raw/librispeech/LibriSpeech/train-clean-100")
MUSAN_PATH = Path("data/raw/musan_synthetic")
VAL_DIR    = Path("outputs/validation")
OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── G.711 A-law ───────────────────────────────────────────────────────────────
def alaw_encode_decode(x):
    import audioop
    x_int16 = np.clip(
        np.round(np.array(x, dtype=np.float64) * 32767),
        -32768, 32767).astype(np.int16)
    alaw_bytes = audioop.lin2alaw(x_int16.tobytes(), 2)
    pcm_bytes  = audioop.alaw2lin(alaw_bytes, 2)
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0

def alaw_func(x, A=87.6):
    y = np.zeros_like(x, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    m1 = np.abs(x) < (1.0 / A)
    m2 = ~m1
    y[m1] = (A * np.abs(x[m1]) / (1 + np.log(A))) * np.sign(x[m1])
    y[m2] = ((1 + np.log(A * np.abs(x[m2]))) / (1 + np.log(A))) * np.sign(x[m2])
    return y

# ── Audio loaders ─────────────────────────────────────────────────────────────
def load_speech(duration=0.5):
    for fp in sorted(LIBRI_PATH.rglob("*.flac"))[:50]:
        audio, sr = librosa.load(str(fp), sr=16000, mono=True)
        if len(audio) >= int(duration * sr):
            audio = audio[:int(duration * sr)]
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            return librosa.resample(audio, orig_sr=16000, target_sr=8000)
    raise FileNotFoundError("No LibriSpeech files. Check LIBRI_PATH.")

def load_noise(duration=0.5):
    for fp in sorted(MUSAN_PATH.rglob("*.wav"))[:50]:
        try:
            audio, sr = librosa.load(str(fp), sr=8000, mono=True)
            if len(audio) >= int(duration * sr):
                audio = audio[:int(duration * sr)]
                return audio / (np.max(np.abs(audio)) + 1e-8)
        except Exception:
            continue
    raise FileNotFoundError("No MUSAN files. Check MUSAN_PATH.")

def collect_speech_samples(n=50000):
    clean_s, alaw_s = [], []
    for fp in sorted(LIBRI_PATH.rglob("*.flac"))[:80]:
        if len(clean_s) >= n:
            break
        try:
            audio, sr = librosa.load(str(fp), sr=16000, mono=True)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            a8k   = librosa.resample(audio, orig_sr=16000, target_sr=8000)
            al    = alaw_encode_decode(a8k)
            clean_s.extend(a8k.tolist())
            alaw_s.extend(al.tolist())
        except Exception:
            continue
    return np.array(clean_s[:n]), np.array(alaw_s[:n])


# ══════════════════════════════════════════════════════════════════════════════
# Figure_1.png — A-law transfer function + amplitude distribution
# NO titles inside figure
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_1():
    print("  Generating Figure_1.png (A-law background) ...")
    try:
        clean, alaw = collect_speech_samples(50000)
    except Exception as e:
        print(f"  [WARN] {e} — using synthetic fallback")
        t     = np.linspace(0, 5, 40000)
        clean = (np.sin(2*np.pi*150*t) * np.exp(-0.3*(t%0.5))).astype(np.float32)
        clean = clean / (np.max(np.abs(clean)) + 1e-8)
        alaw  = alaw_encode_decode(clean)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Left: transfer function — NO title
    ax = axes[0]
    xv = np.linspace(-1, 1, 2000)
    yv = alaw_func(xv)
    ax.plot(xv, yv, color="#1565C0", lw=1.8, label="A-law encoded")
    ax.plot(xv, xv, color="gray", lw=1.2, ls="--", label="Linear (no encoding)")
    ax.axvline(x= 1/87.6, color="#C62828", lw=0.9, ls=":",
               label=r"$|x|=1/A\approx0.011$")
    ax.axvline(x=-1/87.6, color="#C62828", lw=0.9, ls=":")
    ax.set_xlabel(r"Input amplitude $x$")
    ax.set_ylabel(r"Encoded output $y$")
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_aspect("equal")

    # Right: distribution — NO title
    ax2 = axes[1]
    ax2.hist(clean, bins=150, density=True, alpha=0.65,
             color="#1565C0", label="Clean PCM (8 kHz)", range=(-1, 1))
    ax2.hist(alaw,  bins=150, density=True, alpha=0.65,
             color="#C62828", label="G.711 A-law (8 kHz)", range=(-1, 1))
    ax2.set_xlabel("Amplitude")
    ax2.set_ylabel("Probability density")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2, linewidth=0.4)
    ax2.set_xlim([-1, 1])

    plt.tight_layout(pad=0.8)
    out = OUTPUT_DIR / "Figure_1.png"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}  ({DPI} DPI)")


# ══════════════════════════════════════════════════════════════════════════════
# Figure_3.png — Dataset composition
# NO top-level title, subplot axis labels only
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_3():
    print("  Generating Figure_3.png (dataset composition) ...")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    corpora = ["LibriSpeech\ntrain-clean-100", "MUSAN\nNoise/Music"]
    hours   = [100, 109]
    files   = [28539, 2016]
    colors  = ["#1565C0", "#2E7D32"]

    ax = axes[0]
    bars = ax.bar(corpora, hours, color=colors,
                  edgecolor="black", linewidth=0.7, width=0.45)
    ax.set_ylabel("Duration (hours)")
    ax.set_ylim([0, 135])
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    for bar, h, f in zip(bars, hours, files):
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h}h / {f:,} files",
                ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    categories = ["Speech\n(LibriSpeech)", "Noise\n(MUSAN)"]
    windows    = [200000, 200000]
    bars2 = ax2.bar(categories, windows, color=colors,
                    edgecolor="black", linewidth=0.7, width=0.45)
    ax2.set_ylabel("Number of Windows")
    ax2.set_ylim([0, 260000])
    ax2.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
    for bar, w in zip(bars2, windows):
        ax2.text(bar.get_x() + bar.get_width()/2, w + 3000,
                 f"{w//1000}K", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.text(0.5, 0.91, "Total: 400K  |  Split: 70 / 15 / 15",
             transform=ax2.transAxes, ha="center", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = OUTPUT_DIR / "Figure_3.png"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}  ({DPI} DPI)")


# ══════════════════════════════════════════════════════════════════════════════
# Figure_4.png — Simulation validation vs real RTP
# NO top-level title, NO subplot ax.set_title()
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_4():
    print("  Generating Figure_4.png (validation) ...")

    REAL_BYTES = VAL_DIR / "real_rtp_bytes.npy"
    SIM_BYTES  = VAL_DIR / "sim_bytes.npy"
    REAL_PCM   = VAL_DIR / "real_pcm.npy"
    SIM_PCM    = VAL_DIR / "sim_pcm.npy"

    try:
        real_bytes = np.load(str(REAL_BYTES)).astype(np.uint8)
        sim_bytes  = np.load(str(SIM_BYTES)).astype(np.uint8)
        real_pcm   = np.load(str(REAL_PCM)).astype(np.float32)
        sim_pcm    = np.load(str(SIM_PCM)).astype(np.float32)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        print(f"         Expected in: {VAL_DIR.resolve()}")
        print("         Files: real_rtp_bytes.npy, sim_bytes.npy, real_pcm.npy, sim_pcm.npy")
        return

    # Stats
    N          = len(real_bytes)
    byte_match = np.mean(real_bytes == sim_bytes)
    counts_r   = np.bincount(real_bytes, minlength=256).astype(float)
    counts_s   = np.bincount(sim_bytes,  minlength=256).astype(float)
    p = counts_r / counts_r.sum()
    q = counts_s / counts_s.sum()
    mask = (p > 0) & (q > 0)
    kl   = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(real_pcm, sim_pcm)
    dur = N / 8000

    fig = plt.figure(figsize=(7.0, 5.5))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             hspace=0.52, wspace=0.42,
                             height_ratios=[1.1, 1.0])

    # Top left: byte distribution — NO title
    ax0 = fig.add_subplot(gs[0, 0:2])
    xb  = np.arange(256)
    ax0.bar(xb, p, width=1.0, alpha=0.65, color="#1565C0",
            label="Real G.711 PCMA (Wireshark)")
    ax0.bar(xb, q, width=1.0, alpha=0.55, color="#C62828",
            label="Software simulation (audioop)")
    ax0.text(0.02, 0.97,
             f"KL divergence = {kl:.4f}\nByte match = {byte_match*100:.1f}%",
             transform=ax0.transAxes, fontsize=7, va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow",
                       alpha=0.8, edgecolor="gray", lw=0.5))
    ax0.set_xlabel("A-law encoded byte value")
    ax0.set_ylabel("Probability density")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.grid(True, alpha=0.2, linewidth=0.4)

    # Top right: stats table — NO title
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.axis("off")
    rows = [
        ["Metric",      "Real RTP",                  "Simulation"],
        ["Duration",    f"{dur:.1f}s",               f"{dur:.1f}s"],
        ["PCM Mean",    f"{np.mean(real_pcm):.5f}",  f"{np.mean(sim_pcm):.5f}"],
        ["PCM Std",     f"{np.std(real_pcm):.5f}",   f"{np.std(sim_pcm):.5f}"],
        ["Byte Match",  f"{byte_match*100:.1f}%",     "---"],
        ["KL Div",      f"{kl:.4f}",                 "---"],
        ["KS stat",     f"{ks_stat:.4f}",            f"p={ks_p:.2e}"],
        ["Sample Rate", "8000 Hz",                   "8000 Hz"],
        ["Codec",       "G.711 PCMA",                "audioop G.711"],
    ]
    tbl = ax1.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1.0, 1.25)
    for j in range(3):
        tbl[(0,j)].set_facecolor("#1565C0")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)-1):
        for j in range(3):
            tbl[(i+1,j)].set_facecolor(
                "#f0f4ff" if i%2==0 else "#ffffff")

    # Bottom left: PCM waveform — NO title
    ax2 = fig.add_subplot(gs[1, 0])
    n_show = min(1000, len(real_pcm))
    t_ms   = np.arange(n_show) / 8000 * 1000
    ax2.plot(t_ms, real_pcm[:n_show], color="#1565C0",
             lw=0.7, alpha=0.9, label="Real RTP")
    ax2.plot(t_ms, sim_pcm[:n_show],  color="#C62828",
             lw=0.7, alpha=0.7, ls="--", label="Simulation")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.2, linewidth=0.4)
    ax2.text(0.5, 0.97, "Decoded PCM (125 ms)",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=7, color="gray")

    # Bottom middle: PSD — NO title
    ax3 = fig.add_subplot(gs[1, 1])
    from scipy.signal import welch
    f_r, psd_r = welch(real_pcm, fs=8000, nperseg=256)
    f_s, psd_s = welch(sim_pcm,  fs=8000, nperseg=256)
    ax3.semilogy(f_r, psd_r, color="#1565C0", lw=1.0, alpha=0.9, label="Real RTP")
    ax3.semilogy(f_s, psd_s, color="#C62828", lw=1.0, alpha=0.7, ls="--", label="Simulation")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD")
    ax3.set_xlim([0, 4000])
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.2, linewidth=0.4)
    ax3.text(0.5, 0.97, "Power spectral density (0--4000 Hz)",
             transform=ax3.transAxes, ha="center", va="top",
             fontsize=7, color="gray")

    # Bottom right: Q-Q — NO title
    ax4 = fig.add_subplot(gs[1, 2])
    n_qq = min(5000, len(real_pcm), len(sim_pcm))
    q_r  = np.percentile(real_pcm[:n_qq], np.linspace(0, 100, 200))
    q_s  = np.percentile(sim_pcm[:n_qq],  np.linspace(0, 100, 200))
    ax4.scatter(q_r, q_s, s=8, color="#6A1B9A", alpha=0.7, zorder=3)
    lim = max(np.abs(q_r).max(), np.abs(q_s).max()) * 1.05
    ax4.plot([-lim, lim], [-lim, lim], "k--", lw=1.0,
             label="y=x (perfect match)")
    ax4.set_xlabel("Real RTP quantiles")
    ax4.set_ylabel("Simulation quantiles")
    ax4.legend(fontsize=6)
    ax4.grid(True, alpha=0.2, linewidth=0.4)
    ax4.text(0.5, 0.97, "Q-Q plot",
             transform=ax4.transAxes, ha="center", va="top",
             fontsize=7, color="gray")

    plt.tight_layout(pad=0.8)
    out = OUTPUT_DIR / "Figure_4.png"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}  ({DPI} DPI)")


# ══════════════════════════════════════════════════════════════════════════════
# Figure_5.png — Dataset split
# NO top-level title
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_5():
    print("  Generating Figure_5.png (dataset split) ...")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    sizes   = [279999, 60001, 60000]
    labels  = ["Train\n279,999 (70%)", "Validation\n60,001 (15%)",
               "Test\n60,000 (15%)"]
    colors  = ["#1565C0", "#1976D2", "#42A5F5"]
    explode = (0.04, 0.04, 0.04)
    axes[0].pie(sizes, labels=labels, colors=colors, explode=explode,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 8})

    splits   = ["Train", "Validation", "Test"]
    speech_n = [139999, 30001, 30000]
    noise_n  = [140000, 30000, 30000]
    x = np.arange(len(splits))
    w = 0.35
    bars1 = axes[1].bar(x - w/2, speech_n, w, label="Speech",
                         color="#1565C0", edgecolor="black", linewidth=0.6)
    bars2 = axes[1].bar(x + w/2, noise_n,  w, label="Noise",
                         color="#2E7D32", edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_ylabel("Number of Windows")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3, linewidth=0.5)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}K"))
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 500,
                     f"{int(h/1000)}K", ha="center", fontsize=7)

    plt.tight_layout()
    out = OUTPUT_DIR / "Figure_5.png"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}  ({DPI} DPI)")


# ══════════════════════════════════════════════════════════════════════════════
# Figure_6.png — Waveform + spectrogram + amplitude distribution
# NO top-level title, subplot labels only inside panels
# ══════════════════════════════════════════════════════════════════════════════
def gen_figure_6():
    print("  Generating Figure_6.png (waveform/spectrogram) ...")
    speech_8k   = load_speech(duration=0.5)
    noise_8k    = load_noise(duration=0.5)
    speech_alaw = alaw_encode_decode(speech_8k)
    noise_alaw  = alaw_encode_decode(noise_8k)

    fig = plt.figure(figsize=(7.0, 7.5))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.38)
    C   = {"sc":"#1565C0","sa":"#C62828","nc":"#2E7D32","na":"#E65100"}

    # Row 1: Waveforms — labels only, no titles
    pairs_w = [
        (speech_8k,   "(a) Speech, Clean PCM",  C["sc"]),
        (speech_alaw, "(b) Speech, G.711 A-law", C["sa"]),
        (noise_8k,    "(c) Noise, Clean PCM",   C["nc"]),
        (noise_alaw,  "(d) Noise, G.711 A-law",  C["na"]),
    ]
    t_ms = np.arange(len(speech_8k)) / 8000 * 1000
    for col, (sig, label, color) in enumerate(pairs_w):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_ms[:len(sig)], sig, color=color, lw=0.6)
        ax.text(0.5, 0.97, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=color,
                fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=7)
        ax.set_ylabel("Amplitude", fontsize=7)
        ax.set_ylim([-1, 1])
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.tick_params(labelsize=6)

    # Row 2: Spectrograms
    from scipy.signal import spectrogram as sg
    pairs_s = [(speech_8k,C["sc"]),(speech_alaw,C["sa"]),
               (noise_8k,C["nc"]),(noise_alaw,C["na"])]
    for col, (sig, _) in enumerate(pairs_s):
        ax = fig.add_subplot(gs[1, col])
        f, t_s, Sxx = sg(sig, fs=8000, nperseg=64, noverlap=48)
        im = ax.pcolormesh(t_s*1000, f, 10*np.log10(Sxx+1e-10),
                            shading="gouraud", cmap="viridis",
                            vmin=-80, vmax=0)
        ax.axhline(y=300,  color="white", ls="--", lw=0.7, alpha=0.7)
        ax.axhline(y=3400, color="white", ls="--", lw=0.7, alpha=0.7)
        ax.set_ylim([0, 4000])
        ax.set_xlabel("Time (ms)", fontsize=7)
        ax.set_ylabel("Freq (Hz)", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046,
                     label="dB").ax.tick_params(labelsize=6)

    # Row 3: Distributions
    pairs_d = [
        (speech_8k, speech_alaw, C["sc"], C["sa"], "(e) Speech Windows"),
        (noise_8k,  noise_alaw,  C["nc"], C["na"], "(f) Noise Windows"),
    ]
    for col, (clean, alaw, c1, c2, label) in enumerate(pairs_d):
        ax = fig.add_subplot(gs[2, col*2:col*2+2])
        ax.hist(clean, bins=120, density=True, alpha=0.6,
                color=c1, label="Clean PCM (8 kHz)", range=(-1,1))
        ax.hist(alaw,  bins=120, density=True, alpha=0.6,
                color=c2, label="G.711 A-law",       range=(-1,1))
        ax.text(0.5, 0.97, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=8,
                fontweight="bold", color="gray")
        ax.set_xlabel("Amplitude", fontsize=8)
        ax.set_ylabel("Density",   fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.tick_params(labelsize=7)

    # Row side labels
    for row, lbl in enumerate(["Waveform","Spectrogram","Amplitude Distribution"]):
        fig.text(0.005, 0.85 - row*0.32, lbl,
                 fontsize=8, fontweight="bold", rotation=90,
                 va="center", color="gray")

    out = OUTPUT_DIR / "Figure_6.png"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}  ({DPI} DPI)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  Generating dataset + background + validation figures")
    print(f"  Output: {OUTPUT_DIR.resolve()}  |  DPI: {DPI}")
    print(f"{'='*55}\n")

    gen_figure_1()   # A-law background
    # Figure_2 is draw.io pipeline — skip
    gen_figure_3()   # Dataset composition
    gen_figure_4()   # Validation
    gen_figure_5()   # Dataset split
    gen_figure_6()   # Waveform/spectrogram

    print(f"\n{'='*55}")
    print("  Done. Files saved:")
    for n in [1,3,4,5,6]:
        p = OUTPUT_DIR / f"Figure_{n}.png"
        if p.exists():
            kb = p.stat().st_size // 1024
            print(f"    Figure_{n}.png  ({kb} KB)")
    print()
    print("  SKIP (make in draw.io):")
    print("    Figure_2.png  — pipeline flowchart")
    print(f"{'='*55}\n")