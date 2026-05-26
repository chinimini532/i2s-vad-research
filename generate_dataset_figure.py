"""
generate_dataset_figures.py

Generates all dataset figures for the VAD paper.
Run from project root:
    python generate_dataset_figures.py

Output PDFs (place in Overleaf figures/ folder):
    fig_dataset_composition.pdf
    fig_dataset_waveform_spectrogram.pdf
    fig_dataset_split.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import librosa
import glob
import random

random.seed(42)
np.random.seed(42)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  12,
    "legend.fontsize": 10,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

# ── Paths — adjust if needed ──────────────────────────────────────────────────
LIBRI_PATH  = Path("data/raw/librispeech/LibriSpeech/train-clean-100")
MUSAN_PATH  = Path("data/raw/musan_synthetic")
OUTPUT_DIR  = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── G.711 A-law simulation ────────────────────────────────────────────────────
def alaw_encode_decode(x, A=87.6):
    """Full G.711 A-law encode-decode roundtrip on float32 array."""
    x = np.array(x, dtype=np.float64)
    x_int16 = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    import audioop
    alaw_bytes = audioop.lin2alaw(x_int16.tobytes(), 2)
    pcm_bytes  = audioop.alaw2lin(alaw_bytes, 2)
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0

# ── Load one speech file ──────────────────────────────────────────────────────
def load_speech(duration=1.0):
    flac_files = sorted(LIBRI_PATH.rglob("*.flac"))
    for fp in flac_files[:50]:
        audio, sr = librosa.load(str(fp), sr=16000, mono=True)
        if len(audio) >= int(duration * sr):
            audio = audio[:int(duration * sr)]
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio_8k = librosa.resample(audio, orig_sr=16000, target_sr=8000)
            return audio_8k
    raise FileNotFoundError("No LibriSpeech files found.")

# ── Load one noise file ───────────────────────────────────────────────────────
def load_noise(duration=1.0):
    wav_files = sorted(MUSAN_PATH.rglob("*.wav"))
    for fp in wav_files[:50]:
        try:
            audio, sr = librosa.load(str(fp), sr=8000, mono=True)
            if len(audio) >= int(duration * sr):
                audio = audio[:int(duration * sr)]
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                return audio
        except Exception:
            continue
    raise FileNotFoundError("No MUSAN files found.")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1: Dataset Composition
# ══════════════════════════════════════════════════════════════════════════════
def fig_dataset_composition():
    print("  Generating fig_dataset_composition.pdf...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Source corpora
    corpora     = ["LibriSpeech\ntrain-clean-100", "MUSAN\nNoise/Music"]
    hours       = [100, 109]
    files       = [28539, 2016]
    colors      = ["#1565C0", "#2E7D32"]

    ax = axes[0]
    bars = ax.bar(corpora, hours, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Duration (hours)")
    ax.set_title("Source Corpora", fontweight="bold")
    ax.set_ylim([0, 130])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, h, f in zip(bars, hours, files):
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h}h\n({f:,} files)",
                ha="center", va="bottom", fontsize=10)

    # Right: Processed windows
    categories = ["Speech\n(LibriSpeech)", "Noise\n(MUSAN)"]
    windows    = [200000, 200000]
    colors2    = ["#1565C0", "#2E7D32"]

    ax2 = axes[1]
    bars2 = ax2.bar(categories, windows, color=colors2,
                    edgecolor="black", linewidth=0.8, width=0.5)
    ax2.set_ylabel("Number of Windows")
    ax2.set_title("Processed Dataset (256-sample windows, 8 kHz)",
                  fontweight="bold")
    ax2.set_ylim([0, 260000])
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
    for bar, w in zip(bars2, windows):
        ax2.text(bar.get_x() + bar.get_width()/2, w + 3000,
                 f"{w//1000}K", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    ax2.text(0.5, 0.92,
             "Total: 400K windows  |  Split: 70 / 15 / 15",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow",
                       alpha=0.8))

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_dataset_composition.pdf"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2: Waveform + Spectrogram Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_dataset_waveform_spectrogram():
    print("  Generating fig_dataset_waveform_spectrogram.pdf...")

    speech_8k = load_speech(duration=0.5)
    noise_8k  = load_noise(duration=0.5)

    speech_alaw = alaw_encode_decode(speech_8k)
    noise_alaw  = alaw_encode_decode(noise_8k)

    t = np.arange(len(speech_8k)) / 8000 * 1000  # ms

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                             hspace=0.55, wspace=0.35)

    colors = {
        "speech_clean": "#1565C0",
        "speech_alaw":  "#C62828",
        "noise_clean":  "#2E7D32",
        "noise_alaw":   "#E65100",
    }

    # ── Row 1: Waveforms ──────────────────────────────────────────────────
    pairs_wave = [
        (speech_8k,  "Speech — Clean PCM (8 kHz)",    colors["speech_clean"]),
        (speech_alaw,"Speech — G.711 A-law",          colors["speech_alaw"]),
        (noise_8k,   "Noise — Clean PCM (8 kHz)",     colors["noise_clean"]),
        (noise_alaw, "Noise — G.711 A-law",           colors["noise_alaw"]),
    ]
    t_wave = np.arange(len(speech_8k)) / 8000 * 1000

    for col, (sig, title, color) in enumerate(pairs_wave):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_wave[:len(sig)], sig, color=color, lw=0.8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_ylim([-1, 1])
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    # ── Row 2: Spectrograms ───────────────────────────────────────────────
    from scipy.signal import spectrogram as sg

    pairs_spec = [
        (speech_8k,  colors["speech_clean"]),
        (speech_alaw,colors["speech_alaw"]),
        (noise_8k,   colors["noise_clean"]),
        (noise_alaw, colors["noise_alaw"]),
    ]

    for col, (sig, color) in enumerate(pairs_spec):
        ax  = fig.add_subplot(gs[1, col])
        f, t_s, Sxx = sg(sig, fs=8000, nperseg=64, noverlap=48)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax.pcolormesh(t_s * 1000, f, Sxx_db,
                            shading="gouraud", cmap="viridis",
                            vmin=-80, vmax=0)
        ax.axhline(y=300,  color="white", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(y=3400, color="white", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylim([0, 4000])
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Freq (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046,
                     label="dB").ax.tick_params(labelsize=7)

    # ── Row 3: Amplitude distributions ───────────────────────────────────
    pairs_dist = [
        (speech_8k,  speech_alaw, "#1565C0", "#C62828", "Speech Windows"),
        (noise_8k,   noise_alaw,  "#2E7D32", "#E65100", "Noise Windows"),
    ]

    for col, (clean, alaw, c1, c2, title) in enumerate(pairs_dist):
        ax = fig.add_subplot(gs[2, col*2 : col*2+2])
        ax.hist(clean, bins=120, density=True, alpha=0.6,
                color=c1, label="Clean PCM (8 kHz)", range=(-1, 1))
        ax.hist(alaw,  bins=120, density=True, alpha=0.6,
                color=c2, label="G.711 A-law",       range=(-1, 1))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Amplitude", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    # Row labels
    for row, label in enumerate(["Waveform", "Spectrogram",
                                  "Amplitude Distribution"]):
        fig.text(0.01, 0.85 - row * 0.32, label,
                 fontsize=10, fontweight="bold", rotation=90,
                 va="center", color="gray")

    out = OUTPUT_DIR / "fig_dataset_waveform_spectrogram.pdf"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Dataset Split
# ══════════════════════════════════════════════════════════════════════════════
def fig_dataset_split():
    print("  Generating fig_dataset_split.pdf...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: pie chart of split
    sizes  = [279999, 60001, 60000]
    labels = ["Train\n279,999 (70%)",
              "Validation\n60,001 (15%)",
              "Test\n60,000 (15%)"]
    colors = ["#1565C0", "#1976D2", "#42A5F5"]
    explode = (0.04, 0.04, 0.04)

    axes[0].pie(sizes, labels=labels, colors=colors,
                explode=explode, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("Dataset Split\n(Total: 400,000 windows)",
                       fontweight="bold")

    # Right: class balance across splits
    splits   = ["Train", "Validation", "Test"]
    speech_n = [139999, 30001, 30000]
    noise_n  = [140000, 30000, 30000]
    x = np.arange(len(splits))
    w = 0.35

    bars1 = axes[1].bar(x - w/2, speech_n, w,
                         label="Speech", color="#1565C0",
                         edgecolor="black", linewidth=0.6)
    bars2 = axes[1].bar(x + w/2, noise_n, w,
                         label="Noise", color="#2E7D32",
                         edgecolor="black", linewidth=0.6)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_ylabel("Number of Windows")
    axes[1].set_title("Class Balance Across Splits",
                       fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}K"))

    for bar in bars1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 500,
                     f"{int(h/1000)}K", ha="center",
                     fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 500,
                     f"{int(h/1000)}K", ha="center",
                     fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_dataset_split.pdf"
    plt.savefig(str(out))
    plt.close()
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print("  Generating dataset figures for paper")
    print(f"{'='*50}\n")

    fig_dataset_composition()

    try:
        fig_dataset_waveform_spectrogram()
    except FileNotFoundError as e:
        print(f"  [SKIP] Waveform figure: {e}")
        print("         Check LIBRI_PATH and MUSAN_PATH at top of script")

    fig_dataset_split()

    print(f"\n{'='*50}")
    print(f"  Done. PDFs saved to: {OUTPUT_DIR.resolve()}")
    print(f"  Upload these to Overleaf figures/ folder:")
    print(f"    fig_dataset_composition.pdf")
    print(f"    fig_dataset_waveform_spectrogram.pdf")
    print(f"    fig_dataset_split.pdf")
    print(f"{'='*50}\n")