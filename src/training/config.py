"""
src/training/config.py

Central configuration for all experiments.
Change EXPERIMENT variable to switch between experiments.
Everything else runs automatically.

Experiments:
  exp1_alaw_synthetic  : A-law + synthetic noise    (LG Gram, already done)
  exp2_clean_musan     : Clean PCM + real MUSAN     (baseline, MSI)
  exp3_alaw_musan      : A-law + real MUSAN         (main contribution, MSI)
"""

from pathlib import Path

# ─── Change this to switch experiments ────────────────────────────────────────
EXPERIMENT = "exp1_alaw_synthetic"
# EXPERIMENT = "exp2_clean_musan"
# EXPERIMENT = "exp3_alaw_musan"

# ─── Experiment definitions ───────────────────────────────────────────────────
EXPERIMENT_CONFIGS = {

    "exp1_alaw_synthetic": {
        "description":  "A-law codec simulation + synthetic noise (test run)",
        "use_alaw":     True,       # apply A-law encode->decode roundtrip
        "use_musan":    False,      # False = use musan_synthetic folder
        "fraction":     0.01,       # 1% of data  → LG Gram
        "batch_size":   64,
        "lr":           1e-3,
        "max_epochs":   50,
        "patience":     7,
    },

    "exp2_clean_musan": {
        "description":  "Clean PCM (no A-law) + real MUSAN noise (baseline)",
        "use_alaw":     False,      # NO A-law → clean PCM baseline
        "use_musan":    True,       # use real MUSAN dataset
        "fraction":     1.0,        # 100% of data → MSI
        "batch_size":   128,
        "lr":           1e-3,
        "max_epochs":   50,
        "patience":     7,
    },

    "exp3_alaw_musan": {
        "description":  "A-law codec simulation + real MUSAN (main contribution)",
        "use_alaw":     True,       # A-law simulation ON
        "use_musan":    True,       # real MUSAN dataset
        "fraction":     1.0,        # 100% of data → MSI
        "batch_size":   128,
        "lr":           1e-3,
        "max_epochs":   50,
        "patience":     7,
    },
}

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
RAW     = ROOT / "data" / "raw"
SPLITS  = ROOT / "data" / "splits" / EXPERIMENT
PROCESSED = ROOT / "data" / "processed" / EXPERIMENT

MODELS_DIR = ROOT / "outputs" / EXPERIMENT / "models"
STATS_DIR  = ROOT / "outputs" / EXPERIMENT / "stats"
FIGS_DIR   = ROOT / "outputs" / EXPERIMENT / "figures"

# create dirs
for d in [SPLITS, PROCESSED, MODELS_DIR, STATS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Active config ────────────────────────────────────────────────────────────
CFG = EXPERIMENT_CONFIGS[EXPERIMENT]

# ─── Constants ────────────────────────────────────────────────────────────────
SOURCE_SR   = 16000
TARGET_SR   = 8000
WINDOW_SIZE = 256
STRIDE      = 128
SEED        = 42


def print_config():
    print(f"\n{'='*55}")
    print(f"  Experiment : {EXPERIMENT}")
    print(f"  Description: {CFG['description']}")
    print(f"  A-law      : {CFG['use_alaw']}")
    print(f"  MUSAN      : {CFG['use_musan']}")
    print(f"  Fraction   : {CFG['fraction']*100:.1f}%")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  LR         : {CFG['lr']}")
    print(f"  Max epochs : {CFG['max_epochs']}")
    print(f"  Patience   : {CFG['patience']}")
    print(f"  Output dir : outputs/{EXPERIMENT}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    print_config()