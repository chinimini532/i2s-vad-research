"""
src/training/config.py

Central configuration for all experiments.
Change EXPERIMENT variable to switch between experiments.
"""

from pathlib import Path

# ─── Change this to switch experiments ────────────────────────────────────────
EXPERIMENT = "exp2_clean_musan"
# EXPERIMENT = "exp3_alaw_demand"

# ─── Experiment definitions ───────────────────────────────────────────────────
EXPERIMENT_CONFIGS = {

    "exp1_alaw_synthetic": {
        "description":  "A-law + synthetic noise (test run)",
        "use_alaw":     True,
        "use_musan":    False,
        "use_demand":   False,
        "fraction":     1.0,
        "batch_size":   64,
        "lr":           1e-3,
        "max_epochs":   30,
        "patience":     7,
    },

    "exp2_clean_demand": {
        "description":  "Clean PCM (no A-law) + DEMAND noise (baseline)",
        "use_alaw":     False,
        "use_musan":    False,
        "use_demand":   True,
        "fraction":     1.0,
        "batch_size":   128,
        "lr":           0.0003,
        "max_epochs":   50,
        "patience":     7,
    },

    "exp3_alaw_demand": {
        "description":  "A-law codec simulation + DEMAND noise (main contribution)",
        "use_alaw":     True,
        "use_musan":    False,
        "use_demand":   True,
        "fraction":     1.0,
        "batch_size":   128,
        "lr":           0.0003,
        "max_epochs":   50,
        "patience":     7,
    },

    "exp2_clean_musan": {
        "description":  "Clean PCM (no A-law) + real MUSAN noise (baseline)",
        "use_alaw":     False,
        "use_musan":    True,
        "use_demand":   False,
        "fraction":     1.0,
        "batch_size":   128,
        "lr":           0.0003,
        "max_epochs":   50,
        "patience":     7,
    },

    "exp3_alaw_musan": {
        "description":  "A-law codec simulation + real MUSAN (main contribution)",
        "use_alaw":     True,
        "use_musan":    True,
        "use_demand":   False,
        "fraction":     1.0,
        "batch_size":   128,
        "lr":           0.0003,
        "max_epochs":   50,
        "patience":     7,
    },
}

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
RAW       = ROOT / "data" / "raw"
SPLITS    = ROOT / "data"    / "splits"    / EXPERIMENT
PROCESSED = ROOT / "data"    / "processed" / EXPERIMENT

MODELS_DIR = ROOT / "outputs" / EXPERIMENT / "models"
STATS_DIR  = ROOT / "outputs" / EXPERIMENT / "stats"
FIGS_DIR   = ROOT / "outputs" / EXPERIMENT / "figures"

for d in [SPLITS, PROCESSED, MODELS_DIR, STATS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    print(f"  MUSAN     : {CFG.get('use_musan', False)}")
    print(f"  Fraction   : {CFG['fraction']*100:.1f}%")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  LR         : {CFG['lr']}")
    print(f"  Max epochs : {CFG['max_epochs']}")
    print(f"  Patience   : {CFG['patience']}")
    print(f"  Output dir : outputs/{EXPERIMENT}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    print_config()