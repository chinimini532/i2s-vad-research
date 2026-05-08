"""
src/data/preprocess.py

Reads experiment config from src/training/config.py
Automatically applies A-law or clean PCM based on config.
Automatically uses MUSAN or synthetic noise based on config.
Saves processed data to data/processed/{EXPERIMENT}/
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import soundfile as sf
import audioop
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import (
    CFG, EXPERIMENT, RAW, PROCESSED,
    SOURCE_SR, TARGET_SR, WINDOW_SIZE, STRIDE, SEED,
    print_config
)


# ─── A-law simulation ─────────────────────────────────────────────────────────
def apply_alaw_roundtrip(samples_float: np.ndarray) -> np.ndarray:
    """Simulate TP3094 codec: float -> int16 -> A-law -> int16 -> float"""
    samples_int16  = np.clip(samples_float, -1.0, 1.0)
    samples_int16  = (samples_int16 * 32767).astype(np.int16)
    pcm_bytes      = samples_int16.tobytes()
    alaw_bytes     = audioop.lin2alaw(pcm_bytes, 2)
    pcm_restored   = audioop.alaw2lin(alaw_bytes, 2)
    restored_int16 = np.frombuffer(pcm_restored, dtype=np.int16)
    return restored_int16.astype(np.float32) / 32767.0


# ─── Resampling ───────────────────────────────────────────────────────────────
def resample_audio(samples: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return samples
    common = gcd(from_sr, to_sr)
    up     = to_sr   // common
    down   = from_sr // common
    return resample_poly(samples, up, down).astype(np.float32)


# ─── Windowing ────────────────────────────────────────────────────────────────
def make_windows(samples: np.ndarray) -> np.ndarray:
    if len(samples) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32)
    windows = []
    start   = 0
    while start + WINDOW_SIZE <= len(samples):
        windows.append(samples[start : start + WINDOW_SIZE])
        start += STRIDE
    return np.array(windows, dtype=np.float32)


# ─── Process one file ─────────────────────────────────────────────────────────
def process_file(filepath: Path, source_sr: int,
                 use_alaw: bool) -> np.ndarray:
    """
    load -> resample to 8kHz -> (optional A-law) -> window
    use_alaw=True  : simulates TP3094 codec distortion
    use_alaw=False : clean PCM baseline (no codec simulation)
    """
    samples, sr = sf.read(str(filepath), dtype="float32")

    if samples.ndim == 2:
        samples = samples[:, 0]

    samples_8k = resample_audio(samples, sr, TARGET_SR)

    if use_alaw:
        samples_8k = apply_alaw_roundtrip(samples_8k)

    return make_windows(samples_8k)


# ─── Collect files ────────────────────────────────────────────────────────────
def get_speech_files() -> list:
    speech_dir = RAW / "librispeech" / "LibriSpeech" / "train-clean-100"
    all_files  = sorted(speech_dir.rglob("*.flac"))

    if not all_files:
        raise FileNotFoundError(f"No .flac files in {speech_dir}")

    n        = max(1, int(len(all_files) * CFG["fraction"]))
    selected = all_files[:n]
    print(f"  Speech : {len(all_files):,} total -> using {len(selected):,} "
          f"({CFG['fraction']*100:.1f}%)")
    return selected


def get_noise_files() -> list:
    if CFG.get("use_demand", False):
        noise_dir = RAW / "demand"
        if not noise_dir.exists():
            raise FileNotFoundError("DEMAND not found.")
        all_files = sorted(noise_dir.rglob("*.wav"))
        label     = "DEMAND"
    elif CFG["use_musan"]:
        noise_dir = RAW / "musan"
        if not noise_dir.exists():
            raise FileNotFoundError("MUSAN not found.")
        all_files = sorted(noise_dir.rglob("*.wav"))
        label     = "MUSAN (real)"
    else:
        noise_dir = RAW / "musan_synthetic"
        all_files = sorted(noise_dir.rglob("*.wav"))
        label     = "musan_synthetic"

    if not all_files:
        raise FileNotFoundError("No noise files found.")

    n        = max(1, int(len(all_files) * CFG["fraction"]))
    selected = all_files[:n]
    print(f"  Noise  : {len(all_files):,} total -> using {len(selected):,} "
          f"({CFG['fraction']*100:.1f}%) [{label}]")
    return selected


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print_config()
    print(f"\n  A-law processing : {'ON' if CFG['use_alaw'] else 'OFF (clean PCM)'}")
    print(f"  Saving to        : {PROCESSED}\n")

    speech_files = get_speech_files()
    noise_files  = get_noise_files()

    # ── speech (label=1) ──────────────────────────────────────────────────
    print("\n[1/2] Processing speech files...")
    speech_windows = []
    for fp in tqdm(speech_files, desc="Speech"):
        try:
            w = process_file(fp, SOURCE_SR, use_alaw=CFG["use_alaw"])
            if len(w) > 0:
                speech_windows.append(w)
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")

    X_speech = np.concatenate(speech_windows, axis=0)
    y_speech = np.ones(len(X_speech), dtype=np.int64)
    print(f"  -> {len(X_speech):,} speech windows")

    # ── noise (label=0) ───────────────────────────────────────────────────
    print("\n[2/2] Processing noise files...")
    noise_windows = []
    for fp in tqdm(noise_files, desc="Noise"):
        try:
            # noise files are already at various sample rates
            # soundfile reads correct sr automatically
            w = process_file(fp, TARGET_SR, use_alaw=False)
            if len(w) > 0:
                noise_windows.append(w)
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")

    X_noise = np.concatenate(noise_windows, axis=0)
    y_noise = np.zeros(len(X_noise), dtype=np.int64)
    print(f"  -> {len(X_noise):,} noise windows")

    # ── balance + shuffle ─────────────────────────────────────────────────
    MAX_WINDOWS = 200_000
    min_count   = min(len(X_speech), len(X_noise), MAX_WINDOWS)
    print(f"\n  Capping at {MAX_WINDOWS:,} windows per class")
    rng         = np.random.default_rng(seed=SEED)
    speech_idx  = rng.choice(len(X_speech), min_count, replace=False)
    noise_idx   = rng.choice(len(X_noise),  min_count, replace=False)

    X = np.concatenate([X_speech[speech_idx], X_noise[noise_idx]])
    y = np.concatenate([y_speech[speech_idx], y_noise[noise_idx]])

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # ── save ──────────────────────────────────────────────────────────────
    np.save(str(PROCESSED / "X.npy"), X)
    np.save(str(PROCESSED / "y.npy"), y)

    print(f"\n{'='*55}")
    print(f"  Saved X.npy -> shape {X.shape}  ({X.nbytes/1e6:.1f} MB)")
    print(f"  Saved y.npy -> shape {y.shape}")
    print(f"  Speech : {min_count:,}  Noise : {min_count:,}")
    print(f"  Total  : {len(X):,} windows")
    print(f"{'='*55}")
    print(f"\nNext: python src/data/split.py")


if __name__ == "__main__":
    main()