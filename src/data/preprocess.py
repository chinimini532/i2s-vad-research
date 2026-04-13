"""
src/data/preprocess.py

What this file does (in simple words):
---------------------------------------
1. Finds all .flac speech files in LibriSpeech
2. Finds all .wav noise files in musan_synthetic
3. For each file:
   a. Loads the audio
   b. Resamples from 16kHz to 8kHz  (your codec runs at 8kHz)
   c. Applies A-law encode -> decode  (simulates TP3094 codec distortion)
   d. Cuts into fixed 256-sample windows (32ms each at 8kHz)
   e. Labels each window: 1 = speech, 0 = noise/silence
4. Saves everything as numpy arrays in data/processed/
   - X.npy  shape (N, 256)  all windows
   - y.npy  shape (N,)      all labels

To change fraction: edit FRACTION variable below.
   0.01 = use only 1% of files   (LG Gram testing)
   1.00 = use all files          (MSI full training)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import audioop
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample_poly
from math import gcd

# ─── Configuration ────────────────────────────────────────────────────────────
FRACTION    = 0.01     # change to 1.0 on MSI for full training
SOURCE_SR   = 16000    # LibriSpeech sample rate
TARGET_SR   = 8000     # TP3094 codec sample rate
WINDOW_SIZE = 256      # samples per window = 32ms at 8kHz
STRIDE      = 128      # 50% overlap between windows
SEED        = 42

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# ─── A-law simulation ─────────────────────────────────────────────────────────
def apply_alaw_roundtrip(samples_float: np.ndarray) -> np.ndarray:
    """
    Simulates what your TP3094 codec does to audio.

    Steps:
      1. Convert float32 [-1, 1] to int16 PCM
      2. Encode int16 to A-law 8-bit  (codec compresses audio)
      3. Decode A-law 8-bit to int16  (codec expands audio back)
      4. Convert int16 to float32 [-1, 1]

    The encode->decode roundtrip introduces the same quantization
    distortion that your real hardware produces.
    """
    samples_int16 = np.clip(samples_float, -1.0, 1.0)
    samples_int16 = (samples_int16 * 32767).astype(np.int16)
    pcm_bytes     = samples_int16.tobytes()
    alaw_bytes    = audioop.lin2alaw(pcm_bytes, 2)
    pcm_restored  = audioop.alaw2lin(alaw_bytes, 2)
    restored_int16 = np.frombuffer(pcm_restored, dtype=np.int16)
    return restored_int16.astype(np.float32) / 32767.0


# ─── Resampling ───────────────────────────────────────────────────────────────
def resample_audio(samples: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio from one sample rate to another."""
    if from_sr == to_sr:
        return samples
    common = gcd(from_sr, to_sr)
    up     = to_sr   // common
    down   = from_sr // common
    return resample_poly(samples, up, down).astype(np.float32)


# ─── Windowing ────────────────────────────────────────────────────────────────
def make_windows(samples: np.ndarray) -> np.ndarray:
    """
    Cut a 1D array of audio samples into overlapping windows.
    Returns shape: (num_windows, WINDOW_SIZE)
    """
    if len(samples) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32)

    windows = []
    start = 0
    while start + WINDOW_SIZE <= len(samples):
        windows.append(samples[start : start + WINDOW_SIZE])
        start += STRIDE

    return np.array(windows, dtype=np.float32)


# ─── Process one audio file ───────────────────────────────────────────────────
def process_file(filepath: Path, source_sr: int) -> np.ndarray:
    """
    Full pipeline for one audio file:
      load -> resample to 8kHz -> A-law roundtrip -> window
    Returns shape: (num_windows, WINDOW_SIZE)
    """
    samples, sr = sf.read(str(filepath), dtype="float32")

    if samples.ndim == 2:
        samples = samples[:, 0]

    samples_8k   = resample_audio(samples, sr, TARGET_SR)
    samples_alaw = apply_alaw_roundtrip(samples_8k)
    windows      = make_windows(samples_alaw)

    return windows


# ─── Collect file paths ───────────────────────────────────────────────────────
def get_speech_files() -> list:
    speech_dir = RAW / "librispeech" / "LibriSpeech" / "train-clean-100"
    all_files  = sorted(speech_dir.rglob("*.flac"))

    if not all_files:
        raise FileNotFoundError(f"No .flac files found in {speech_dir}")

    n        = max(1, int(len(all_files) * FRACTION))
    selected = all_files[:n]
    print(f"  Speech files : {len(all_files):,} total -> using {len(selected):,} "
          f"({FRACTION*100:.1f}%)")
    return selected


def get_noise_files() -> list:
    musan_dir = RAW / "musan"
    if musan_dir.exists():
        all_files = sorted(musan_dir.rglob("*.wav"))
        label     = "MUSAN"
    else:
        musan_dir = RAW / "musan_synthetic"
        all_files = sorted(musan_dir.rglob("*.wav"))
        label     = "musan_synthetic"

    if not all_files:
        raise FileNotFoundError("No noise .wav files found.")

    n        = max(1, int(len(all_files) * FRACTION))
    selected = all_files[:n]
    print(f"  Noise files  : {len(all_files):,} total -> using {len(selected):,} "
          f"({FRACTION*100:.1f}%) [{label}]")
    return selected


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  Preprocessing  |  fraction = {FRACTION*100:.1f}%")
    print(f"  Window size    |  {WINDOW_SIZE} samples = "
          f"{WINDOW_SIZE/TARGET_SR*1000:.0f}ms at {TARGET_SR}Hz")
    print(f"  Stride         |  {STRIDE} samples = "
          f"{STRIDE/TARGET_SR*1000:.0f}ms overlap")
    print(f"  Target SR      |  {TARGET_SR} Hz (TP3094 codec rate)")
    print(f"  A-law          |  encode->decode roundtrip enabled")
    print(f"{'='*55}\n")

    speech_files = get_speech_files()
    noise_files  = get_noise_files()

    # ── process speech (label = 1) ────────────────────────────────────────
    print("\n[1/2] Processing speech files...")
    speech_windows = []
    for fp in tqdm(speech_files, desc="Speech"):
        try:
            w = process_file(fp, SOURCE_SR)
            if len(w) > 0:
                speech_windows.append(w)
        except Exception as e:
            print(f"  [warn] skipped {fp.name}: {e}")

    X_speech = np.concatenate(speech_windows, axis=0)
    y_speech = np.ones(len(X_speech), dtype=np.int64)
    print(f"  -> {len(X_speech):,} speech windows created")

    # ── process noise (label = 0) ─────────────────────────────────────────
    print("\n[2/2] Processing noise files...")
    noise_windows = []
    for fp in tqdm(noise_files, desc="Noise"):
        try:
            w = process_file(fp, TARGET_SR)
            if len(w) > 0:
                noise_windows.append(w)
        except Exception as e:
            print(f"  [warn] skipped {fp.name}: {e}")

    X_noise = np.concatenate(noise_windows, axis=0)
    y_noise = np.zeros(len(X_noise), dtype=np.int64)
    print(f"  -> {len(X_noise):,} noise windows created")

    # ── balance classes ───────────────────────────────────────────────────
    min_count = min(len(X_speech), len(X_noise))
    print(f"\n  Balancing classes -> {min_count:,} windows each")

    rng         = np.random.default_rng(seed=SEED)
    speech_idx  = rng.choice(len(X_speech), min_count, replace=False)
    noise_idx   = rng.choice(len(X_noise),  min_count, replace=False)

    X = np.concatenate([X_speech[speech_idx], X_noise[noise_idx]], axis=0)
    y = np.concatenate([y_speech[speech_idx], y_noise[noise_idx]], axis=0)

    # ── shuffle ───────────────────────────────────────────────────────────
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # ── save ──────────────────────────────────────────────────────────────
    np.save(str(PROCESSED / "X.npy"), X)
    np.save(str(PROCESSED / "y.npy"), y)

    print(f"\n{'='*55}")
    print(f"  Saved X.npy  ->  shape {X.shape}  ({X.nbytes/1e6:.1f} MB)")
    print(f"  Saved y.npy  ->  shape {y.shape}")
    print(f"  Speech windows : {min_count:,}  (label=1)")
    print(f"  Noise windows  : {min_count:,}  (label=0)")
    print(f"  Total windows  : {len(X):,}")
    print(f"{'='*55}")
    print(f"\nNext step: python src/data/split.py")


if __name__ == "__main__":
    main()