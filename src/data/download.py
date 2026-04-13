"""
src/data/download.py
Downloads LibriSpeech and MUSAN datasets.
--mode test  → small subset for LG Gram (quick, ~1GB)
--mode full  → complete dataset for MSI training
"""

import os
import argparse
import requests
import tarfile
from tqdm import tqdm
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # project root
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ─── Dataset URLs ─────────────────────────────────────────────────────────────
LIBRISPEECH_URLS = {
    # test mode  → ~200MB, ~1h of clean speech
    "test": [
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    ],
    # full mode → ~55GB total speech
    "full": [
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    ],
}

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"  # ~11GB


# ─── Helpers ──────────────────────────────────────────────────────────────────
def download_file(url: str, dest_path: Path) -> None:
    """Download a file with a tqdm progress bar. Skips if already exists."""
    if dest_path.exists():
        print(f"  [skip] already downloaded: {dest_path.name}")
        return

    print(f"  [download] {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest_path.name,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            bar.update(len(chunk))


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    """Extract a .tar.gz file. Skips if target folder already exists."""
    # derive expected folder name from tar filename
    folder_name = tar_path.name.replace(".tar.gz", "")
    target = extract_to / folder_name

    if target.exists():
        print(f"  [skip] already extracted: {folder_name}")
        return

    print(f"  [extract] {tar_path.name} → {extract_to}")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {tar_path.name}"):
            tar.extract(member, extract_to)
    print(f"  [done] extracted to {extract_to}")


def download_librispeech(mode: str) -> None:
    print("\n=== LibriSpeech ===")
    urls = LIBRISPEECH_URLS[mode]
    dest_dir = RAW / "librispeech"
    dest_dir.mkdir(exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        tar_path = dest_dir / filename
        download_file(url, tar_path)
        extract_tar(tar_path, dest_dir)

    print("[LibriSpeech] download complete.")


def download_musan(mode: str) -> None:
    print("\n=== MUSAN (noise dataset) ===")

    if mode == "test":
        print("  [test mode] skipping full MUSAN download.")
        print("  We will generate synthetic silence instead for LG Gram testing.")
        _create_synthetic_noise()
        return

    dest_dir = RAW / "musan"
    dest_dir.mkdir(exist_ok=True)
    tar_path = dest_dir / "musan.tar.gz"
    download_file(MUSAN_URL, tar_path)
    extract_tar(tar_path, dest_dir)
    print("[MUSAN] download complete.")


def _create_synthetic_noise() -> None:
    """
    For LG Gram test mode: generate simple synthetic noise/silence WAV files
    instead of downloading 11GB MUSAN. This lets us verify the full pipeline
    works before committing to the full download on MSI.
    """
    import numpy as np
    import soundfile as sf

    noise_dir = RAW / "musan_synthetic"
    noise_dir.mkdir(exist_ok=True)

    sample_rate = 16000  # will be downsampled to 8kHz in preprocess.py
    duration_s  = 10     # 10 seconds per file
    n_files     = 20     # 20 files × 10s = 200s of noise data

    print(f"  Generating {n_files} synthetic noise files in {noise_dir} ...")
    for i in tqdm(range(n_files), desc="Generating noise"):
        # mix: white noise + occasional silence
        noise_type = i % 3
        if noise_type == 0:
            # white noise
            samples = np.random.normal(0, 0.05, sample_rate * duration_s)
        elif noise_type == 1:
            # silence (near zero)
            samples = np.random.normal(0, 0.001, sample_rate * duration_s)
        else:
            # pink-ish noise (more realistic background)
            white = np.random.normal(0, 0.05, sample_rate * duration_s)
            samples = np.cumsum(white) * 0.01
            samples = np.clip(samples, -1.0, 1.0)

        samples = samples.astype(np.float32)
        out_path = noise_dir / f"noise_{i:04d}.wav"
        sf.write(out_path, samples, sample_rate)

    print(f"  [done] synthetic noise files saved to {noise_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download LibriSpeech + MUSAN datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="test = small subset for LG Gram | full = complete dataset for MSI",
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Download mode: {args.mode.upper()}")
    print(f"  Saving to:     {RAW}")
    print(f"{'='*50}")

    if args.mode == "test":
        print("\n[test mode] Will download train-clean-100 only (~6.3GB)")
        print("            MUSAN replaced with synthetic noise files.")
        print("            This is enough to verify the full pipeline.\n")
    else:
        print("\n[full mode] Will download all LibriSpeech splits + MUSAN (~66GB)")
        print("            Make sure you have enough disk space.\n")

    download_librispeech(args.mode)
    download_musan(args.mode)

    print("\n=== All downloads complete ===")
    print(f"Data saved to: {RAW}")
    print("Next step: run  python src/data/preprocess.py")


if __name__ == "__main__":
    main()