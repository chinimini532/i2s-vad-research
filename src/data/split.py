"""
src/data/split.py

What this file does (in simple words):
---------------------------------------
Loads X.npy and y.npy from data/processed/
Splits them into three sets:
  - train (70%) : model learns from this
  - val   (15%) : used during training to check progress
  - test  (15%) : used ONLY for final evaluation, never during training

Saves six files in data/splits/:
  X_train.npy, y_train.npy
  X_val.npy,   y_val.npy
  X_test.npy,  y_test.npy
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
SPLITS    = ROOT / "data" / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────
VAL_SIZE  = 0.15   # 15% for validation
TEST_SIZE = 0.15   # 15% for test
SEED      = 42


def main():
    # ── load ──────────────────────────────────────────────────────────────
    print("\nLoading processed data...")
    X = np.load(str(PROCESSED / "X.npy"))
    y = np.load(str(PROCESSED / "y.npy"))
    print(f"  Loaded X: {X.shape}  y: {y.shape}")
    print(f"  Speech windows (label=1): {y.sum():,}")
    print(f"  Noise windows  (label=0): {(y==0).sum():,}")

    # ── first split: separate test set ────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y        # keeps class balance in each split
    )

    # ── second split: separate val from remaining ─────────────────────────
    # val_size relative to the temp set so final ratio is correct
    val_size_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=SEED,
        stratify=y_temp
    )

    # ── save ──────────────────────────────────────────────────────────────
    np.save(str(SPLITS / "X_train.npy"), X_train)
    np.save(str(SPLITS / "y_train.npy"), y_train)
    np.save(str(SPLITS / "X_val.npy"),   X_val)
    np.save(str(SPLITS / "y_val.npy"),   y_val)
    np.save(str(SPLITS / "X_test.npy"),  X_test)
    np.save(str(SPLITS / "y_test.npy"),  y_test)

    # ── summary ───────────────────────────────────────────────────────────
    total = len(X)
    print(f"\n{'='*55}")
    print(f"  Split summary")
    print(f"{'='*55}")
    print(f"  Total   : {total:,} windows (100%)")
    print(f"  Train   : {len(X_train):,} windows "
          f"({len(X_train)/total*100:.1f}%)  -> X_train.npy, y_train.npy")
    print(f"  Val     : {len(X_val):,} windows "
          f"({len(X_val)/total*100:.1f}%)   -> X_val.npy,   y_val.npy")
    print(f"  Test    : {len(X_test):,} windows "
          f"({len(X_test)/total*100:.1f}%)   -> X_test.npy,  y_test.npy")
    print(f"{'='*55}")
    print(f"\n  Train class balance: "
          f"{y_train.sum()} speech / {(y_train==0).sum()} noise")
    print(f"  Val   class balance: "
          f"{y_val.sum()} speech / {(y_val==0).sum()} noise")
    print(f"  Test  class balance: "
          f"{y_test.sum()} speech / {(y_test==0).sum()} noise")
    print(f"\n  Files saved to: {SPLITS}")
    print(f"\nNext step: python src/training/train.py")


if __name__ == "__main__":
    main()