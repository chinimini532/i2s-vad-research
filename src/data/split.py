"""
src/data/split.py

Reads experiment config from src/training/config.py
Loads processed data from data/processed/{EXPERIMENT}/
Saves splits to data/splits/{EXPERIMENT}/
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.config import (
    EXPERIMENT, PROCESSED, SPLITS, SEED, print_config
)

VAL_SIZE  = 0.15
TEST_SIZE = 0.15


def main():
    print_config()

    print("\nLoading processed data...")
    X = np.load(str(PROCESSED / "X.npy"))
    y = np.load(str(PROCESSED / "y.npy"))
    print(f"  X: {X.shape}  y: {y.shape}")
    print(f"  Speech: {y.sum():,}  Noise: {(y==0).sum():,}")

    # first split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=SEED, stratify=y
    )

    # split remainder into train/val
    val_adj = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adj,
        random_state=SEED, stratify=y_temp
    )

    # save
    np.save(str(SPLITS / "X_train.npy"), X_train)
    np.save(str(SPLITS / "y_train.npy"), y_train)
    np.save(str(SPLITS / "X_val.npy"),   X_val)
    np.save(str(SPLITS / "y_val.npy"),   y_val)
    np.save(str(SPLITS / "X_test.npy"),  X_test)
    np.save(str(SPLITS / "y_test.npy"),  y_test)

    total = len(X)
    print(f"\n{'='*55}")
    print(f"  Experiment : {EXPERIMENT}")
    print(f"  Train : {len(X_train):,}  ({len(X_train)/total*100:.1f}%)")
    print(f"  Val   : {len(X_val):,}   ({len(X_val)/total*100:.1f}%)")
    print(f"  Test  : {len(X_test):,}   ({len(X_test)/total*100:.1f}%)")
    print(f"  Saved to: {SPLITS}")
    print(f"{'='*55}")
    print(f"\nNext: python src/training/train.py")


if __name__ == "__main__":
    main()