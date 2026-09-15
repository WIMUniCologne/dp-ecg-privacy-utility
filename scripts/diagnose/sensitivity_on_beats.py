"""
Recompute jump sensitivity on the actual DP input — the z-normalised, 
qsPeaks-extracted, resampled beats — instead of the raw record-level signal.

This is the right surface to calibrate on, because the DP mechanism is
applied to these beat samples (not to the original recording).

Run:
    python scripts/diagnose/sensitivity_on_beats.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.models import read_mitbih_wfdb


def main() -> None:
    print("Loading MIT-BIH DS1 + DS2 beats (z-normalised, qsPeaks-extracted)...")
    X_train, _ = read_mitbih_wfdb(trainset=1, beat_extraction='qspeaks',
                                   seed=42, allow_remote=True)
    X_test, _ = read_mitbih_wfdb(trainset=0, beat_extraction='qspeaks',
                                  seed=42, allow_remote=True)

    print(f"  DS1 beats array shape: {X_train.shape}")
    print(f"  DS2 beats array shape: {X_test.shape}")

    # Concatenate all beats and compute per-beat sample-to-sample jumps
    # Both arrays are (n_seqs, beats_per_group, beat_length).
    all_seqs = np.concatenate([X_train, X_test], axis=0)
    n_seqs, n_beats, beat_len = all_seqs.shape
    print(f"  Combined: {n_seqs * n_beats:,} beats, {beat_len} samples each")

    # Reshape to (n_beats_total, beat_length)
    beats_flat = all_seqs.reshape(-1, beat_len)
    # |x[i+1] - x[i]| within each beat
    diffs = np.abs(np.diff(beats_flat, axis=1)).flatten()
    print(f"  Total |dx| values: {len(diffs):,}")
    print()

    # Pooled corpus statistics
    print("Jump sensitivity on extracted beats (z-normalised):")
    print(f"  50th pct:    {np.percentile(diffs, 50):.4f}")
    print(f"  90th pct:    {np.percentile(diffs, 90):.4f}")
    print(f"  95th pct:    {np.percentile(diffs, 95):.4f}")
    print(f"  99th pct:    {np.percentile(diffs, 99):.4f}")
    print(f"  99.9th pct:  {np.percentile(diffs, 99.9):.4f}")
    print(f"  max:         {np.max(diffs):.4f}")
    print()

    # Reference: amplitude range
    print("Amplitude range (z-normalised beats):")
    print(f"  95th pct |x|:   {np.percentile(np.abs(beats_flat), 95):.4f}")
    print(f"  99th pct |x|:   {np.percentile(np.abs(beats_flat), 99):.4f}")
    print(f"  max |x|:        {np.max(np.abs(beats_flat)):.4f}")


if __name__ == "__main__":
    main()
