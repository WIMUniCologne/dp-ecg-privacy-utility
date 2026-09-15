"""
Recompute the jump sensitivity on ECG-ID — the corpus the re-identification and
attribute experiments also attack — instead of reusing the MIT-BIH value.

Why this exists
---------------
The pipeline calibrates Δ once, on MIT-BIH, as the 95th percentile of the
per-sample jump |x_(i+1) - x_i| of the z-normalised signal (see
`notebooks/00_sensitivity_calibration.ipynb` and
`scripts/diagnose/sensitivity_on_beats.py`), and then uses Δ = 0.29 *uniformly*
for every mechanism, ε, AND corpus — including ECG-ID. But ECG-ID differs from
MIT-BIH in two ways that change the jump statistic:

  * sampling rate 500 Hz vs 360 Hz (closer-spaced samples -> smaller jumps), and
  * per-recording z-normalisation (`_preprocess_ecgid`) vs MIT-BIH's whole-record
    z-norm.

So the MIT-BIH-calibrated 0.29 is an *assumption* on ECG-ID, not a measured
value. This script measures ECG-ID's own jump sensitivity on exactly the signal
the DP mechanism perturbs (the per-recording z-normalised ECG-ID signal, which
the re-ID/attribute windows are contiguous slices of), and prints how far 0.29
is from the 95th-percentile criterion there. If they differ materially, pass the
measured value via `--sensitivity` to `train_reid.py` / `train_attribute.py`
(and the ECG-ID denoise run) to calibrate ECG-ID on its own terms.

Run:
    python scripts/diagnose/sensitivity_on_ecgid.py
    python scripts/diagnose/sensitivity_on_ecgid.py --on-windows      # on the literal DP-input windows
    python scripts/diagnose/sensitivity_on_ecgid.py --compare-mitbih  # add MIT-BIH re-ID windows for apples-to-apples
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from configs import DP


def jump_percentiles(diffs: np.ndarray) -> dict:
    return {
        "p50": float(np.percentile(diffs, 50)),
        "p90": float(np.percentile(diffs, 90)),
        "p95": float(np.percentile(diffs, 95)),
        "p99": float(np.percentile(diffs, 99)),
        "p99_9": float(np.percentile(diffs, 99.9)),
        "max": float(np.max(diffs)),
    }


def report(name: str, diffs: np.ndarray, abs_vals: np.ndarray, fs: float,
           n_units: int, unit: str) -> float:
    p = jump_percentiles(diffs)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"  fs = {fs:g} Hz | {n_units:,} {unit} | {len(diffs):,} |dx| values")
    print("  jump sensitivity (z-normalised |x_(i+1) - x_i|):")
    print(f"    50th pct:   {p['p50']:.4f}")
    print(f"    90th pct:   {p['p90']:.4f}")
    print(f"    95th pct:   {p['p95']:.4f}   <- sensitivity criterion")
    print(f"    99th pct:   {p['p99']:.4f}")
    print(f"    99.9th pct: {p['p99_9']:.4f}")
    print(f"    max:        {p['max']:.4f}")
    print(f"  amplitude |x|: p95={np.percentile(abs_vals,95):.4f} "
          f"p99={np.percentile(abs_vals,99):.4f} max={np.max(abs_vals):.4f}")
    return p["p95"]


# ---------------------------------------------------------------------------
# ECG-ID: jump sensitivity on the per-recording z-normalised signal
# ---------------------------------------------------------------------------
def ecgid_record_level(allow_remote: bool) -> "tuple[np.ndarray, np.ndarray, float, int]":
    from src.data.ecgid import (
        _ensure_ecgid_downloaded, _list_ecgid_records, _load_record,
        _preprocess_ecgid,
    )
    if allow_remote:
        _ensure_ecgid_downloaded(None)
    records = _list_ecgid_records(None)
    if not records:
        raise RuntimeError("No ECG-ID records found. Use default (remote) or "
                           "download to data/ecg-id-database/ first.")
    dx, ax = [], []
    fs_val = 500.0
    subjects = set()
    for subject, rec_path in records:
        subjects.add(subject)
        sig, fs_val = _load_record(rec_path)        # filtered channel, like the re-ID loader
        z = _preprocess_ecgid(sig).astype(np.float64)
        z = z[np.isfinite(z)]
        if len(z) < 2:
            continue
        dx.append(np.abs(np.diff(z)))
        ax.append(np.abs(z))
    return (np.concatenate(dx), np.concatenate(ax), float(fs_val), len(subjects))


def jumps_within_windows(X: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """|dx| within each window row (the literal DP-input surface)."""
    X = np.asarray(X, dtype=np.float64)
    return np.abs(np.diff(X, axis=1)).reshape(-1), np.abs(X).reshape(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--no-remote", action="store_true",
                   help="Use only locally present data; do not download.")
    p.add_argument("--on-windows", action="store_true",
                   help="Also compute on the literal build_ecgid_split() DP-input "
                        "windows (50%% overlap), not just the full recordings.")
    p.add_argument("--compare-mitbih", action="store_true",
                   help="Also compute on MIT-BIH re-ID windows (build_reid_ds1) "
                        "for an apples-to-apples comparison on the SAME surface.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    allow_remote = not args.no_remote
    delta_pipeline = DP.sensitivity

    print("=" * 70)
    print("ECG-ID JUMP-SENSITIVITY CALIBRATION")
    print("=" * 70)
    print(f"Pipeline currently uses Δ = {delta_pipeline} (calibrated on MIT-BIH, "
          f"reused for ECG-ID).")

    results = {}

    # --- ECG-ID, record-level z-normalised signal (windows are slices of this) ---
    try:
        dx, ax, fs, n_subj = ecgid_record_level(allow_remote)
        results["ECG-ID (record-level z-norm)"] = report(
            "ECG-ID — full per-recording z-normalised signal",
            dx, ax, fs, n_subj, "subjects")
    except Exception as e:  # noqa: BLE001
        print(f"\nECG-ID record-level: could not load ({type(e).__name__}: {e})")

    # --- ECG-ID, literal DP-input windows (optional) ---
    if args.on_windows:
        try:
            from src.data import build_ecgid_split
            tr, te = build_ecgid_split(allow_remote=allow_remote)
            X = np.concatenate([tr.X, te.X], axis=0)
            dxw, axw = jumps_within_windows(X)
            results["ECG-ID (DP-input windows)"] = report(
                "ECG-ID — build_ecgid_split() windows (literal DP input, overlapped)",
                dxw, axw, float(tr.fs), len(X), "windows")
        except Exception as e:  # noqa: BLE001
            print(f"\nECG-ID windows: could not load ({type(e).__name__}: {e})")

    # --- MIT-BIH re-ID windows, same surface (optional, apples-to-apples) ---
    if args.compare_mitbih:
        try:
            from src.data import build_reid_ds1
            tr, te = build_reid_ds1(allow_remote=allow_remote)
            X = np.concatenate([tr.X, te.X], axis=0)
            dxm, axm = jumps_within_windows(X)
            results["MIT-BIH (re-ID windows)"] = report(
                "MIT-BIH — build_reid_ds1() re-ID windows (same surface, 360 Hz)",
                dxm, axm, float(tr.fs), len(X), "windows")
        except Exception as e:  # noqa: BLE001
            print(f"\nMIT-BIH re-ID windows: could not load ({type(e).__name__}: {e})")

    # --- Verdict ---
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if not results:
        print("No data could be loaded; nothing to compare.")
        return
    print(f"{'surface':<34} {'Δ95 (measured)':>15} {'vs 0.29':>12}")
    print("-" * 64)
    for name, p95 in results.items():
        ratio = p95 / delta_pipeline if delta_pipeline else float("nan")
        print(f"{name:<34} {p95:>15.4f} {ratio:>11.2f}x")
    ecg_key = next((k for k in results if k.startswith("ECG-ID (record")), None)
    if ecg_key is not None:
        ecg = results[ecg_key]
        print()
        if abs(ecg - delta_pipeline) / delta_pipeline < 0.15:
            print(f"ECG-ID's own 95th-pct jump sensitivity ({ecg:.3f}) is within ~15% "
                  f"of the pipeline's 0.29 -> reusing 0.29 on ECG-ID is defensible.")
        else:
            direction = "smaller" if ecg < delta_pipeline else "larger"
            print(f"ECG-ID's own 95th-pct jump sensitivity ({ecg:.3f}) is materially "
                  f"{direction} than the pipeline's 0.29.")
            print(f"To calibrate ECG-ID on its own terms, pass --sensitivity {ecg:.2f} to:")
            print("    scripts/train_reid.py --dataset ecgid ...")
            print("    scripts/train_attribute.py --dataset ecgid ...")
            print("    scripts/train_reid_denoise.py --dataset ecgid ...")
        print("\nNote: a larger Δ means more noise per ε (more privacy, less utility); "
              "a smaller Δ means less. Report whichever you use, per corpus.")


if __name__ == "__main__":
    main()
