"""
Shared loader utilities for the analysis notebooks.

These read the multi-seed result layout produced by:
  scripts/train_baseline.py
  scripts/train_with_dp.py
  scripts/train_reid.py

Result layout (multi-seed):
  results/baseline/seed_<seed>/metrics.pkl
  results/dp/<mech>/eps_<eps>_delta_<delta>/seed_<seed>/metrics.pkl
  results/reid/baseline/seed_<seed>.pkl
  results/reid/dp/<mech>/eps_<eps>_delta_<delta>/<threat_model>/seed_<seed>/metrics.pkl
  results/reid_ecgid/...  (same as reid/)

All loaders return pandas DataFrames where each row is one (mech, eps, delta,
seed) combination. The notebooks then aggregate across seeds for plotting.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


_EPS_DELTA_RE = re.compile(r"eps_([0-9.eE+-]+)_delta_([0-9.eE+-]+)")
_SEED_DIR_RE = re.compile(r"seed_(\d+)")


# ---------------------------------------------------------------------------
# Baseline loaders
# ---------------------------------------------------------------------------
def load_utility_baseline(results_dir: Path) -> pd.DataFrame:
    """
    Load all utility baseline runs (one per seed).

    Returns DataFrame columns: seed, macro_f1, f1_N, f1_S, f1_V.
    """
    base = Path(results_dir) / "baseline"
    rows = []
    if not base.exists():
        return pd.DataFrame()

    # Two possible layouts:
    #   results/baseline/seed_<seed>/metrics.pkl  (used by run_all script)
    #   results/baseline/metrics.pkl              (single-seed legacy)
    for seed_dir in sorted(base.iterdir()):
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            mp = seed_dir / "metrics.pkl"
            if not mp.exists():
                continue
            seed = int(seed_dir.name.split("_")[1])
            with open(mp, "rb") as f:
                r = pickle.load(f)
            rows.append({
                "seed": seed,
                "macro_f1": r["macro_f1"],
                "f1_N": r["per_class_f1"][0],
                "f1_S": r["per_class_f1"][1],
                "f1_V": r["per_class_f1"][2],
            })
    # Legacy fallback: single file at results/baseline/metrics.pkl
    legacy = base / "metrics.pkl"
    if not rows and legacy.exists():
        with open(legacy, "rb") as f:
            r = pickle.load(f)
        rows.append({
            "seed": r.get("args", {}).get("seed", 0),
            "macro_f1": r["macro_f1"],
            "f1_N": r["per_class_f1"][0],
            "f1_S": r["per_class_f1"][1],
            "f1_V": r["per_class_f1"][2],
        })
    return pd.DataFrame(rows)


def load_reid_baseline(results_dir: Path) -> pd.DataFrame:
    """
    Load Re-ID baseline runs (one per seed).
    Note: Re-ID baseline writes flat .pkl files, not subdirectories.
    """
    base = Path(results_dir) / "baseline"
    if not base.exists():
        return pd.DataFrame()

    rows = []
    # Layout: results/reid/baseline/seed_<seed>.pkl
    for f in sorted(base.glob("seed_*.pkl")):
        seed_match = _SEED_DIR_RE.match(f.stem)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        rows.append({
            "seed": seed,
            "n_subjects": r["n_patients"],
            "reid_accuracy": r["overall_accuracy"],
            "random_baseline": r["random_baseline"],
        })
    # Legacy fallback: results/reid/baseline/metrics.pkl
    legacy = base / "metrics.pkl"
    if not rows and legacy.exists():
        with open(legacy, "rb") as fh:
            r = pickle.load(fh)
        rows.append({
            "seed": 0,
            "n_subjects": r["n_patients"],
            "reid_accuracy": r["overall_accuracy"],
            "random_baseline": r["random_baseline"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DP sweep loaders
# ---------------------------------------------------------------------------
def load_utility_sweep(results_dir: Path) -> pd.DataFrame:
    """
    Load all utility DP-sweep results across (mechanism, eps, delta, seed).

    Returns DataFrame columns:
      mechanism, eps, delta, seed, macro_f1, f1_N, f1_S, f1_V.
    """
    root = Path(results_dir) / "dp"
    if not root.exists():
        return pd.DataFrame()

    rows = []
    for mech_dir in sorted(root.iterdir()):
        if not mech_dir.is_dir() or mech_dir.name == "parallel_logs":
            continue
        mech = mech_dir.name
        for run_dir in sorted(mech_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = _EPS_DELTA_RE.match(run_dir.name)
            if not m:
                continue
            eps = float(m.group(1))
            delta = float(m.group(2))

            # Look for seed_<S>/metrics.pkl (new layout)
            for seed_dir in sorted(run_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                mp = seed_dir / "metrics.pkl"
                if not mp.exists():
                    continue
                seed = int(seed_dir.name.split("_")[1])
                try:
                    with open(mp, "rb") as f:
                        r = pickle.load(f)
                except Exception:
                    continue
                rows.append({
                    "mechanism": mech,
                    "eps": eps,
                    "delta": delta,
                    "seed": seed,
                    "macro_f1": r["macro_f1"],
                    "f1_N": r["per_class_f1"][0],
                    "f1_S": r["per_class_f1"][1],
                    "f1_V": r["per_class_f1"][2],
                })
            # Legacy: results/dp/<mech>/eps_X_delta_Y/metrics.pkl (no seed dir)
            legacy = run_dir / "metrics.pkl"
            if legacy.exists():
                try:
                    with open(legacy, "rb") as f:
                        r = pickle.load(f)
                    rows.append({
                        "mechanism": mech,
                        "eps": eps,
                        "delta": delta,
                        "seed": r.get("seed", 0),
                        "macro_f1": r["macro_f1"],
                        "f1_N": r["per_class_f1"][0],
                        "f1_S": r["per_class_f1"][1],
                        "f1_V": r["per_class_f1"][2],
                    })
                except Exception:
                    pass
    return pd.DataFrame(rows).sort_values(
        ["mechanism", "eps", "delta", "seed"]
    ).reset_index(drop=True)


def load_reid_sweep(results_dir: Path) -> pd.DataFrame:
    """
    Load Re-ID DP sweep across (mechanism, eps, delta, threat_model, seed).

    Returns DataFrame columns:
      mechanism, eps, delta, threat_model, seed, reid_accuracy.
    """
    root = Path(results_dir) / "dp"
    if not root.exists():
        return pd.DataFrame()

    rows = []
    for mech_dir in sorted(root.iterdir()):
        if not mech_dir.is_dir():
            continue
        mech = mech_dir.name
        for run_dir in sorted(mech_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = _EPS_DELTA_RE.match(run_dir.name)
            if not m:
                continue
            eps = float(m.group(1))
            delta = float(m.group(2))
            for tmode_dir in sorted(run_dir.iterdir()):
                if not tmode_dir.is_dir():
                    continue
                tmode = tmode_dir.name
                # seed_<S>/metrics.pkl
                for seed_dir in sorted(tmode_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    mp = seed_dir / "metrics.pkl"
                    if not mp.exists():
                        continue
                    seed = int(seed_dir.name.split("_")[1])
                    try:
                        with open(mp, "rb") as f:
                            r = pickle.load(f)
                    except Exception:
                        continue
                    rows.append({
                        "mechanism": mech,
                        "eps": eps,
                        "delta": delta,
                        "threat_model": tmode,
                        "seed": seed,
                        "reid_accuracy": r["overall_accuracy"],
                    })
                # Legacy
                legacy = tmode_dir / "metrics.pkl"
                if legacy.exists():
                    try:
                        with open(legacy, "rb") as f:
                            r = pickle.load(f)
                        rows.append({
                            "mechanism": mech,
                            "eps": eps,
                            "delta": delta,
                            "threat_model": tmode,
                            "seed": r.get("seed", 0),
                            "reid_accuracy": r["overall_accuracy"],
                        })
                    except Exception:
                        pass
    return pd.DataFrame(rows).sort_values(
        ["mechanism", "eps", "delta", "threat_model", "seed"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate_over_seeds(
    df: pd.DataFrame,
    group_cols: List[str],
    value_cols: List[str],
) -> pd.DataFrame:
    """
    Aggregate a multi-seed dataframe to (mean, std, min, max, count) per group.

    Returns a DataFrame indexed by group_cols, with hierarchical columns:
      (value_col, mean), (value_col, std), (value_col, min), (value_col, max),
      (value_col, n).
    """
    agg = df.groupby(group_cols)[value_cols].agg(
        ["mean", "std", "min", "max", "count"]
    )
    return agg


def flatten_aggregated(agg: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Convenience: pull mean / std / min / max for a single value column into a
    flat DataFrame with renamed columns (e.g., 'macro_f1_mean').
    """
    out = agg[value_col].copy()
    out.columns = [f"{value_col}_{stat}" for stat in out.columns]
    return out.reset_index()
