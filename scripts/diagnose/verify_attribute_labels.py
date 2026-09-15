"""
GATE for the attribute-inference experiment: verify label availability + balance
BEFORE training anything. Run this on the real data and read the report; it
decides which attributes are worth running per corpus.

What it prints, per corpus:
  * how many subjects/records carry sex and age in the header,
  * the sex balance (and the resulting majority-class chance floor),
  * the age distribution and the proposed 2-band (median) split balance,
  * a recommendation (sex always; age only where the bands are populated).

Usage:
    python scripts/diagnose/verify_attribute_labels.py
    python scripts/diagnose/verify_attribute_labels.py --no-remote   # local only
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.data.attributes import (
    age_band_edges,
    age_to_band,
    collect_ecgid_demographics,
    collect_mitbih_demographics,
)


def _report(name: str, rows: list, subject_key: str) -> None:
    print("=" * 64)
    print(f"{name}: {len(rows)} records loaded")
    print("=" * 64)

    # Deduplicate to subject level for the honest balance.
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r[subject_key], r)
    subj_rows = list(by_subject.values())
    n_subj = len(subj_rows)
    print(f"Distinct subjects: {n_subj}")

    # --- Sex ---
    sexes = [r["sex"] for r in subj_rows if r.get("sex") in ("M", "F")]
    n_sex = len(sexes)
    c = Counter(sexes)
    print(f"\nSEX: present for {n_sex}/{n_subj} subjects")
    if n_sex:
        floor = max(c.values()) / n_sex
        print(f"  counts: M={c.get('M',0)}  F={c.get('F',0)}  "
              f"-> majority-class floor = {floor:.3f}")
        verdict = "OK (binary, usable)" if min(c.values()) >= max(3, 0.15 * n_sex) \
            else "THIN (usable but small minority class)"
        print(f"  verdict: {verdict}")

    # --- Age ---
    ages = [r["age"] for r in subj_rows if r.get("age") is not None]
    n_age = len(ages)
    print(f"\nAGE: present for {n_age}/{n_subj} subjects")
    if n_age:
        a = np.asarray(ages)
        print(f"  range {a.min()}-{a.max()}, mean {a.mean():.1f}, median {np.median(a):.0f}")
        edges = age_band_edges(ages, 2)
        bands = Counter(age_to_band(x, edges) for x in ages)
        print(f"  2-band median split at {edges[0]:.0f}: "
              f"band0(<{edges[0]:.0f})={bands.get(0,0)}  "
              f"band1(>={edges[0]:.0f})={bands.get(1,0)}")
        if n_age >= 60 and min(bands.values()) >= 0.25 * n_age:
            print("  verdict: OK for 2 age bands")
        else:
            print("  verdict: TOO THIN / skewed -> recommend DROPPING age here")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-remote", action="store_true",
                   help="Do not download; use only locally present data.")
    args = p.parse_args()
    allow_remote = not args.no_remote

    print("\nVerifying attribute labels (sex/age) before building Experiment 1.\n")

    try:
        mit = collect_mitbih_demographics(allow_remote=allow_remote)
        _report("MIT-BIH", mit, subject_key="subject")
    except Exception as e:  # noqa: BLE001
        print(f"MIT-BIH: could not load ({type(e).__name__}: {e})")

    print()
    try:
        ecg = collect_ecgid_demographics(allow_remote=allow_remote)
        _report("ECG-ID", ecg, subject_key="subject")
    except Exception as e:  # noqa: BLE001
        print(f"ECG-ID: could not load ({type(e).__name__}: {e})")

    print("\n" + "=" * 64)
    print("Recommendation encoded in src/data/attributes.py decisions:")
    print("  MIT-BIH -> SEX only (age dropped as too thin/skewed).")
    print("  ECG-ID  -> SEX + AGE (2 bands at the cohort median).")
    print("Confirm the balances above match before trusting the leakage curves.")
    print("=" * 64)


if __name__ == "__main__":
    main()
