"""
Diagnose 4: Per-patient beat counts after qsPeaks extraction
            vs. Mousavi's interpatient_processing.txt (DS1 + DS2).

The reference numbers come from Mousavi's MATLAB run output, which is the
ground truth of his pre-prepared .mat file. If we match these counts after
qsPeaks filtering, we know our pipeline matches his.

Run from project root:
    python scripts/diagnose_qspeaks_counts.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from configs import SEQ2SEQ, DATA
from src.data import (
    load_mitbih_record,
    preprocess_signal,
    EXCLUDED_RECORDS,
    extract_beats_qspeaks,
)


# Reference per-patient counts from Mousavi's MATLAB run (interpatient_processing.txt)
# Format: {record_id: {'N': n, 'S': s, 'V': v, 'F': f, 'Q': q}}
MOUSAVI_DS1_COUNTS = {
    '101': {'N': 1858, 'S': 3,   'V': 0,   'F': 0,   'Q': 1},
    '106': {'N': 1506, 'S': 0,   'V': 520, 'F': 0,   'Q': 0},
    '108': {'N': 1735, 'S': 4,   'V': 17,  'F': 2,   'Q': 0},
    '109': {'N': 2491, 'S': 0,   'V': 38,  'F': 2,   'Q': 0},
    '112': {'N': 2535, 'S': 2,   'V': 0,   'F': 0,   'Q': 0},
    '114': {'N': 1819, 'S': 12,  'V': 43,  'F': 4,   'Q': 0},
    '115': {'N': 1950, 'S': 0,   'V': 0,   'F': 0,   'Q': 0},
    '116': {'N': 2301, 'S': 1,   'V': 109, 'F': 0,   'Q': 0},
    '118': {'N': 2164, 'S': 95,  'V': 16,  'F': 0,   'Q': 0},
    '119': {'N': 1542, 'S': 0,   'V': 444, 'F': 0,   'Q': 0},
    '122': {'N': 2475, 'S': 0,   'V': 0,   'F': 0,   'Q': 0},
    '124': {'N': 1535, 'S': 31,  'V': 47,  'F': 5,   'Q': 0},
    '201': {'N': 1634, 'S': 128, 'V': 198, 'F': 2,   'Q': 0},
    '203': {'N': 2510, 'S': 2,   'V': 441, 'F': 1,   'Q': 4},
    '205': {'N': 2566, 'S': 3,   'V': 70,  'F': 11,  'Q': 0},
    '207': {'N': 1538, 'S': 106, 'V': 210, 'F': 0,   'Q': 0},
    '208': {'N': 1583, 'S': 2,   'V': 992, 'F': 372, 'Q': 2},
    '209': {'N': 2617, 'S': 382, 'V': 1,   'F': 0,   'Q': 0},
    '215': {'N': 3192, 'S': 3,   'V': 163, 'F': 1,   'Q': 0},
    '220': {'N': 1951, 'S': 94,  'V': 0,   'F': 0,   'Q': 0},
    '223': {'N': 2044, 'S': 73,  'V': 472, 'F': 14,  'Q': 0},
    '230': {'N': 2252, 'S': 0,   'V': 1,   'F': 0,   'Q': 0},
}

MOUSAVI_DS2_COUNTS = {
    '100': {'N': 2237, 'S': 33,   'V': 1,   'F': 0,   'Q': 0},
    '103': {'N': 2081, 'S': 2,    'V': 0,   'F': 0,   'Q': 0},
    '105': {'N': 2515, 'S': 0,    'V': 41,  'F': 0,   'Q': 5},
    '111': {'N': 2122, 'S': 0,    'V': 1,   'F': 0,   'Q': 0},
    '113': {'N': 1788, 'S': 6,    'V': 0,   'F': 0,   'Q': 0},
    '117': {'N': 1533, 'S': 1,    'V': 0,   'F': 0,   'Q': 0},
    '121': {'N': 1858, 'S': 1,    'V': 1,   'F': 0,   'Q': 0},
    '123': {'N': 1513, 'S': 0,    'V': 3,   'F': 0,   'Q': 0},
    '200': {'N': 1740, 'S': 30,   'V': 826, 'F': 2,   'Q': 0},
    '202': {'N': 2060, 'S': 55,   'V': 19,  'F': 1,   'Q': 0},
    '210': {'N': 2419, 'S': 22,   'V': 194, 'F': 10,  'Q': 0},
    '212': {'N': 2747, 'S': 0,    'V': 0,   'F': 0,   'Q': 0},
    '213': {'N': 2640, 'S': 28,   'V': 219, 'F': 362, 'Q': 0},
    '214': {'N': 2001, 'S': 0,    'V': 256, 'F': 1,   'Q': 2},
    '219': {'N': 2073, 'S': 7,    'V': 64,  'F': 1,   'Q': 0},
    '221': {'N': 2029, 'S': 0,    'V': 396, 'F': 0,   'Q': 0},
    '222': {'N': 2268, 'S': 209,  'V': 0,   'F': 0,   'Q': 0},
    '228': {'N': 1682, 'S': 3,    'V': 362, 'F': 0,   'Q': 0},
    '231': {'N': 1567, 'S': 1,    'V': 1,   'F': 0,   'Q': 0},
    '232': {'N': 398,  'S': 1381, 'V': 0,   'F': 0,   'Q': 0},
    '233': {'N': 2229, 'S': 7,    'V': 830, 'F': 11,  'Q': 0},
    '234': {'N': 2698, 'S': 50,   'V': 3,   'F': 0,   'Q': 0},
}


def main() -> None:
    print(f"Using extraction: {DATA.beat_extraction}")
    print()

    splits = [
        ("DS1", SEQ2SEQ.ds1_records, MOUSAVI_DS1_COUNTS),
        ("DS2", SEQ2SEQ.ds2_records, MOUSAVI_DS2_COUNTS),
    ]

    grand_total_ok = 0
    grand_total = 0

    for split_name, records, mousavi_ref in splits:
        print("=" * 75)
        print(f"{split_name}: comparing per-patient counts vs. Mousavi's MATLAB output")
        print("=" * 75)
        print(f"{'rec':>4} | {'cls':>3} | {'ours':>5} | {'Mousavi':>7} | {'diff':>5}")
        print("-" * 75)

        split_ours_total = Counter()
        split_mou_total = Counter()
        n_ok_records = 0

        for rec_id in records:
            if rec_id in EXCLUDED_RECORDS:
                continue

            try:
                rec = load_mitbih_record(rec_id, allow_remote=True)
            except Exception as e:
                print(f"  ! Failed to load {rec_id}: {e}")
                continue

            signal = preprocess_signal(rec.signal)  # whole-record znorm
            beats, labels = extract_beats_qspeaks(
                signal, rec.r_peaks, rec.aami_labels,
                beat_length=SEQ2SEQ.beat_length,
                fs=rec.fs,
                keep_classes=("N", "S", "V", "F", "Q"),
                max_first_beat_samples=int(round(rec.fs)),
            )
            ours = Counter(labels)
            ref = mousavi_ref.get(rec_id, {})

            all_match = True
            for cls in ("N", "S", "V", "F", "Q"):
                ours_n = ours.get(cls, 0)
                ref_n = ref.get(cls, 0)
                diff = ours_n - ref_n
                if abs(diff) > 2:
                    all_match = False
                if cls != "Q":  # Q is mostly irrelevant for our 3-class setup
                    split_ours_total[cls] += ours_n
                    split_mou_total[cls] += ref_n
                marker = "" if abs(diff) <= 2 else "  ⚠"
                if abs(diff) > 0 or cls in ("S", "V"):  # show non-zero diffs and minority classes
                    print(f"{rec_id:>4} | {cls:>3} | {ours_n:>5} | {ref_n:>7} | {diff:>+5}{marker}")
            if all_match:
                n_ok_records += 1

            grand_total += 1
            if all_match:
                grand_total_ok += 1

        print()
        print(f"{split_name} TOTALS (excl. Q):")
        for cls in ("N", "S", "V", "F"):
            o = split_ours_total[cls]
            m = split_mou_total[cls]
            print(f"  {cls}: ours={o}, Mousavi={m}, diff={o - m:+}")
        n_records = sum(1 for r in records if r not in EXCLUDED_RECORDS)
        print(f"Records where all classes within ±2 of Mousavi: {n_ok_records}/{n_records}")
        print()

    print("=" * 75)
    print(f"Grand total: {grand_total_ok}/{grand_total} records match Mousavi within ±2")
    print()
    if grand_total_ok == grand_total:
        print("✓ Pipeline perfectly matches Mousavi MATLAB output.")
    elif grand_total_ok > 0.8 * grand_total:
        print("≈ Mostly matches Mousavi; small differences likely from peak-detection nuances.")
    else:
        print("⚠ Significant differences; investigate per-record.")


if __name__ == "__main__":
    main()
