"""
Data preparation for Re-Identification experiments on MIT-BIH.

Each MIT-BIH record is split temporally (e.g., first 70% train, last 30% test).
Sliding windows over the signal serve as input. The classification target is
the patient identity (1 of N records).

This is the "closed-set" re-ID setting: the attacker knows the database of
patients and just has to assign a window to one of them.

The DP perturbation is applied to the WINDOWS (not the beats), to stay
consistent with the threat model: an attacker observes the privatised signal
and tries to identify the patient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from configs import DATA, REID, SEQ2SEQ
from src.data import EXCLUDED_RECORDS, load_mitbih_record, preprocess_signal


@dataclass
class ReIDDataset:
    """Re-ID training data: signal windows + patient IDs."""
    X: np.ndarray              # (n_windows, window_samples)
    y: np.ndarray              # (n_windows,) integer patient IDs
    patient_ids: List[str]     # mapping from int id -> record string
    fs: float


def _window_signal(
    signal: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> np.ndarray:
    """Slide non-overlapping (or overlapping) windows over a 1D signal."""
    if len(signal) < window_samples:
        return np.empty((0, window_samples), dtype=np.float32)
    n_windows = 1 + (len(signal) - window_samples) // step_samples
    windows = np.empty((n_windows, window_samples), dtype=np.float32)
    for i in range(n_windows):
        s = i * step_samples
        windows[i] = signal[s:s + window_samples]
    return windows


def build_reid_split(
    record_ids: Sequence[str],
    fs: float = 360.0,
    window_sec: float = REID.window_sec,
    step_sec: float = REID.step_sec,
    train_frac: float = 0.7,
    allow_remote: bool = True,
) -> Tuple[ReIDDataset, ReIDDataset]:
    """
    Build train and test Re-ID datasets from a list of MIT-BIH records.

    Each record's signal is temporally split: first `train_frac` of samples
    go into the train set, the rest into the test set. Sliding windows are
    then extracted from each half.

    Returns:
        train_ds, test_ds: ReIDDataset instances. Patient labels are integer
        IDs corresponding to the index in `patient_ids`.
    """
    window_samples = int(round(fs * window_sec))
    step_samples = int(round(fs * step_sec))

    train_X: List[np.ndarray] = []
    train_y: List[int] = []
    test_X: List[np.ndarray] = []
    test_y: List[int] = []
    patient_ids: List[str] = []

    pid = 0
    for rec_id in record_ids:
        if rec_id in EXCLUDED_RECORDS:
            continue
        rec = load_mitbih_record(rec_id, allow_remote=allow_remote)
        sig = preprocess_signal(rec.signal)

        split_idx = int(len(sig) * train_frac)
        sig_train = sig[:split_idx]
        sig_test = sig[split_idx:]

        w_train = _window_signal(sig_train, window_samples, step_samples)
        w_test = _window_signal(sig_test, window_samples, step_samples)

        if len(w_train) == 0 or len(w_test) == 0:
            continue

        train_X.append(w_train)
        train_y.extend([pid] * len(w_train))
        test_X.append(w_test)
        test_y.extend([pid] * len(w_test))
        patient_ids.append(rec_id)
        pid += 1

    return (
        ReIDDataset(
            X=np.concatenate(train_X, axis=0),
            y=np.asarray(train_y, dtype=np.int32),
            patient_ids=patient_ids,
            fs=fs,
        ),
        ReIDDataset(
            X=np.concatenate(test_X, axis=0),
            y=np.asarray(test_y, dtype=np.int32),
            patient_ids=patient_ids,
            fs=fs,
        ),
    )


def build_reid_ds1(
    fs: float = 360.0,
    train_frac: float = 0.7,
    allow_remote: bool = True,
) -> Tuple[ReIDDataset, ReIDDataset]:
    """Build Re-ID train/test datasets using MIT-BIH DS1 (22 patients)."""
    return build_reid_split(
        SEQ2SEQ.ds1_records, fs=fs, train_frac=train_frac,
        allow_remote=allow_remote,
    )


def build_reid_all_mitbih(
    fs: float = 360.0,
    train_frac: float = 0.7,
    allow_remote: bool = True,
) -> Tuple[ReIDDataset, ReIDDataset]:
    """Build Re-ID train/test datasets using DS1 + DS2 (44 patients)."""
    records = tuple(SEQ2SEQ.ds1_records) + tuple(SEQ2SEQ.ds2_records)
    return build_reid_split(
        records, fs=fs, train_frac=train_frac, allow_remote=allow_remote,
    )
