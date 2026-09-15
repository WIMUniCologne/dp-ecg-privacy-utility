"""
ECG-ID Database loader (PhysioNet ecgiddb/1.0.0).

Used for Re-Identification robustness experiments. The database contains
310 ECG recordings from 90 subjects, 20 s each at 500 Hz, with 1-22 sessions
per subject collected over up to 6 months.

This is a complementary dataset to MIT-BIH for re-ID:
  * MIT-BIH: 22 (or 44) patients, but intra-recording temporal split
  * ECG-ID:  90 subjects, multi-session split (different days)

Of the 90 subjects, `build_ecgid_split` uses the 89 with at least two
recordings (`min_sessions=2`): a session-disjoint split needs one session to
train on and one to be evaluated against, and Person_74 has only a single
recording. The chance level quoted for this corpus is therefore 1/89.

The session-based split is the harder benchmark — the attacker must
re-identify a subject across recording sessions, not just across time
within one recording.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import re

import numpy as np
import wfdb

from configs import DATA, ECGID_DIR, REID

logger = logging.getLogger(__name__)


@dataclass
class ECGIDDataset:
    """Re-ID data from ECG-ID."""
    X: np.ndarray          # (n_windows, window_samples)
    y: np.ndarray          # (n_windows,) integer subject IDs
    subject_ids: List[str] # int id -> 'Person_01' etc.
    fs: float


def _list_ecgid_records(data_dir: Optional[Path] = None) -> List[Tuple[str, str]]:
    """
    List ECG-ID records as (subject_id, record_path) tuples.

    The ECG-ID layout is:
        ecg-id-database-1.0.0/Person_01/rec_1
        ecg-id-database-1.0.0/Person_01/rec_2
        ...

    If data_dir is None, returns the canonical empty list; the caller can
    then trigger a remote download.
    """
    if data_dir is None:
        data_dir = ECGID_DIR
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    records = []
    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("Person_"):
            continue
        subject = person_dir.name
        for f in sorted(person_dir.iterdir()):
            if f.suffix == ".hea":
                # rec_1.hea -> rec_1
                rec_name = f.stem
                records.append((subject, str(person_dir / rec_name)))
    return records


def _ensure_ecgid_downloaded(data_dir: Optional[Path] = None) -> Path:
    """Download ECG-ID via wfdb if not already present."""
    if data_dir is None:
        data_dir = ECGID_DIR
    data_dir = Path(data_dir)

    if list(_list_ecgid_records(data_dir)):
        return data_dir

    logger.info(f"Downloading ECG-ID Database to {data_dir} ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("ecgiddb/1.0.0", str(data_dir), keep_subdirs=True)
    logger.info("ECG-ID download complete.")
    return data_dir


def _load_record(record_path: str) -> Tuple[np.ndarray, float]:
    """Load one ECG-ID record. Returns (filtered_signal, fs)."""
    record = wfdb.rdrecord(record_path)
    # ECG-ID records have two channels: 'ECG I' (raw) and 'ECG I filtered'.
    sig_names = [s.lower() for s in record.sig_name]
    # Prefer the filtered channel; fall back to raw
    if "ecg i filtered" in sig_names:
        ch_idx = sig_names.index("ecg i filtered")
    elif "ecg i_filtered" in sig_names:
        ch_idx = sig_names.index("ecg i_filtered")
    else:
        # Any signal containing 'filt' in the name
        filt_candidates = [i for i, n in enumerate(sig_names) if "filt" in n]
        ch_idx = filt_candidates[0] if filt_candidates else 0
    signal = record.p_signal[:, ch_idx].astype(np.float32)
    return signal, float(record.fs)


def _preprocess_ecgid(signal: np.ndarray) -> np.ndarray:
    """Z-normalize the whole recording, consistent with MIT-BIH preprocessing."""
    s = signal.astype(np.float32)
    if DATA.znormalize:
        mean = float(s.mean())
        std = float(s.std())
        if std > 0:
            s = (s - mean) / std
    return s


def _window_signal(signal: np.ndarray, window_samples: int, step_samples: int) -> np.ndarray:
    if len(signal) < window_samples:
        return np.empty((0, window_samples), dtype=np.float32)
    n = 1 + (len(signal) - window_samples) // step_samples
    out = np.empty((n, window_samples), dtype=np.float32)
    for i in range(n):
        s = i * step_samples
        out[i] = signal[s:s + window_samples]
    return out


def _record_number(rec_path) -> tuple:
    """Trailing integer of a record name, for chronological ordering.

    ECG-ID records are named rec_1 ... rec_N. Falls back to the plain name
    when no trailing number is present, so ordering stays deterministic.
    """
    name = Path(rec_path).stem
    m = re.search(r"(\d+)$", name)
    return (0, int(m.group(1))) if m else (1, name)


def build_ecgid_split(
    fs: float = DATA.ecgid_fs,
    window_sec: float = REID.window_sec,
    step_sec: float = REID.step_sec,
    min_sessions: int = 2,
    allow_remote: bool = True,
    data_dir: Optional[Path] = None,
) -> Tuple[ECGIDDataset, ECGIDDataset]:
    """
    Build a session-level train/test split.

    For each subject with >= `min_sessions` sessions:
      - sort recordings chronologically (by filename)
      - first half of sessions → train, second half → test

    This ensures the attacker is evaluated on sessions it has never seen
    (much harder than intra-session temporal splits).

    Args:
        fs: expected sampling rate (verifies consistency).
        window_sec: sliding-window length.
        step_sec: sliding-window step.
        min_sessions: drop subjects with fewer recordings than this.
        allow_remote: download from PhysioNet if data not present locally.
        data_dir: override the default ECG-ID data directory.

    Returns:
        (train_ds, test_ds): ECGIDDataset instances with integer subject IDs.
    """
    if allow_remote:
        _ensure_ecgid_downloaded(data_dir)

    records = _list_ecgid_records(data_dir)
    if not records:
        raise RuntimeError(
            f"No ECG-ID records found in {data_dir or ECGID_DIR}. "
            f"Set allow_remote=True or download manually."
        )

    # Group records by subject
    by_subject: dict = {}
    for subject, rec_path in records:
        by_subject.setdefault(subject, []).append(rec_path)

    window_samples = int(round(fs * window_sec))
    step_samples = int(round(fs * step_sec))

    train_X: List[np.ndarray] = []
    train_y: List[int] = []
    test_X: List[np.ndarray] = []
    test_y: List[int] = []
    subject_ids: List[str] = []

    sid = 0
    for subject, recs in sorted(by_subject.items()):
        if len(recs) < min_sessions:
            continue
        # Sort by the trailing record number, not lexicographically: plain
        # sorted() gives rec_1, rec_10, rec_11, ..., rec_2, which is not the
        # recording order. Three subjects have ten or more sessions
        # (Person_01: 20, Person_02: 22, Person_52: 11) and together carry
        # 16 % of the window-weighted test set, so the difference matters.
        recs_sorted = sorted(recs, key=_record_number)
        # First half -> train, second half -> test
        n_train = max(1, len(recs_sorted) // 2)
        train_recs = recs_sorted[:n_train]
        test_recs = recs_sorted[n_train:]
        if not test_recs:  # need at least one test session
            continue

        sub_train_windows: List[np.ndarray] = []
        for rec_path in train_recs:
            sig, rec_fs = _load_record(rec_path)
            sig = _preprocess_ecgid(sig)
            w = _window_signal(sig, window_samples, step_samples)
            if len(w) > 0:
                sub_train_windows.append(w)

        sub_test_windows: List[np.ndarray] = []
        for rec_path in test_recs:
            sig, rec_fs = _load_record(rec_path)
            sig = _preprocess_ecgid(sig)
            w = _window_signal(sig, window_samples, step_samples)
            if len(w) > 0:
                sub_test_windows.append(w)

        if not sub_train_windows or not sub_test_windows:
            continue

        sub_train = np.concatenate(sub_train_windows, axis=0)
        sub_test = np.concatenate(sub_test_windows, axis=0)

        train_X.append(sub_train)
        train_y.extend([sid] * len(sub_train))
        test_X.append(sub_test)
        test_y.extend([sid] * len(sub_test))
        subject_ids.append(subject)
        sid += 1

    if not subject_ids:
        raise RuntimeError("No usable ECG-ID subjects found.")

    train_ds = ECGIDDataset(
        X=np.concatenate(train_X, axis=0),
        y=np.asarray(train_y, dtype=np.int32),
        subject_ids=subject_ids,
        fs=fs,
    )
    test_ds = ECGIDDataset(
        X=np.concatenate(test_X, axis=0),
        y=np.asarray(test_y, dtype=np.int32),
        subject_ids=subject_ids,
        fs=fs,
    )
    return train_ds, test_ds
