"""
MIT-BIH Arrhythmia Database loader.

Two main entry points:
- load_mitbih_signal(record): raw ECG + R-peaks + AAMI labels per beat
- load_mitbih_for_seq2seq(): full inter-patient DS1/DS2 split, grouped beats
- load_mitbih_for_reid(): all 47 subjects, windowed (no R-peak), for re-ID

Data download:
    Place MIT-BIH Arrhythmia files in `data/mit-bih-arrhythmia/`
    (Either pre-downloaded, or wfdb will fetch them remotely if allow_remote=True.)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wfdb
from scipy.signal import detrend as scipy_detrend

from configs import MITBIH_DIR, DATA, SEQ2SEQ


# -----------------------------------------------------------------------------
# AAMI heartbeat class mapping (per Chazal et al. 2004 / AAMI EC57)
# -----------------------------------------------------------------------------
# Annotation symbols → AAMI superclass
AAMI_MAP: Dict[str, str] = {
    # N: normal beats
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S: supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V: ventricular ectopic
    "V": "V", "E": "V",
    # F: fusion (often merged or excluded)
    "F": "F",
    # Q: unknown / paced
    "/": "Q", "f": "Q", "Q": "Q",
}


# Records to fully exclude (paced beats, per AAMI convention)
EXCLUDED_RECORDS = ("102", "104", "107", "217")


# All 48 records in MIT-BIH (47 subjects, 48 recordings)
ALL_RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108",
    "109", "111", "112", "113", "114", "115", "116", "117", "118",
    "119", "121", "122", "123", "124", "200", "201", "202", "203",
    "205", "207", "208", "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223", "228", "230", "231",
    "232", "233", "234",
)


@dataclass
class MITBIHRecord:
    """A loaded MIT-BIH record."""
    record_id: str
    signal: np.ndarray         # shape (n_samples,), float32, single channel
    fs: float
    r_peaks: np.ndarray        # sample indices of beats
    aami_labels: List[str]     # AAMI class per R-peak, same length as r_peaks


# -----------------------------------------------------------------------------
# Low-level loading
# -----------------------------------------------------------------------------
def _pick_channel(sig_names: List[str], preferred: str) -> int:
    """Return channel index for preferred lead, fallback to 0."""
    names_upper = [s.upper() for s in sig_names]
    if preferred.upper() in names_upper:
        return names_upper.index(preferred.upper())
    return 0


def load_mitbih_record(
    record_id: str,
    local_dir: Optional[Path] = None,
    allow_remote: bool = True,
) -> MITBIHRecord:
    """
    Load a single MIT-BIH record (signal + beat annotations).

    Tries local directory first, falls back to PhysioNet if allow_remote.
    """
    local_dir = local_dir or MITBIH_DIR
    local_path = local_dir / record_id

    # Try local
    if (local_dir / f"{record_id}.hea").exists():
        rec = wfdb.rdrecord(str(local_path))
        ann = wfdb.rdann(str(local_path), "atr")
    elif allow_remote:
        rec = wfdb.rdrecord(record_id, pn_dir="mitdb")
        ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb")
    else:
        raise FileNotFoundError(
            f"Record {record_id} not found in {local_dir} and allow_remote=False"
        )

    # Pick channel
    ch_idx = _pick_channel(rec.sig_name, DATA.mitbih_channel)
    signal = rec.p_signal[:, ch_idx].astype(np.float32)
    fs = float(rec.fs)

    # Extract beat annotations
    r_peaks = np.asarray(ann.sample)
    symbols = ann.symbol
    aami_labels = [AAMI_MAP.get(s, "Q") for s in symbols]

    return MITBIHRecord(
        record_id=record_id,
        signal=signal,
        fs=fs,
        r_peaks=r_peaks,
        aami_labels=aami_labels,
    )


# -----------------------------------------------------------------------------
# Preprocessing (signal-level)
# -----------------------------------------------------------------------------
def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """Detrend + z-normalize. Same for all downstream pipelines."""
    s = signal.astype(np.float32)
    if DATA.detrend:
        s = scipy_detrend(s).astype(np.float32)
    if DATA.znormalize:
        mean = s.mean()
        std = s.std()
        if std > 0:
            s = (s - mean) / std
    return s
