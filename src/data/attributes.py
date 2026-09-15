"""
Demographic attribute labels (sex, age) for the attribute-inference experiment.

Second leakage axis: a record counted as re-identification-protected may still
leak demographic attributes. This module provides the labels and, crucially,
**subject-disjoint** windowed datasets for measuring that leakage.

Why subject-disjoint is mandatory
---------------------------------
A demographic attribute is constant across all of one subject's beats/windows.
If any subject appears in both train and test, the classifier can memorise
identity and read the attribute off the identity -- i.e. it becomes disguised
re-identification, and the leakage number is meaningless. So the split here is
on the SUBJECT level (no subject in both halves). This is a *different* split
from the within-record temporal Re-ID split in ``src/data/reid.py`` and must not
be reused.

Label sources (verified against real PhysioNet headers)
-------------------------------------------------------
MIT-BIH: first header comment line is ``<age> <sex> ...`` e.g. ``69 M``.
ECG-ID:  per-record header comments ``Age: 25`` / ``Sex: male``.

Decisions (see scripts/diagnose/verify_attribute_labels.py for the balance):
  * MIT-BIH: SEX only (47 subjects, age skewed/old -> too thin for age bands).
  * ECG-ID:  SEX, plus AGE as 2 bands at the cohort median (90 subjects).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from configs import DATA, ECGID_DIR, MITBIH_DIR, REID
from src.data.mitbih import ALL_RECORDS, preprocess_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------
def parse_demographics(comments: Sequence[str], source: str) -> Dict[str, Optional[object]]:
    """
    Parse {'age': int|None, 'sex': 'M'|'F'|None} from a record's header comments.

    Robust to the two real formats:
      MIT-BIH: ['69 M 1085 1629 x1', 'Aldomet, Inderal']
      ECG-ID : ['Age: 25', 'Sex: male', 'ECG date: ...']
    """
    age: Optional[int] = None
    sex: Optional[str] = None

    def norm_sex(tok: str) -> Optional[str]:
        t = tok.strip().lower()
        if t in ("m", "male", "man"):
            return "M"
        if t in ("f", "female", "woman"):
            return "F"
        return None

    if source == "mitbih":
        # First non-empty comment: "<age> <sex> ...".
        for c in comments:
            toks = c.strip().split()
            if len(toks) >= 2 and toks[0].lstrip("-").isdigit():
                a = int(toks[0])
                age = a if 0 < a < 120 else None
                sex = norm_sex(toks[1])
                break
    else:  # ecgid (and any "Key: value" style)
        for c in comments:
            if ":" not in c:
                continue
            key, _, val = c.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key.startswith("age"):
                digits = "".join(ch for ch in val if ch.isdigit())
                if digits:
                    a = int(digits)
                    age = a if 0 < a < 120 else None
            elif key.startswith("sex") or key.startswith("gender"):
                sex = norm_sex(val)
    return {"age": age, "sex": sex}


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
@dataclass
class AttributeDataset:
    """Windows + attribute labels, with subject id per window for honest eval."""
    X: np.ndarray              # (n_windows, window_samples)
    y: np.ndarray              # (n_windows,) integer attribute-class ids
    subject_per_window: np.ndarray  # (n_windows,) integer subject ids
    class_names: List[str]     # int id -> class label string
    subject_ids: List[str]     # int subject id -> record/person string
    fs: float
    attribute: str             # 'sex' or 'age'


def _window_signal(signal: np.ndarray, window_samples: int, step_samples: int) -> np.ndarray:
    if len(signal) < window_samples:
        return np.empty((0, window_samples), dtype=np.float32)
    n = 1 + (len(signal) - window_samples) // step_samples
    out = np.empty((n, window_samples), dtype=np.float32)
    for i in range(n):
        s = i * step_samples
        out[i] = signal[s:s + window_samples]
    return out


def _subject_split(
    subject_keys: Sequence[str], train_frac: float, split_seed: int
) -> Tuple[set, set]:
    """Deterministic subject-level split. Fixed across model seeds on purpose."""
    uniq = sorted(set(subject_keys))
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(uniq))
    n_train = max(1, int(round(len(uniq) * train_frac)))
    n_train = min(n_train, len(uniq) - 1)  # guarantee a non-empty test set
    train_subj = {uniq[i] for i in perm[:n_train]}
    test_subj = {uniq[i] for i in perm[n_train:]}
    return train_subj, test_subj


# ---------------------------------------------------------------------------
# MIT-BIH loaders
# ---------------------------------------------------------------------------
def _load_mitbih_signal_and_demo(
    record_id: str, allow_remote: bool
) -> Tuple[np.ndarray, float, Dict]:
    import wfdb
    local = MITBIH_DIR / record_id
    if (MITBIH_DIR / f"{record_id}.hea").exists():
        rec = wfdb.rdrecord(str(local))
    elif allow_remote:
        rec = wfdb.rdrecord(record_id, pn_dir="mitdb")
    else:
        raise FileNotFoundError(f"MIT-BIH {record_id} not local and allow_remote=False")
    ch = 0
    names = [s.upper() for s in rec.sig_name]
    if DATA.mitbih_channel.upper() in names:
        ch = names.index(DATA.mitbih_channel.upper())
    sig = rec.p_signal[:, ch].astype(np.float32)
    demo = parse_demographics(rec.comments, "mitbih")
    return sig, float(rec.fs), demo


def _mitbih_subject_key(record_id: str) -> str:
    """201 and 202 are the same subject; everything else is its own subject."""
    return "200s_subj" if record_id in ("201", "202") else record_id


def collect_mitbih_demographics(
    records: Sequence[str] = ALL_RECORDS, allow_remote: bool = True,
) -> List[Dict]:
    """Return per-record {record, subject, age, sex} for verification/reporting."""
    out = []
    for rid in records:
        try:
            _, _, demo = _load_mitbih_signal_and_demo(rid, allow_remote)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MIT-BIH {rid}: {e}")
            demo = {"age": None, "sex": None}
        out.append({"record": rid, "subject": _mitbih_subject_key(rid), **demo})
    return out


# ---------------------------------------------------------------------------
# ECG-ID loaders
# ---------------------------------------------------------------------------
def _list_ecgid_records(data_dir: Optional[Path] = None) -> List[Tuple[str, str]]:
    data_dir = Path(data_dir or ECGID_DIR)
    if not data_dir.exists():
        return []
    records = []
    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("Person_"):
            continue
        for f in sorted(person_dir.iterdir()):
            if f.suffix == ".hea":
                records.append((person_dir.name, str(person_dir / f.stem)))
    return records


def _load_ecgid_signal_and_demo(record_path: str) -> Tuple[np.ndarray, float, Dict]:
    import wfdb
    rec = wfdb.rdrecord(record_path)
    names = [s.lower() for s in rec.sig_name]
    if "ecg i filtered" in names:
        ch = names.index("ecg i filtered")
    else:
        filt = [i for i, n in enumerate(names) if "filt" in n]
        ch = filt[0] if filt else 0
    sig = rec.p_signal[:, ch].astype(np.float32)
    demo = parse_demographics(rec.comments, "ecgid")
    return sig, float(rec.fs), demo


def _preprocess_ecgid(signal: np.ndarray) -> np.ndarray:
    s = signal.astype(np.float32)
    if DATA.znormalize:
        mean, std = float(s.mean()), float(s.std())
        if std > 0:
            s = (s - mean) / std
    return s


def collect_ecgid_demographics(
    allow_remote: bool = True, data_dir: Optional[Path] = None,
) -> List[Dict]:
    """Return per-subject {subject, age, sex} for verification/reporting."""
    if allow_remote:
        from src.data.ecgid import _ensure_ecgid_downloaded
        _ensure_ecgid_downloaded(data_dir)
    records = _list_ecgid_records(data_dir)
    by_subject: Dict[str, str] = {}
    for subj, path in records:
        by_subject.setdefault(subj, path)  # first record per subject is enough
    out = []
    for subj, path in sorted(by_subject.items()):
        try:
            _, _, demo = _load_ecgid_signal_and_demo(path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ECG-ID {subj}: {e}")
            demo = {"age": None, "sex": None}
        out.append({"subject": subj, **demo})
    return out


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------
def age_band_edges(ages: Sequence[int], n_bands: int = 2) -> List[float]:
    """Band edges at quantiles of the cohort age distribution (median for 2)."""
    a = np.asarray([x for x in ages if x is not None], dtype=float)
    qs = np.linspace(0, 1, n_bands + 1)[1:-1]
    return [float(np.quantile(a, q)) for q in qs]


def age_to_band(age: int, edges: Sequence[float]) -> int:
    b = 0
    for e in edges:
        if age >= e:
            b += 1
    return b


# ---------------------------------------------------------------------------
# Subject-disjoint windowed datasets
# ---------------------------------------------------------------------------
def build_attribute_split_mitbih(
    attribute: str = "sex",
    records: Sequence[str] = ALL_RECORDS,
    train_frac: float = 0.6,
    split_seed: int = 12345,
    window_sec: float = REID.window_sec,
    step_sec: float = REID.step_sec,
    allow_remote: bool = True,
    n_age_bands: int = 2,
) -> Tuple[AttributeDataset, AttributeDataset]:
    """MIT-BIH subject-disjoint windows labelled by `attribute` ('sex'|'age')."""
    if attribute not in ("sex", "age"):
        raise ValueError("attribute must be 'sex' or 'age'")

    # Gather signals + demographics per record.
    entries = []  # (subject_key, signal, fs, demo)
    for rid in records:
        try:
            sig, fs, demo = _load_mitbih_signal_and_demo(rid, allow_remote)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MIT-BIH {rid}: {e}; skipping")
            continue
        if demo.get(attribute) is None:
            continue
        entries.append((_mitbih_subject_key(rid), preprocess_signal(sig), fs, demo))

    if not entries:
        raise RuntimeError(f"No MIT-BIH records with attribute '{attribute}'.")

    fs = entries[0][2]
    edges = (age_band_edges([d["age"] for _, _, _, d in entries], n_age_bands)
             if attribute == "age" else [])
    return _assemble(entries, attribute, edges, fs, train_frac, split_seed,
                     window_sec, step_sec)


def build_attribute_split_ecgid(
    attribute: str = "sex",
    train_frac: float = 0.6,
    split_seed: int = 12345,
    window_sec: float = REID.window_sec,
    step_sec: float = REID.step_sec,
    allow_remote: bool = True,
    data_dir: Optional[Path] = None,
    n_age_bands: int = 2,
    one_record_per_subject: bool = False,
) -> Tuple[AttributeDataset, AttributeDataset]:
    """ECG-ID subject-disjoint windows labelled by `attribute` ('sex'|'age')."""
    if attribute not in ("sex", "age"):
        raise ValueError("attribute must be 'sex' or 'age'")
    if allow_remote:
        from src.data.ecgid import _ensure_ecgid_downloaded
        _ensure_ecgid_downloaded(data_dir)

    records = _list_ecgid_records(data_dir)
    if not records:
        raise RuntimeError("No ECG-ID records found.")

    # demo per subject (constant), but window every record of the subject.
    demo_by_subject: Dict[str, Dict] = {}
    for subj, path in records:
        if subj not in demo_by_subject:
            _, _, demo_by_subject[subj] = _load_ecgid_signal_and_demo(path)

    entries = []
    seen = set()
    fs_val = DATA.ecgid_fs
    for subj, path in records:
        if one_record_per_subject and subj in seen:
            continue
        demo = demo_by_subject.get(subj, {})
        if demo.get(attribute) is None:
            continue
        sig, fs_val, _ = _load_ecgid_signal_and_demo(path)
        entries.append((subj, _preprocess_ecgid(sig), fs_val, demo))
        seen.add(subj)

    if not entries:
        raise RuntimeError(f"No ECG-ID records with attribute '{attribute}'.")

    edges = (age_band_edges([d["age"] for _, _, _, d in entries], n_age_bands)
             if attribute == "age" else [])
    return _assemble(entries, attribute, edges, float(fs_val), train_frac,
                     split_seed, window_sec, step_sec)


def _assemble(entries, attribute, edges, fs, train_frac, split_seed,
              window_sec, step_sec) -> Tuple[AttributeDataset, AttributeDataset]:
    """Window entries and split subject-disjoint. Shared by both corpora."""
    window_samples = int(round(fs * window_sec))
    step_samples = int(round(fs * step_sec))

    # Class names.
    if attribute == "sex":
        class_names = ["F", "M"]

        def label_of(demo):
            return class_names.index(demo["sex"])
    else:
        class_names = ([f"age<{edges[0]:.0f}", f"age>={edges[0]:.0f}"]
                       if len(edges) == 1
                       else [f"age_band_{i}" for i in range(len(edges) + 1)])

        def label_of(demo):
            return age_to_band(demo["age"], edges)

    subj_keys = [e[0] for e in entries]
    train_subj, test_subj = _subject_split(subj_keys, train_frac, split_seed)
    subject_index = {s: i for i, s in enumerate(sorted(set(subj_keys)))}

    def make(split_subj):
        Xs, ys, subj_ids = [], [], []
        for subj, sig, _fs, demo in entries:
            if subj not in split_subj:
                continue
            w = _window_signal(sig, window_samples, step_samples)
            if len(w) == 0:
                continue
            Xs.append(w)
            ys.extend([label_of(demo)] * len(w))
            subj_ids.extend([subject_index[subj]] * len(w))
        if not Xs:
            return (np.empty((0, window_samples), np.float32),
                    np.empty((0,), np.int32), np.empty((0,), np.int32))
        return (np.concatenate(Xs, 0),
                np.asarray(ys, np.int32),
                np.asarray(subj_ids, np.int32))

    Xtr, ytr, str_ = make(train_subj)
    Xte, yte, ste = make(test_subj)
    subject_ids = sorted(set(subj_keys))

    train = AttributeDataset(Xtr, ytr, str_, class_names, subject_ids, fs, attribute)
    test = AttributeDataset(Xte, yte, ste, class_names, subject_ids, fs, attribute)
    return train, test


def majority_class_floor(y: np.ndarray, n_classes: int) -> float:
    """Chance line = the test-set majority-class rate (not naive 1/n)."""
    if len(y) == 0:
        return float("nan")
    counts = np.bincount(y, minlength=n_classes)
    return float(counts.max() / counts.sum())
