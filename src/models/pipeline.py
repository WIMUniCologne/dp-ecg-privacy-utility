"""
Training-data pipeline for the Mousavi seq2seq classifier.

Mirrors the exact data construction used in the reference code (Mousavi &
Afghah's seq_seq_annot_DS1DS2.py, via Ricarda's TF2 port 00_interpatient_DS.py):

  1. Load all beats from MIT-BIH (DS1 or DS2 split).
  2. Concatenate beats CLASS-BY-CLASS (N, then S, then V), not in temporal
     patient order. (This is critical: it produces pure-class sequences after
     reshape, which the seq2seq model exploits.)
  3. Truncate to a multiple of `max_time`, slice into shape
     (n_seqs, max_time, beat_length). Every sequence is pure-class.
  4. Shuffle sequences (not beats within sequences).
  5. Apply SMOTE per class to oversample minorities, then re-pure-slice.

This module provides:
  - `read_mitbih_wfdb`: loads beats via WFDB + qsPeaks from raw MIT-BIH.
  - `read_mitbih_mat`: loads beats from a pre-processed .mat file (Mousavi's
    distributed dataset). Optional, used only for reproduction checks.
  - `apply_smote`: oversamples minority classes and rebuilds pure-class
    sequences.
  - `build_decoder_inputs`: prepends <GO> token to make teacher-forcing inputs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE

from configs import SEQ2SEQ
from src.data import (
    EXCLUDED_RECORDS,
    extract_beats_qspeaks,
    extract_beats_rr_resample,
    load_mitbih_record,
    preprocess_signal,
)


# ---------------------------------------------------------------------------
# Beat loading
# ---------------------------------------------------------------------------
def _load_all_beats_wfdb(
    record_ids: Sequence[str],
    beat_extraction: str = "qspeaks",
    beat_length: int = 280,
    allow_remote: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate beats from a list of MIT-BIH records."""
    all_beats: List[np.ndarray] = []
    all_labels: List[str] = []

    for rec_id in record_ids:
        if rec_id in EXCLUDED_RECORDS:
            continue
        rec = load_mitbih_record(rec_id, allow_remote=allow_remote)
        sig = preprocess_signal(rec.signal)
        if beat_extraction == "qspeaks":
            beats, labels = extract_beats_qspeaks(
                sig, rec.r_peaks, rec.aami_labels,
                beat_length=beat_length, fs=rec.fs,
                max_first_beat_samples=int(rec.fs),
            )
        elif beat_extraction == "rr_resample":
            beats, labels = extract_beats_rr_resample(
                sig, rec.r_peaks, rec.aami_labels,
                beat_length=beat_length,
                max_first_beat_samples=int(rec.fs),
            )
        else:
            raise ValueError(f"Unknown beat_extraction: {beat_extraction!r}")
        all_beats.append(beats.astype(np.float32))
        all_labels.extend(labels)

    return np.concatenate(all_beats, axis=0), np.asarray(all_labels)


def _build_pure_class_sequences(
    beats: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str] = ("N", "S", "V"),
    max_time: int = 10,
    max_label: int = 50_000,
    seed: int = 654,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build pure-class sequences via Mousavi's class-by-class concatenation.

    For each class in `classes`: collect beats, shuffle within-class, cap to
    `max_label`, append to the global pool. Then reshape into (n_seqs, max_time,
    beat_length). The result has the property that every sequence consists of
    `max_time` beats of the same class.
    """
    rng = np.random.RandomState(seed)
    selected_beats = np.empty((0, beats.shape[1]), dtype=np.float32)
    selected_labels = np.empty((0,), dtype="<U1")

    for cl in classes:
        idx = np.where(labels == cl)[0]
        perm = rng.permutation(len(idx))[:max_label]
        idx = idx[perm]
        selected_beats = np.concatenate([selected_beats, beats[idx]])
        selected_labels = np.concatenate([selected_labels, labels[idx]])

    n_total = (len(selected_beats) // max_time) * max_time
    selected_beats = selected_beats[:n_total]
    selected_labels = selected_labels[:n_total]

    data = selected_beats.reshape(-1, max_time, selected_beats.shape[1])
    seq_labels = selected_labels.reshape(-1, max_time)

    # Shuffle SEQUENCES only (beats within each sequence stay class-homogeneous)
    perm = rng.permutation(len(seq_labels))
    return data[perm], seq_labels[perm]


def read_mitbih_wfdb(
    trainset: int,
    classes: Sequence[str] = ("N", "S", "V"),
    max_time: int = 10,
    beat_length: int = 280,
    max_label: int = 50_000,
    beat_extraction: str = "qspeaks",
    seed: int = 654,
    allow_remote: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MIT-BIH inter-patient split (DS1 if trainset=1 else DS2), extract beats
    via WFDB+qsPeaks, build pure-class sequences.

    Returns:
        data: (n_seqs, max_time, beat_length) float32
        labels: (n_seqs, max_time) char array (each row is all one class)
    """
    if trainset == 1:
        record_ids = SEQ2SEQ.ds1_records
    else:
        record_ids = SEQ2SEQ.ds2_records

    beats, labels = _load_all_beats_wfdb(
        record_ids,
        beat_extraction=beat_extraction,
        beat_length=beat_length,
        allow_remote=allow_remote,
    )
    return _build_pure_class_sequences(
        beats, labels,
        classes=classes, max_time=max_time, max_label=max_label, seed=seed,
    )


def read_mitbih_mat(
    mat_path: str,
    trainset: int,
    classes: Sequence[str] = ("N", "S", "V"),
    max_time: int = 10,
    max_label: int = 50_000,
    seed: int = 654,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load beats from Mousavi's pre-processed .mat file (s2s_mitbih_aami_DS1DS2.mat).
    Used for reproduction checks. Normal experiments use read_mitbih_wfdb().
    """
    from scipy.io import loadmat

    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    samples = d["s2s_mitbih_DS1" if trainset == 1 else "s2s_mitbih_DS2"]

    all_beats: List[np.ndarray] = []
    all_labels: List[str] = []
    for i in range(len(samples)):
        sv = samples[i].seg_values
        sl = samples[i].seg_labels

        if isinstance(sv, np.ndarray) and sv.dtype == object:
            patient_beats = np.array(
                [np.asarray(b).flatten().astype(np.float32) for b in sv]
            )
        else:
            patient_beats = np.asarray(sv, dtype=np.float32)
            if patient_beats.ndim == 1:
                patient_beats = patient_beats.reshape(1, -1)

        if isinstance(sl, str):
            labels_str = sl
        elif isinstance(sl, np.ndarray):
            labels_str = "".join(sl.flatten().astype(str))
        else:
            labels_str = str(sl)

        n = min(patient_beats.shape[0], len(labels_str))
        all_beats.append(patient_beats[:n])
        all_labels.extend(list(labels_str[:n]))

    beats = np.concatenate(all_beats, axis=0)
    labels = np.asarray(all_labels)
    return _build_pure_class_sequences(
        beats, labels,
        classes=classes, max_time=max_time, max_label=max_label, seed=seed,
    )


# ---------------------------------------------------------------------------
# Class vocabulary + decoder inputs
# ---------------------------------------------------------------------------
def build_vocab(classes: Sequence[str], go_token: str = "<GO>") -> Tuple[Dict[str, int], int]:
    """
    Build class → int mapping with <GO> appended at the end.

    Returns:
        char2num: dict mapping each class char to an int id.
        go_id: the id assigned to <GO>.
    """
    char2num = {c: i for i, c in enumerate(classes)}
    char2num[go_token] = len(char2num)
    return char2num, char2num[go_token]


def labels_to_targets(
    seq_labels: np.ndarray, char2num: Dict[str, int],
) -> np.ndarray:
    """Convert (n_seqs, max_time) char labels → int32 targets."""
    return np.array(
        [[char2num[c] for c in seq] for seq in seq_labels], dtype=np.int32,
    )


def build_decoder_inputs(targets: np.ndarray, go_id: int) -> np.ndarray:
    """
    Build teacher-forcing decoder inputs from targets.
    decoder_inputs[i] = [<GO>, targets[i, 0], ..., targets[i, max_time - 2]]
    """
    dec_in = np.empty_like(targets)
    dec_in[:, 0] = go_id
    dec_in[:, 1:] = targets[:, :-1]
    return dec_in


# ---------------------------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------------------------
def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    char2num: Dict[str, int],
    classes: Sequence[str],
    smote_targets: Dict[str, int],
    max_time: int = 10,
    smote_seed: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE on the beat level, then reshape into pure-class sequences.

    Args:
        X: (n_seqs, max_time, beat_length) input sequences.
        y: (n_seqs, max_time) integer targets (class IDs, no <GO>).
        char2num: vocabulary mapping.
        classes: class names (e.g. ("N", "S", "V")).
        smote_targets: per-class target count after SMOTE. Use None or omit
            to leave a class unchanged. Example: {"N": None, "S": 7000, "V": 6000}.
        max_time: beats per sequence (for reshape).
        smote_seed: random seed for the SMOTE algorithm itself.

    Returns:
        (X_smote, y_smote): same shapes as input but with more sequences.
    """
    n_seqs, _, beat_length = X.shape

    # Flatten to beat level
    X_flat = X.reshape(-1, beat_length)
    y_flat = y.flatten()

    # Build sampling_strategy dict using class IDs
    ratio: Dict[int, int] = {}
    for cl in classes:
        cid = char2num[cl]
        current = int((y_flat == cid).sum())
        tgt = smote_targets.get(cl, None)
        if tgt is None or tgt <= current:
            # Keep current count
            ratio[cid] = current
        else:
            ratio[cid] = int(tgt)

    sm = SMOTE(random_state=smote_seed, sampling_strategy=ratio)
    X_res, y_res = sm.fit_resample(X_flat, y_flat)

    # Truncate and reshape into pure-class sequences
    n = (len(X_res) // max_time) * max_time
    X_res = X_res[:n].reshape(-1, max_time, beat_length).astype(np.float32)
    y_res = y_res[:n].reshape(-1, max_time).astype(np.int32)

    return X_res, y_res


# ---------------------------------------------------------------------------
# Per-class macro-F1 helpers (used by training loop)
# ---------------------------------------------------------------------------
def confusion_matrix_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Compact confusion matrix counting only real classes (no <GO>)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    mask = (y_true < n_classes) & (y_pred < n_classes)
    yt = y_true[mask]
    yp = y_pred[mask]
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sensitivity, precision, F1) per class."""
    eps = 1e-10
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    sens = tp / (tp + fn + eps)
    ppv = tp / (tp + fp + eps)
    f1 = 2 * sens * ppv / (sens + ppv + eps)
    return sens, ppv, f1
