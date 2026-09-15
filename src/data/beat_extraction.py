"""
Beat extraction from MIT-BIH ECG signals.

This module exposes two beat extraction methods compatible with Mousavi &
Afghah's seq2seq pipeline:

- `extract_beats_qspeaks`: T-wave to T-wave segmentation via qsPeaks
  fiducial point detection (drops beats where P/QRSon/Q/R/S/QRSoff/T cannot
  be located reliably). This matches the MATLAB pipeline used by Mousavi.

- `extract_beats_rr_resample`: simpler R-peak to R-peak segments, resampled
  to the target length. Kept for ablation / comparison.

Both return:
    beats: (n_beats, beat_length) float32 array
    labels: list of AAMI class chars ('N', 'S', 'V', 'F', 'Q')

The signal passed in is expected to be already preprocessed (z-normalized).
See `src/data/mitbih.preprocess_signal()` for the canonical preprocessing.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.data.qspeaks import qs_peaks


# ---------------------------------------------------------------------------
# 1D resampling
# ---------------------------------------------------------------------------
def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D signal to fixed length via linear interpolation."""
    n = len(x)
    if n == target_len:
        return x.astype(np.float32)
    if n < 2:
        return np.full(target_len, float(x[0]) if n == 1 else 0.0, dtype=np.float32)
    old_idx = np.linspace(0.0, 1.0, n)
    new_idx = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


# ---------------------------------------------------------------------------
# qsPeaks-based extraction (Mousavi MATLAB-exact, T-wave to T-wave)
# ---------------------------------------------------------------------------
def extract_beats_qspeaks(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    aami_labels: List[str],
    beat_length: int,
    fs: float,
    keep_classes: Tuple[str, ...] = ("N", "S", "V", "F", "Q"),
    max_first_beat_samples: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract beats via Mousavi's qsPeaks pipeline (T-wave to T-wave).

    Reference: seq2seq_mitbih_AAMI_DS1DS2.m
      1. Run qsPeaks → keep only beats with all 7 fiducials detected.
      2. Use T-wave positions (column 6) for segmentation.
      3. Beat i (i > 0): signal[tpeaks[i-1] : tpeaks[i]], resampled.
         Beat 0: signal[0 : tpeaks[0]], trimmed to ~1 sec.

    Args:
        signal: 1D preprocessed (z-normalized) ECG signal.
        r_peaks: indices of R-peak positions in `signal`.
        aami_labels: AAMI class char per R-peak (same length as r_peaks).
        beat_length: target length per beat after resampling (e.g., 280).
        fs: sampling rate (Hz).
        keep_classes: tuple of AAMI classes to retain. Others are dropped.
        max_first_beat_samples: clip the first beat segment to this many
            samples (Mousavi uses min(fs, len) for ~1 second).

    Returns:
        beats: (n_beats, beat_length) float32 array.
        labels: list of AAMI class chars, aligned with beats.
    """
    if len(r_peaks) == 0:
        return np.empty((0, beat_length), dtype=np.float32), []

    fid_pks = qs_peaks(signal, r_peaks, fs)
    if fid_pks.shape[0] == 0:
        return np.empty((0, beat_length), dtype=np.float32), []

    tpeaks = fid_pks[:, 6]  # T-wave positions
    kept_r_peaks = fid_pks[:, 3]  # R positions of kept beats

    # Filter labels to align with the subset of beats qsPeaks accepted
    r_peaks_arr = np.asarray(r_peaks).astype(np.int64)
    kept_set = set(int(x) for x in kept_r_peaks)
    kept_mask_in_orig = np.array([int(rp) in kept_set for rp in r_peaks_arr])
    filtered_labels = [lbl for lbl, k in zip(aami_labels, kept_mask_in_orig) if k]

    # Defensive alignment in case lengths differ
    if len(filtered_labels) != len(kept_r_peaks):
        n = min(len(filtered_labels), len(kept_r_peaks))
        filtered_labels = filtered_labels[:n]
        tpeaks = tpeaks[:n]

    n_signal = len(signal)
    beats: List[np.ndarray] = []
    labels: List[str] = []

    # First beat: from start of signal to first T-wave (clipped to ~1s)
    if len(filtered_labels) > 0 and filtered_labels[0] in keep_classes:
        t0 = int(tpeaks[0])
        if t0 > 1:
            seg = signal[:t0]
            if max_first_beat_samples is not None and len(seg) > max_first_beat_samples:
                seg = seg[:max_first_beat_samples]
            if len(seg) >= 2:
                beats.append(_resample_1d(seg, beat_length))
                labels.append(filtered_labels[0])

    # Subsequent beats: T-to-T segments
    for i in range(1, len(filtered_labels)):
        lbl = filtered_labels[i]
        if lbl not in keep_classes:
            continue
        start = int(tpeaks[i - 1])
        end = int(tpeaks[i])
        if end <= start + 1 or end > n_signal:
            continue
        seg = signal[start:end]
        if len(seg) < 2:
            continue
        beats.append(_resample_1d(seg, beat_length))
        labels.append(lbl)

    if not beats:
        return np.empty((0, beat_length), dtype=np.float32), []
    return np.stack(beats).astype(np.float32), labels


# ---------------------------------------------------------------------------
# R-to-R extraction (simpler, used for ablation)
# ---------------------------------------------------------------------------
def extract_beats_rr_resample(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    aami_labels: List[str],
    beat_length: int,
    keep_classes: Tuple[str, ...] = ("N", "S", "V", "F", "Q"),
    max_first_beat_samples: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract beats as RR intervals, resampled to fixed length.

    Beat i (i > 0): signal[r_peaks[i-1] : r_peaks[i]], resampled to
    beat_length. Beat 0: signal[0 : r_peaks[0]].

    Simpler alternative to qsPeaks. Used for ablation experiments.
    """
    if len(r_peaks) == 0:
        return np.empty((0, beat_length), dtype=np.float32), []

    beats: List[np.ndarray] = []
    labels: List[str] = []

    # First beat
    if len(aami_labels) > 0 and aami_labels[0] in keep_classes:
        end = int(r_peaks[0])
        if end > 0:
            seg = signal[:end]
            if max_first_beat_samples is not None and len(seg) > max_first_beat_samples:
                seg = seg[:max_first_beat_samples]
            beats.append(_resample_1d(seg, beat_length))
            labels.append(aami_labels[0])

    # Subsequent beats
    for i in range(1, len(r_peaks)):
        lbl = aami_labels[i]
        if lbl not in keep_classes:
            continue
        start = int(r_peaks[i - 1])
        end = int(r_peaks[i])
        if end <= start:
            continue
        seg = signal[start:end]
        if len(seg) < 2:
            continue
        beats.append(_resample_1d(seg, beat_length))
        labels.append(lbl)

    if not beats:
        return np.empty((0, beat_length), dtype=np.float32), []
    return np.stack(beats).astype(np.float32), labels
