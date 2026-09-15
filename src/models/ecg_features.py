"""
Hand-crafted per-beat ECG features for the second (feature-based) utility model.

Experiment 2 asks whether the L-shaped utility curve is an artifact of the one
deep architecture (Mousavi & Afghah seq2seq). To answer it we train a classical
Random Forest on standard morphology descriptors -- an architecturally distant
model that shares none of the seq2seq inductive biases.

Consistency note (important for a fair comparison)
--------------------------------------------------
The main model consumes the qsPeaks T-to-T beat resampled to a fixed length
(280 samples) and z-normalized; the DP mechanism perturbs exactly those sample
values. To keep "everything except the architecture identical", these features
are computed from that SAME DP-protected beat vector. We therefore do NOT use
raw RR / inter-beat timing: it is normalised out of the beat representation the
main model sees and is not touched by the amplitude DP, so feeding it in would
(a) break the identical-protection requirement and (b) leak utility independent
of epsilon. The descriptors below are all intra-beat morphology -- the protected
information -- so both models see the same protected signal.

No tuning: the feature set and the model defaults are fixed. The point is that a
generic second model shows the same shape, not that it is optimised.
"""
from __future__ import annotations

from typing import List

import numpy as np

N_SEGMENTS = 10
N_FREQ = 8


def feature_names(beat_length: int = 280) -> List[str]:
    names = [
        "mean", "std", "min", "max", "range", "rms", "mean_abs",
        "median", "iqr", "skew", "kurtosis",
        "argmax_norm", "argmin_norm", "val_at_max", "val_at_min",
        "zero_crossings", "turning_points", "total_variation", "max_abs_slope",
    ]
    names += [f"seg_energy_{i}" for i in range(N_SEGMENTS)]
    names += [f"rfft_mag_{i}" for i in range(N_FREQ)]
    return names


def extract_beat_features(beats: np.ndarray) -> np.ndarray:
    """
    Vectorised morphology features for a batch of beats.

    Parameters
    ----------
    beats : (n_beats, beat_length) float array (z-normalized, possibly DP-noised).

    Returns
    -------
    (n_beats, n_features) float32 array, columns aligned with feature_names().
    """
    X = np.asarray(beats, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    n, L = X.shape
    eps = 1e-8

    mean = X.mean(1)
    std = X.std(1)
    xmin = X.min(1)
    xmax = X.max(1)
    rng = xmax - xmin
    rms = np.sqrt((X ** 2).mean(1))
    mean_abs = np.abs(X).mean(1)
    median = np.median(X, axis=1)
    q75 = np.quantile(X, 0.75, axis=1)
    q25 = np.quantile(X, 0.25, axis=1)
    iqr = q75 - q25

    z = (X - mean[:, None]) / (std[:, None] + eps)
    skew = (z ** 3).mean(1)
    kurtosis = (z ** 4).mean(1) - 3.0

    argmax = X.argmax(1) / max(L - 1, 1)
    argmin = X.argmin(1) / max(L - 1, 1)
    val_at_max = xmax
    val_at_min = xmin

    sign = np.sign(X)
    zero_crossings = (np.abs(np.diff(sign, axis=1)) > 0).sum(1).astype(float)
    d = np.diff(X, axis=1)
    turning = (np.abs(np.diff(np.sign(d), axis=1)) > 0).sum(1).astype(float)
    total_variation = np.abs(d).sum(1)
    max_abs_slope = np.abs(d).max(1)

    # Segment energies (P / QRS / T distribution along the beat).
    seg = np.array_split(np.arange(L), N_SEGMENTS)
    seg_energy = np.stack([(X[:, idx] ** 2).sum(1) for idx in seg], axis=1)

    # Low-order spectral magnitude (compact morphology).
    mag = np.abs(np.fft.rfft(X, axis=1))
    freq = mag[:, :N_FREQ] if mag.shape[1] >= N_FREQ else np.pad(
        mag, ((0, 0), (0, N_FREQ - mag.shape[1])))

    feats = np.column_stack([
        mean, std, xmin, xmax, rng, rms, mean_abs, median, iqr, skew, kurtosis,
        argmax, argmin, val_at_max, val_at_min,
        zero_crossings, turning, total_variation, max_abs_slope,
        seg_energy, freq,
    ])
    return feats.astype(np.float32)
