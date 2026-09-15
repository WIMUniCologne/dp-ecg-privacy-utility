"""
Data-level differential privacy mechanisms for ECG signals.

Three mechanisms, applied sample-wise to a 1D signal:
- Laplace            (pure DP, delta=0 allowed)
- LaplaceBounded     (Laplace with bounded support, requires delta > 0)
- GaussianAnalytic   (analytic Gaussian, requires delta > 0)

Sensitivity calibration is done OUTSIDE this module — pass a fixed sensitivity in.
All randomization goes through diffprivlib (same as your existing code).

Usage:
    from src.dp.mechanisms import apply_dp
    noisy = apply_dp(signal, mechanism="laplace", eps=1.0, delta=0.0,
                     sensitivity=0.29, seed=42)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from diffprivlib.mechanisms import (
    Laplace,
    LaplaceBoundedNoise,
    GaussianAnalytic,
)


# Valid mechanism names
MECHANISMS = ("laplace", "laplace_bounded", "gaussian_analytic")


def _vector_randomise(mech, x: np.ndarray) -> np.ndarray:
    """Apply diffprivlib's randomise sample-wise. Returns float32 array."""
    flat = np.ravel(x)
    noisy = np.fromiter(
        (mech.randomise(float(v)) for v in flat),
        dtype=np.float64,
        count=flat.size,
    )
    return noisy.reshape(x.shape).astype(np.float32)


def _make_seed(rng: np.random.Generator) -> int:
    """Draw a random 31-bit integer from rng for diffprivlib seeding."""
    return int(rng.integers(0, 2**31 - 1))


def apply_laplace(
    signal: np.ndarray,
    eps: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply Laplace mechanism (pure DP)."""
    if eps <= 0:
        raise ValueError("epsilon must be > 0")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")
    mech = Laplace(
        epsilon=eps,
        sensitivity=sensitivity,
        random_state=_make_seed(rng),
    )
    return _vector_randomise(mech, signal)


def apply_laplace_bounded(
    signal: np.ndarray,
    eps: float,
    delta: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply LaplaceBoundedNoise mechanism. Requires delta > 0."""
    if eps <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("LaplaceBounded requires 0 < delta < 1")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")
    mech = LaplaceBoundedNoise(
        epsilon=eps,
        sensitivity=sensitivity,
        delta=delta,
        random_state=_make_seed(rng),
    )
    return _vector_randomise(mech, signal)


def apply_gaussian_analytic(
    signal: np.ndarray,
    eps: float,
    delta: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply analytic Gaussian mechanism. Requires delta > 0."""
    if eps <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("GaussianAnalytic requires 0 < delta < 1")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")
    mech = GaussianAnalytic(
        epsilon=eps,
        delta=delta,
        sensitivity=sensitivity,
        random_state=_make_seed(rng),
    )
    return _vector_randomise(mech, signal)


def apply_dp(
    signal: np.ndarray,
    mechanism: str,
    eps: float,
    delta: float,
    sensitivity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply a DP mechanism to an ECG signal (1D or 2D array).

    Parameters
    ----------
    signal : np.ndarray
        The signal to perturb. Any shape; noise is added element-wise.
    mechanism : str
        One of "laplace", "laplace_bounded", "gaussian_analytic".
    eps : float
        Privacy parameter epsilon (> 0).
    delta : float
        Privacy parameter delta. Must be > 0 for bounded/gaussian, may be 0 for laplace.
    sensitivity : float
        L1 (or per-sample) sensitivity. Project default: 0.29.
    seed : int, optional
        If set, makes noise reproducible. If None, uses fresh randomness.

    Returns
    -------
    np.ndarray
        Noisy signal with same shape and dtype float32.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. Valid: {MECHANISMS}"
        )
    rng = np.random.default_rng(seed)

    if mechanism == "laplace":
        return apply_laplace(signal, eps, sensitivity, rng)
    if mechanism == "laplace_bounded":
        return apply_laplace_bounded(signal, eps, delta, sensitivity, rng)
    if mechanism == "gaussian_analytic":
        return apply_gaussian_analytic(signal, eps, delta, sensitivity, rng)

    # Unreachable, kept for safety
    raise RuntimeError(f"Unhandled mechanism: {mechanism}")


def is_valid_config(mechanism: str, eps: float, delta: float) -> bool:
    """
    Check whether (mechanism, eps, delta) is a valid configuration.

    Useful for skipping invalid grid points (e.g., delta=0 for gaussian).
    """
    if eps <= 0:
        return False
    if mechanism == "laplace":
        return 0 <= delta < 1
    if mechanism in ("laplace_bounded", "gaussian_analytic"):
        return 0 < delta < 1
    return False
