"""
Apply a DP mechanism to a whole dataset.

By default uses the vectorised implementation in `mechanisms_fast.py`
(~100x faster than the diffprivlib loop). Set DP.use_fast = False in
configs/config.py to fall back to the diffprivlib path.

Outputs are statistically identical for Laplace, and use the same
calibration algorithm (Balle-Wang for Gaussian, the truncated Laplacian
of Geng et al. for LaplaceBoundedNoise -- which bounds the *noise*, not
the output domain). The only thing that differs is the speed.

This perturbs ONLY the signal data (encoder inputs). Labels and the
decoder side are never touched.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from configs import DP
from src.dp.mechanisms import apply_dp as apply_dp_slow
from src.dp.mechanisms_fast import apply_dp_fast


def perturb_encoder_inputs(
    encoder_inputs: np.ndarray,
    mechanism: str,
    eps: float,
    delta: float,
    sensitivity: float,
    seed: Optional[int] = None,
    use_fast: Optional[bool] = None,
    clip: Optional[float] = None,
) -> np.ndarray:
    """
    Apply a DP mechanism to all encoder inputs at once.

    Parameters
    ----------
    encoder_inputs : (n_sequences, beats_per_group, beat_length)
        The signal data to perturb.
    mechanism, eps, delta, sensitivity, seed
        Forwarded to the underlying DP function.
    use_fast : bool, optional
        Override the default fast/slow choice. If None, uses DP.use_fast.
    clip : float, optional
        If set, clamp every value to [-clip, +clip] BEFORE adding noise. This is
        what an unbounded-adjacency reading of DP requires: without a bound on
        the value a single sample may take, the global sensitivity of releasing
        the signal is unbounded. With clipping at C the global L1 sensitivity is
        2C, which the caller must pass as `sensitivity`. Left as None (the
        default) nothing is clipped and the calibration is the jump-based one
        used throughout the main experiments.

    Returns
    -------
    np.ndarray
        Perturbed array with the same shape and dtype float32.
    """
    if encoder_inputs.size == 0:
        return encoder_inputs.copy()

    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be > 0")
        encoder_inputs = np.clip(encoder_inputs, -clip, clip)

    if use_fast is None:
        use_fast = DP.use_fast

    if use_fast:
        return apply_dp_fast(
            encoder_inputs,
            mechanism=mechanism,
            eps=eps,
            delta=delta,
            sensitivity=sensitivity,
            seed=seed,
        ).astype(np.float32)

    # Slow path: flatten + diffprivlib loop
    original_shape = encoder_inputs.shape
    flat = encoder_inputs.reshape(-1)
    noisy_flat = apply_dp_slow(
        flat,
        mechanism=mechanism,
        eps=eps,
        delta=delta,
        sensitivity=sensitivity,
        seed=seed,
    )
    return noisy_flat.reshape(original_shape).astype(np.float32)
