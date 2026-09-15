"""
Temporal-correlation denoisers for the re-identification stress test.

Motivation
----------
The DP mechanisms in this repository add *independent* (i.i.d.) noise to each
sample of a z-normalized ECG window. An ECG window is smooth and strongly
autocorrelated, so a low-pass / denoising step can average much of that noise
away while leaving the signal largely intact: the signal is correlated across
time, the noise is not. A temporally-aware attacker exploits exactly this gap.

This module provides cheap, standard denoisers that operate in the SAME domain
the re-identification attacker consumes -- the z-normalized signal *window*
(shape ``(..., window_samples)``), applied identically to the attacker's train
and test inputs (see ``scripts/train_reid_denoise.py``). It does NOT touch the
privacy mechanism: the released (noised) data is held fixed and only the
adversary gains a stronger pre-processing front end. Same data, smarter
adversary.

Important
---------
The re-ID attacker does not consume 280-sample *beats*; it consumes sliding
signal *windows* (MIT-BIH: 360 Hz x 2 s = 720 samples; ECG-ID: 500 Hz x 2 s =
1000 samples). The Savitzky-Golay window is therefore chosen from the sampling
rate, not from a fixed sample count, so that it is short enough not to blunt the
QRS complex (~80-100 ms). Verify on plots (``scripts/diagnose/plot_denoise_sanity.py``)
before trusting any number.

Only numpy is required for the moving-average and Gaussian methods; SciPy is
used for Savitzky-Golay (and is already a dependency of the data pipeline). No
TensorFlow dependency here -- the optional denoising-autoencoder attacker lives
in ``src/models/denoise_ae.py``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Methods that this module knows how to apply.
DENOISE_METHODS = ("savgol", "moving_average", "gaussian")

# Default Savitzky-Golay smoothing time-scale, in milliseconds. At 360 Hz this
# is ~9 samples, at 500 Hz ~13 samples -- well below a QRS complex (~80-100 ms),
# so the R-peak morphology is preserved while sample-level i.i.d. noise is
# averaged out.
DEFAULT_SAVGOL_MS = 25.0
DEFAULT_SAVGOL_POLYORDER = 3

# Default Gaussian smoothing time-scale (sigma), in milliseconds.
DEFAULT_GAUSSIAN_MS = 8.0


def savgol_window_for_fs(
    fs: float,
    ms: float = DEFAULT_SAVGOL_MS,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> int:
    """
    Choose an odd Savitzky-Golay window length (in samples) from a sampling
    rate and a target smoothing time-scale.

    The window is forced odd (a SavGol requirement) and forced to be strictly
    greater than ``polyorder`` so the polynomial fit is well posed.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    ms : float
        Target smoothing time-scale in milliseconds.
    polyorder : int
        Polynomial order the window must exceed.

    Returns
    -------
    int
        Odd window length in samples.
    """
    win = int(round(fs * ms / 1000.0))
    if win % 2 == 0:
        win += 1
    # SavGol requires window_length > polyorder.
    if win <= polyorder:
        win = polyorder + 1 + (polyorder % 2)  # smallest valid odd window
        if win % 2 == 0:
            win += 1
    return win


def _as_2d(x: np.ndarray) -> "tuple[np.ndarray, tuple]":
    """Reshape ``(..., L)`` to ``(N, L)`` for batch filtering; remember shape."""
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    if x.ndim == 1:
        return x[None, :], orig_shape
    return x.reshape(-1, orig_shape[-1]), orig_shape


def _savgol(
    x: np.ndarray,
    fs: Optional[float],
    window_length: Optional[int],
    polyorder: int,
    ms: float,
) -> np.ndarray:
    from scipy.signal import savgol_filter

    if window_length is None:
        if fs is None:
            raise ValueError(
                "savgol requires either an explicit window_length or fs so the "
                "window can be derived from the sampling rate."
            )
        window_length = savgol_window_for_fs(fs, ms=ms, polyorder=polyorder)
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        raise ValueError(
            f"savgol window_length ({window_length}) must exceed polyorder "
            f"({polyorder})."
        )
    L = x.shape[-1]
    if window_length > L:
        # Window cannot exceed the signal length; fall back to the largest
        # valid odd window.
        window_length = L if L % 2 == 1 else L - 1
        if window_length <= polyorder:
            return x.copy()
    return savgol_filter(
        x, window_length=window_length, polyorder=polyorder,
        axis=-1, mode="interp",
    )


def _moving_average(
    x: np.ndarray,
    fs: Optional[float],
    size: Optional[int],
    ms: float,
) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    if size is None:
        if fs is None:
            raise ValueError(
                "moving_average requires either an explicit size or fs."
            )
        size = max(2, int(round(fs * ms / 1000.0)))
    return uniform_filter1d(x, size=size, axis=-1, mode="nearest")


def _gaussian(
    x: np.ndarray,
    fs: Optional[float],
    sigma: Optional[float],
    ms: float,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    if sigma is None:
        if fs is None:
            raise ValueError(
                "gaussian requires either an explicit sigma or fs."
            )
        sigma = max(0.5, fs * ms / 1000.0)
    return gaussian_filter1d(x, sigma=sigma, axis=-1, mode="nearest")


def denoise(
    x: np.ndarray,
    method: str = "savgol",
    *,
    fs: Optional[float] = None,
    window_length: Optional[int] = None,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    savgol_ms: float = DEFAULT_SAVGOL_MS,
    size: Optional[int] = None,
    moving_average_ms: float = DEFAULT_SAVGOL_MS,
    sigma: Optional[float] = None,
    gaussian_ms: float = DEFAULT_GAUSSIAN_MS,
) -> np.ndarray:
    """
    Denoise a beat / window or a batch of them along the last axis.

    The transform is deterministic and stateless, so it can be applied
    identically to the attacker's train and test inputs.

    Parameters
    ----------
    x : np.ndarray
        Signal of shape ``(..., window_samples)``. The last axis is time.
    method : {"savgol", "moving_average", "gaussian"}
        - ``savgol``: Savitzky-Golay filter. Preserves the QRS peak shape
          better than a plain moving average; the strongest of the cheap
          low-pass attackers and the recommended default.
        - ``moving_average``: uniform (boxcar) smoothing. A weaker attacker; a
          good lower bound on what trivial post-processing recovers.
        - ``gaussian``: Gaussian smoothing. Between the two above.
    fs : float, optional
        Sampling rate in Hz, used to derive a sensible window/sigma when one is
        not given explicitly. Strongly recommended.
    window_length, polyorder, savgol_ms
        Savitzky-Golay parameters. ``window_length`` overrides the fs-derived
        window; ``savgol_ms`` sets the fs-derived window's time-scale.
    size, moving_average_ms
        Moving-average parameters (boxcar width in samples / milliseconds).
    sigma, gaussian_ms
        Gaussian parameters (sigma in samples / milliseconds).

    Returns
    -------
    np.ndarray
        Denoised array, same shape as ``x``, dtype float32.
    """
    if x is None or np.size(x) == 0:
        return np.asarray(x, dtype=np.float32)

    x2d, orig_shape = _as_2d(x)

    if method == "savgol":
        out = _savgol(x2d, fs, window_length, polyorder, savgol_ms)
    elif method == "moving_average":
        out = _moving_average(x2d, fs, size, moving_average_ms)
    elif method == "gaussian":
        out = _gaussian(x2d, fs, sigma, gaussian_ms)
    else:
        raise ValueError(
            f"Unknown denoise method '{method}'. "
            f"Choose one of {DENOISE_METHODS}."
        )

    return out.reshape(orig_shape).astype(np.float32)


def describe_denoise(
    method: str,
    fs: Optional[float],
    *,
    window_length: Optional[int] = None,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    savgol_ms: float = DEFAULT_SAVGOL_MS,
    size: Optional[int] = None,
    moving_average_ms: float = DEFAULT_SAVGOL_MS,
    sigma: Optional[float] = None,
    gaussian_ms: float = DEFAULT_GAUSSIAN_MS,
) -> dict:
    """
    Return a JSON-serialisable record of the denoiser's effective parameters,
    for storing alongside the attacker metrics so a run is fully reproducible.
    """
    rec: dict = {"method": method, "fs": fs}
    if method == "savgol":
        eff_win = window_length
        if eff_win is None and fs is not None:
            eff_win = savgol_window_for_fs(fs, ms=savgol_ms, polyorder=polyorder)
        rec.update(window_length=eff_win, polyorder=polyorder, savgol_ms=savgol_ms)
    elif method == "moving_average":
        eff_size = size
        if eff_size is None and fs is not None:
            eff_size = max(2, int(round(fs * moving_average_ms / 1000.0)))
        rec.update(size=eff_size, moving_average_ms=moving_average_ms)
    elif method == "gaussian":
        eff_sigma = sigma
        if eff_sigma is None and fs is not None:
            eff_sigma = max(0.5, fs * gaussian_ms / 1000.0)
        rec.update(sigma=eff_sigma, gaussian_ms=gaussian_ms)
    return rec
