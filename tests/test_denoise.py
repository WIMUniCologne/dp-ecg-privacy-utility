"""
Unit tests for the temporal-correlation denoisers (src/dp/denoise.py).

The defining property of the attack: on a smooth, autocorrelated signal
corrupted by i.i.d. noise, a low-pass denoiser must (a) preserve shape and
length, (b) remove a large fraction of the noise (reduce MSE-to-clean), and
(c) keep the QRS peak roughly intact. These tests assert exactly that on a
synthetic ECG-like signal, so they run without TensorFlow or PhysioNet data.

Run:
    python -m pytest tests/test_denoise.py -q
or, without pytest:
    python tests/test_denoise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dp.denoise import (
    DENOISE_METHODS,
    denoise,
    describe_denoise,
    savgol_window_for_fs,
)


def _synthetic_ecg(n: int, window_samples: int, fs: float, seed: int = 0):
    """n z-normalized autocorrelated ECG-like windows."""
    rng = np.random.default_rng(seed)
    t = np.arange(window_samples) / fs

    def g(c, w, a):
        return a * np.exp(-0.5 * ((t - c) / w) ** 2)

    rr = 0.8
    out = np.empty((n, window_samples), dtype=np.float64)
    for i in range(n):
        sig = np.zeros(window_samples)
        for c in np.arange(-rr, t[-1] + rr, rr) + rng.uniform(0, rr):
            sig += g(c - 0.2 * rr, 0.025, 0.15)
            sig += g(c, 0.012, 1.0)            # QRS R-peak
            sig += g(c + 0.02 * rr, 0.012, -0.25)
            sig += g(c + 0.3 * rr, 0.040, 0.30)
        out[i] = (sig - sig.mean()) / (sig.std() + 1e-8)
    return out


def _add_iid_noise(x, sigma, seed=1):
    rng = np.random.default_rng(seed)
    return (x + rng.normal(0.0, sigma, size=x.shape)).astype(np.float32)


def test_savgol_window_is_odd_and_valid():
    for fs in (360.0, 500.0, 128.0):
        w = savgol_window_for_fs(fs)
        assert w % 2 == 1, f"window must be odd, got {w} at fs={fs}"
        assert w > 3, f"window must exceed polyorder 3, got {w}"


def test_shape_and_dtype_preserved():
    fs = 360.0
    x = _synthetic_ecg(4, int(fs * 2), fs)
    noised = _add_iid_noise(x, 0.4)
    for method in DENOISE_METHODS:
        out = denoise(noised, method=method, fs=fs)
        assert out.shape == noised.shape
        assert out.dtype == np.float32
    # 1D input round-trips too.
    out1d = denoise(noised[0], method="savgol", fs=fs)
    assert out1d.shape == noised[0].shape


def test_denoise_reduces_noise_mse():
    """Each filter must move the noised signal closer to the clean one."""
    fs = 360.0
    x = _synthetic_ecg(8, int(fs * 2), fs)
    noised = _add_iid_noise(x, 0.4)
    mse_noised = np.mean((noised - x) ** 2)
    for method in DENOISE_METHODS:
        denoised = denoise(noised, method=method, fs=fs)
        mse_denoised = np.mean((denoised - x) ** 2)
        assert mse_denoised < 0.6 * mse_noised, (
            f"{method}: expected >40% MSE drop, got "
            f"{1 - mse_denoised / mse_noised:.2%}"
        )


def test_qrs_peak_preserved():
    """The R-peak amplitude must survive Savitzky-Golay smoothing."""
    fs = 360.0
    x = _synthetic_ecg(8, int(fs * 2), fs)
    noised = _add_iid_noise(x, 0.3)
    denoised = denoise(noised, method="savgol", fs=fs)
    ratios = []
    for c, d in zip(x, denoised):
        k = int(np.argmax(np.abs(c)))
        ratios.append(abs(d[k]) / abs(c[k]))
    mean_ratio = float(np.mean(ratios))
    assert 0.7 < mean_ratio < 1.3, f"QRS peak not preserved: ratio={mean_ratio:.3f}"


def test_determinism_and_train_test_consistency():
    """Same input -> same output (a fixed transform usable on train and test)."""
    fs = 500.0
    x = _synthetic_ecg(3, int(fs * 2), fs)
    noised = _add_iid_noise(x, 0.5)
    a = denoise(noised, method="savgol", fs=fs)
    b = denoise(noised, method="savgol", fs=fs)
    assert np.array_equal(a, b)


def test_empty_and_describe():
    assert denoise(np.empty((0, 720), dtype=np.float32), method="savgol", fs=360).size == 0
    rec = describe_denoise("savgol", fs=360.0)
    assert rec["method"] == "savgol" and rec["window_length"] % 2 == 1


def test_unknown_method_raises():
    try:
        denoise(np.zeros((2, 100), dtype=np.float32), method="nope", fs=360)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
