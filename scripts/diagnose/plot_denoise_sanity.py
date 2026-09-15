"""
Sanity check for the denoising attacker BEFORE trusting any re-ID number.

Plots, for a few windows at a low and a high epsilon:
    original (clean)  /  noised (DP)  /  denoised
and prints, per epsilon, how much of the i.i.d. noise the filter removes
(MSE to clean) and how well it preserves the QRS peak (peak-amplitude ratio).

If the QRS peak is blunted or the denoised trace barely tracks the clean
signal, the filter window is wrong -- fix it here before running the attacker.

Data: uses real MIT-BIH/ECG-ID windows when the database is available locally;
otherwise falls back to a synthetic autocorrelated ECG-like signal so the check
runs anywhere (the filter behaviour on i.i.d. noise vs. a smooth signal is the
point, and that is reproduced by the synthetic case).

Usage:
    python scripts/diagnose/plot_denoise_sanity.py
    python scripts/diagnose/plot_denoise_sanity.py --dataset mitbih --method savgol
    python scripts/diagnose/plot_denoise_sanity.py --mechanism gaussian_analytic \
        --eps-low 0.1 --eps-high 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from configs import DP, RESULTS_DIR
from src.dp import denoise, describe_denoise, perturb_encoder_inputs


def synthetic_ecg_windows(
    n: int, window_samples: int, fs: float, seed: int = 0
) -> np.ndarray:
    """A few z-normalized, autocorrelated ECG-like windows (P-QRS-T trains)."""
    rng = np.random.default_rng(seed)
    t = np.arange(window_samples) / fs

    def gauss(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    rr = 0.8  # ~75 bpm
    out = np.empty((n, window_samples), dtype=np.float64)
    for i in range(n):
        sig = np.zeros(window_samples)
        phase = rng.uniform(0, rr)
        beat_centers = np.arange(-rr, t[-1] + rr, rr) + phase
        for c in beat_centers:
            sig += gauss(c - 0.20 * rr, 0.025, 0.15)   # P
            sig += gauss(c - 0.02 * rr, 0.010, -0.10)  # Q
            sig += gauss(c, 0.012, 1.00)               # R (QRS peak)
            sig += gauss(c + 0.02 * rr, 0.012, -0.25)  # S
            sig += gauss(c + 0.30 * rr, 0.040, 0.30)   # T
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        out[i] = sig
    return out


def load_real_windows(dataset: str, n: int) -> "np.ndarray | None":
    """Return a few clean windows from a locally-available corpus, or None."""
    try:
        if dataset == "mitbih":
            from src.data import build_reid_ds1
            train_ds, _ = build_reid_ds1(allow_remote=False)
            X, fs = train_ds.X, train_ds.fs
        else:
            from src.data import build_ecgid_split
            train_ds, _ = build_ecgid_split(allow_remote=False)
            X, fs = train_ds.X, train_ds.fs
    except Exception:
        return None
    if X is None or len(X) == 0:
        return None
    idx = np.linspace(0, len(X) - 1, n).astype(int)
    return X[idx], float(fs)


def qrs_peak_ratio(clean: np.ndarray, other: np.ndarray) -> float:
    """Peak-amplitude preserved at the clean signal's QRS location (mean abs)."""
    ratios = []
    for c, o in zip(clean, other):
        k = int(np.argmax(np.abs(c)))
        denom = abs(c[k])
        if denom > 1e-6:
            ratios.append(abs(o[k]) / denom)
    return float(np.mean(ratios)) if ratios else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Denoiser sanity check.")
    p.add_argument("--dataset", choices=["mitbih", "ecgid", "synthetic"],
                   default="synthetic")
    p.add_argument("--method", default="savgol",
                   choices=["savgol", "moving_average", "gaussian"])
    p.add_argument("--mechanism", default="laplace", choices=list(DP.mechanisms))
    p.add_argument("--eps-low", type=float, default=0.1)
    p.add_argument("--eps-high", type=float, default=1.0)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--fs", type=float, default=None,
                   help="Sampling rate for the synthetic case (default 360).")
    p.add_argument("--window-samples", type=int, default=None,
                   help="Window length for the synthetic case (default 2 s).")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # ---- Get clean windows ----
    real = None if args.dataset == "synthetic" else load_real_windows(
        args.dataset, args.n_windows
    )
    if real is not None:
        clean, fs = real
        source = f"{args.dataset} (real)"
    else:
        fs = args.fs or 360.0
        window_samples = args.window_samples or int(round(fs * 2.0))
        clean = synthetic_ecg_windows(args.n_windows, window_samples, fs)
        source = "synthetic"
        if args.dataset != "synthetic":
            print(f"[note] {args.dataset} not available locally; using synthetic.")

    fs = float(fs)
    win = describe_denoise(args.method, fs=fs)
    print(f"Source: {source}  fs={fs:g} Hz  windows={clean.shape}  "
          f"mechanism={args.mechanism}")
    print(f"Denoiser: {win}")

    delta = DP.delta_for(args.mechanism)

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_values = [args.eps_low, args.eps_high]
    n_rows = len(clean)
    n_cols = len(eps_values)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.5 * n_cols, 2.2 * n_rows),
        squeeze=False, sharex=True,
    )

    print("\nNoise removed (MSE to clean) and QRS peak preservation:")
    print(f"{'eps':>6} | {'MSE noised':>11} | {'MSE denoised':>12} | "
          f"{'MSE drop':>9} | {'QRS ratio':>9}")
    print("-" * 62)

    for j, eps in enumerate(eps_values):
        noised = perturb_encoder_inputs(
            clean.astype(np.float32), mechanism=args.mechanism, eps=eps,
            delta=delta, sensitivity=DP.sensitivity, seed=42,
        )
        denoised = denoise(noised, method=args.method, fs=fs)

        mse_noised = float(np.mean((noised - clean) ** 2))
        mse_denoised = float(np.mean((denoised - clean) ** 2))
        drop = 1.0 - (mse_denoised / mse_noised) if mse_noised > 0 else float("nan")
        qrs = qrs_peak_ratio(clean, denoised)
        print(f"{eps:>6.3g} | {mse_noised:>11.4f} | {mse_denoised:>12.4f} | "
              f"{drop:>8.1%} | {qrs:>9.3f}")

        for i in range(n_rows):
            ax = axes[i][j]
            ax.plot(noised[i], color="0.7", lw=0.7, label="noised (DP)")
            ax.plot(clean[i], color="C0", lw=1.3, label="clean")
            ax.plot(denoised[i], color="C3", lw=1.1, label="denoised")
            if i == 0:
                ax.set_title(f"{args.mechanism}, eps={eps:g}")
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            ax.set_yticks([])
    axes[-1][0].set_xlabel("sample")
    fig.suptitle(
        f"Denoiser sanity: {args.method} (window={win.get('window_length', '-')}), "
        f"{source}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / "figures"
        / f"denoise_sanity_{args.dataset}_{args.method}_{args.mechanism}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure: {out_path}")
    print("Check: the red (denoised) trace should hug the blue (clean) one and "
          "keep the QRS peak (ratio ~1). If it is blunted, shrink the window.")


if __name__ == "__main__":
    main()
