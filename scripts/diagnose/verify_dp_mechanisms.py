"""
Verify that the fast vectorised DP mechanisms (mechanisms_fast.py) produce
the same distribution as diffprivlib's reference implementations.

We compare three mechanisms (Laplace, bounded Laplace, analytic Gaussian) on
several (epsilon, delta) settings via a two-sample Kolmogorov-Smirnov test.

Run:
    python scripts/diagnose/verify_dp_mechanisms.py

Output:
    - Console table with KS statistics and p-values per (mechanism, eps, delta).
    - Pass criterion: p > 0.01 for all tests (large sample, so KS catches
      any meaningful distributional difference).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.stats import ks_2samp

from src.dp.mechanisms import apply_dp as apply_dp_slow
from src.dp.mechanisms_fast import apply_dp_fast


def ks_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test, returns (statistic, p-value)."""
    stat, pval = ks_2samp(x, y)
    return float(stat), float(pval)


def main() -> None:
    from configs import DP
    SENSITIVITY = DP.sensitivity
    N_SAMPLES = 100_000  # large enough for KS to detect small differences
    SEED = 42

    test_configs = [
        # (mechanism, eps, delta)
        ("laplace", 0.1, 0.0),
        ("laplace", 1.0, 0.0),
        ("laplace", 10.0, 0.0),
        ("laplace_bounded", 0.5, 0.01),
        ("laplace_bounded", 2.0, 0.05),
        ("gaussian_analytic", 0.5, 0.01),
        ("gaussian_analytic", 2.0, 0.01),
        ("gaussian_analytic", 10.0, 0.05),
    ]

    # Deterministic input: zeros, so output = pure noise samples
    signal = np.zeros(N_SAMPLES, dtype=np.float32)

    print(f"DP Mechanism Verification")
    print(f"  Sensitivity:  {SENSITIVITY}")
    print(f"  N samples:    {N_SAMPLES:,}")
    print(f"  KS criterion: p > 0.01 (large sample => tight)")
    print()
    print(f"{'mechanism':<22} {'eps':>6} {'delta':>7} {'KS stat':>10} {'p-value':>10} {'verdict':>8}")
    print("-" * 70)

    all_pass = True
    for mech, eps, delta in test_configs:
        # Slow reference (diffprivlib per-sample)
        ref = apply_dp_slow(
            signal, mechanism=mech, eps=eps, delta=delta,
            sensitivity=SENSITIVITY, seed=SEED,
        )
        # Fast vectorised
        fast = apply_dp_fast(
            signal, mechanism=mech, eps=eps, delta=delta,
            sensitivity=SENSITIVITY, seed=SEED,
        )
        stat, p = ks_test(ref, fast)
        ok = p > 0.01
        all_pass = all_pass and ok
        verdict = "PASS" if ok else "FAIL"
        print(f"{mech:<22} {eps:>6.2f} {delta:>7.3f} {stat:>10.4f} {p:>10.4f} {verdict:>8}")

    print()
    if all_pass:
        print("All mechanisms verified: fast == diffprivlib distribution-wise.")
    else:
        print("WARNING: Some mechanisms diverge from diffprivlib.")
        sys.exit(1)


if __name__ == "__main__":
    main()
