"""
Diagnose the per-epoch training trajectory for one or more baseline runs.

Usage:
    python scripts/diagnose/diagnose_training_trajectory.py results/baseline/seed_*/metrics.pkl

Shows for each pickled result:
  - macro F1 over time
  - S-class F1 over time (the most volatile class)
  - When the "best" epoch was hit
  - Whether F1 was still improving or had plateaued
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path


def diagnose_one(path: Path) -> None:
    with open(path, "rb") as f:
        r = pickle.load(f)

    hist = r.get("history", [])
    if not hist:
        print(f"  No history in {path}")
        return

    macro = [h["macro_f1"] for h in hist]
    s_f1 = [h["per_class_f1"][1] for h in hist]  # S is index 1
    epochs = [h["epoch"] for h in hist]

    final_macro = r["macro_f1"]
    final_per_class = r["per_class_f1"]

    best_idx = max(range(len(macro)), key=lambda i: macro[i])
    best_ep = epochs[best_idx]
    best_macro = macro[best_idx]

    last_idx = len(macro) - 1
    last_macro = macro[last_idx]

    # Did F1 drop after the peak? (best weights select would still recover)
    post_peak_drop = best_macro - last_macro

    # Plateau check: macro F1 in last 5 evals
    if len(macro) >= 5:
        recent = macro[-5:]
        plateau_range = max(recent) - min(recent)
    else:
        plateau_range = float('nan')

    # Was S ever decent?
    s_max = max(s_f1)
    s_max_ep = epochs[s_f1.index(s_max)]

    print(f"\n== {path} ==")
    print(f"  Final (best weights): macro={final_macro:.4f}, per-class={[f'{x:.3f}' for x in final_per_class]}")
    print(f"  Best macro F1 epoch:  ep={best_ep}, macro={best_macro:.4f}")
    print(f"  Final epoch (no best-weights):  ep={epochs[-1]}, macro={last_macro:.4f}")
    print(f"  Post-peak drop:       {post_peak_drop:+.4f}")
    print(f"  Recent plateau range: {plateau_range:.4f}")
    print(f"  S-F1 best (any ep):   {s_max:.4f} at ep={s_max_ep}")
    print(f"  S-F1 trajectory:      {[f'{x:.2f}' for x in s_f1]}")
    print(f"  Macro F1 trajectory:  {[f'{x:.2f}' for x in macro]}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python diagnose_training_trajectory.py <metrics.pkl> [more...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        diagnose_one(Path(p))


if __name__ == "__main__":
    main()
