"""Rebuild results/attacker_band_summary.csv from the raw per-run metrics.

The notebook that first produced this table carries notebook state; this script
reads the same artefacts directly so the table can be regenerated after new runs
land. Schema and column meanings are unchanged.
"""
from __future__ import annotations

import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MECHS = [("laplace", "0.0"), ("laplace_bounded", "1e-05"), ("gaussian_analytic", "1e-05")]
CORPORA = [("mitbih", "results/reid"), ("ecgid", "results/reid_ecgid")]


def _read(pattern: str, key: str) -> np.ndarray:
    out = []
    for f in glob.glob(str(ROOT / pattern)):
        try:
            m = pickle.load(open(f, "rb"))
        except Exception:
            continue
        if key in m:
            out.append(float(m[key]))
    return np.array(out)


def _eps_values(mech: str, delta: str) -> list[str]:
    seen = set()
    for d in glob.glob(str(ROOT / f"results/dp/{mech}/eps_*_delta_{delta}")):
        seen.add(Path(d).name.split("_")[1])
    return sorted(seen, key=float)


rows = []
for db, reid_dir in CORPORA:
    for mech, delta in MECHS:
        for eps in _eps_values(mech, delta):
            util = _read(f"results/dp/{mech}/eps_{eps}_delta_{delta}/seed_*/metrics.pkl", "macro_f1")
            naive = _read(f"{reid_dir}/dp/{mech}/eps_{eps}_delta_{delta}/clean_attacker/seed_*/metrics.pkl",
                          "overall_accuracy")
            adapt = _read(f"{reid_dir}/dp/{mech}/eps_{eps}_delta_{delta}/adaptive_attacker/seed_*/metrics.pkl",
                          "overall_accuracy")
            if not len(util) or not len(adapt):
                continue
            rows.append(dict(
                database=db, mechanism=mech, eps=float(eps), delta=float(delta),
                # Seed counts, so that a point resting on three seeds is not
                # silently averaged next to one resting on eight.
                util_n=len(util), naive_n=len(naive), adapt_n=len(adapt),
                util_mean=util.mean(), util_std=util.std(ddof=1) if len(util) > 1 else 0.0,
                naive_mean=naive.mean() if len(naive) else np.nan,
                naive_std=naive.std(ddof=1) if len(naive) > 1 else 0.0,
                adapt_mean=adapt.mean(), adapt_std=adapt.std(ddof=1) if len(adapt) > 1 else 0.0,
                gap=(adapt.mean() - naive.mean()) if len(naive) else np.nan,
            ))

df = pd.DataFrame(rows).sort_values(["database", "mechanism", "eps"]).reset_index(drop=True)
out = ROOT / "results/attacker_band_summary.csv"
df.to_csv(out, index=False)
print(f"{len(df)} rows -> {out}")
