"""Auswertung des Option-B-Sweeps (strenge Adjazenz + Clipping)."""
import pickle, glob, re, sys
from pathlib import Path
import numpy as np

D = 0.2902
CS = {"p95": (2.0812, 4.1624), "p99": (4.3178, 8.6356), "max": (13.3111, 26.6222)}
ROOT = Path("results/option_b")

def load(pat):
    out = {}
    for f in glob.glob(pat):
        m = re.search(r"eps_([0-9.e+]+)_delta", f)
        if not m: continue
        try: d = pickle.load(open(f, "rb"))
        except Exception: continue
        out.setdefault(float(m.group(1)), []).append(d)
    return out

print("=" * 78)
print("B0 — CLIPPING ALLEIN (eps=1e9, Rauschen vernachlaessigbar)")
print("=" * 78)
print(f"{'C':<6} {'C-Wert':>8} {'2C':>8} {'R=2C/D':>8} {'macro-F1':>10} {'+/- sd':>8}   per-class N/S/V")
for tag, (C, S) in CS.items():
    r = load(str(ROOT / f"util/C_{tag}_cliponly/laplace/*/seed_*/metrics.pkl"))
    for eps, ds in sorted(r.items()):
        f1 = [d["macro_f1"] for d in ds]
        pc = np.mean([d["per_class_f1"] for d in ds], axis=0)
        print(f"{tag:<6} {C:>8.3f} {S:>8.3f} {S/D:>8.1f} {np.mean(f1):>10.3f} "
              f"{np.std(f1):>8.3f}   {pc[0]:.3f}/{pc[1]:.3f}/{pc[2]:.3f}  (n={len(f1)})")
print("\n  Referenz ohne Privacy und ohne Clipping: macro-F1 = 0.950 (N .994 / S .864 / V .992)")

print("\n" + "=" * 78)
print("B1-B3 — CLIPPING + LAPLACE bei sens = 2C")
print("=" * 78)
for tag, (C, S) in CS.items():
    r = load(str(ROOT / f"util/C_{tag}/laplace/*/seed_*/metrics.pkl"))
    if not r: continue
    print(f"\n  C = {tag} ({C:.3f}), sens = 2C = {S:.3f}, Rauschfaktor {S/D:.1f}x")
    print(f"    {'eps':>7} {'macro-F1':>10} {'+/- sd':>8}   per-class N/S/V")
    for eps, ds in sorted(r.items()):
        f1 = [d["macro_f1"] for d in ds]
        pc = np.mean([d["per_class_f1"] for d in ds], axis=0)
        print(f"    {eps:>7} {np.mean(f1):>10.3f} {np.std(f1):>8.3f}   "
              f"{pc[0]:.3f}/{pc[1]:.3f}/{pc[2]:.3f}  (n={len(f1)})")

print("\n" + "=" * 78)
print("B4 — RE-ID UNTER OPTION B (C=p95, adaptiver Angreifer)")
print("=" * 78)
for ds_name, floor_u, floor_m in [("mitbih", 1/22, 0.0455), ("ecgid", 1/89, 0.0647)]:
    r = load(str(ROOT / f"reid_{ds_name}_C_p95/dp/laplace/*/adaptive_attacker/seed_*/metrics.pkl"))
    if not r: continue
    print(f"\n  {ds_name}:  1/n = {floor_u:.4f}   Majority-Floor = {floor_m:.4f}")
    print(f"    {'eps':>7} {'reid acc':>10} {'+/- sd':>8} {'x 1/n':>8} {'x majority':>11}")
    for eps, dd in sorted(r.items()):
        a = [d["overall_accuracy"] for d in dd]
        m = np.mean(a)
        print(f"    {eps:>7} {m:>10.3f} {np.std(a):>8.3f} {m/floor_u:>8.1f} {m/floor_m:>11.1f}")

print("\n" + "=" * 78)
print("QUERCHECK — Rauschanteil aus dem bestehenden Sweep (ohne Clipping)")
print("=" * 78)
import pandas as pd
lap = pd.read_csv("results/dp_sweep_summary.csv")
lap = lap[lap.mechanism == "laplace"].set_index("eps")["macro_f1_mean"].to_dict()
grid = sorted(lap)
print("  Option B bei eps  ==  bestehende Kalibrierung bei eps/R  (nur Rauschen, kein Clipping)")
print(f"    {'C':<6} {'eps':>7} {'aequiv eps':>11} {'Utility nur-Rauschen':>21}")
for tag, (C, S) in CS.items():
    for e in [1.0, 4.0, 20.0]:
        eq = e / (S / D)
        if eq < min(grid): v = "unter dem Grid"
        else:
            n = min(grid, key=lambda g: abs(g - eq)); v = f"{lap[n]:.3f} (eps={n})"
        print(f"    {tag:<6} {e:>7} {eq:>11.4f} {v:>21}")
print("\n  Differenz zu B1-B3 oben = der reine Clipping-Beitrag.")
