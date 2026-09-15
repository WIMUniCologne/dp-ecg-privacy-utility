"""Auswertung des Nachtlaufs: F1 Seed-Robustheit, F2 geschuetztes Testset, F3 ResNet."""
import pickle, glob, numpy as np
BAND=[0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.75]
MECH=[("laplace","0.0","Laplace"),("laplace_bounded","1e-05","B.Laplace"),
      ("gaussian_analytic","1e-05","Gaussian")]
def V(pat,key):
    fs=glob.glob(pat); return np.array([pickle.load(open(f,'rb'))[key] for f in fs])
def ci(x):
    if len(x)<2: return (np.nan,np.nan)
    s=x.std(ddof=1)/np.sqrt(len(x)); return x.mean()-1.96*s, x.mean()+1.96*s

print("="*100); print("F1 — BETRIEBSBAND MIT ALLEN SEEDS: Neuanwendung der Auswahlregel"); print("="*100)
print("  relaxed: Utility >= 0.80 UND Leak <= 0.25   |   strikt: >= 0.85 UND <= 0.20\n")
best={}
for m,d,lab in MECH:
    print(f"  {lab}")
    print(f"    {'eps':>6} {'n':>3} {'Utility':>16} {'MIT-BIH':>16} {'ECG-ID':>16}  {'rel':>4}{'str':>4}")
    for e in BAND:
        u=V(f"results/dp/{m}/eps_{e}_delta_{d}/seed_*/metrics.pkl","macro_f1")
        a=V(f"results/reid/dp/{m}/eps_{e}_delta_{d}/adaptive_attacker/seed_*/metrics.pkl","overall_accuracy")
        b=V(f"results/reid_ecgid/dp/{m}/eps_{e}_delta_{d}/adaptive_attacker/seed_*/metrics.pkl","overall_accuracy")
        if not len(u) or not len(b): continue
        ul,uh=ci(u); bl,bh=ci(b)
        rel="JA" if (u.mean()>=0.80 and b.mean()<=0.25) else "-"
        st ="JA" if (u.mean()>=0.85 and b.mean()<=0.20) else "-"
        if rel=="JA": best.setdefault(lab,[]).append((e,u.mean(),b.mean()))
        print(f"    {e:>6} {len(u):>3} {u.mean():>7.3f}[{ul:.2f},{uh:.2f}] "
              f"{a.mean():>7.3f}{'':>8} {b.mean():>7.3f}[{bl:.2f},{bh:.2f}]  {rel:>4}{st:>4}")
    print()
print("  Konfigurationen, die die relaxed-Schranke erfuellen:")
if not best: print("    KEINE — der leere Winkel ist auch relaxed leer.")
for k,v in best.items():
    e,u,b=max(v,key=lambda r:r[1]); print(f"    {k}: eps={e} Utility {u:.3f} Leak {b:.3f}")

print("\n"+"="*100); print("F2 — UTILITY MIT GESCHUETZTEM TESTSET vs. sauberem Testset"); print("="*100)
print(f"  {'mech':<12}{'eps':>6}{'Testset sauber':>16}{'Testset geschuetzt':>20}{'Diff':>9}")
for m,d,lab in MECH:
    for e in [0.15,0.25,0.3,0.5,0.75,1.0]:
        a=V(f"results/dp/{m}/eps_{e}_delta_{d}/seed_*/metrics.pkl","macro_f1")
        p=V(f"results/dp_ptest/{m}/eps_{e}_delta_{d}/seed_*/metrics.pkl","macro_f1")
        if not len(a) or not len(p): continue
        print(f"  {lab:<12}{e:>6}{a.mean():>16.3f}{p.mean():>20.3f}{p.mean()-a.mean():>+9.3f}")

print("\n"+"="*100); print("F3 — ZWEITE ANGREIFER-ARCHITEKTUR (ResNet) vs. generischer CNN"); print("="*100)
for ds,rc,rr in [("MIT-BIH","reid","reid_resnet"),("ECG-ID","reid_ecgid","reid_ecgid_resnet")]:
    print(f"\n  {ds}")
    print(f"    {'mech':<12}{'eps':>6}{'CNN':>9}{'ResNet':>9}{'Diff':>9}")
    for m,d,lab in MECH:
        for e in [0.25,0.3,0.5,0.75]:
            c=V(f"results/{rc}/dp/{m}/eps_{e}_delta_{d}/adaptive_attacker/seed_*/metrics.pkl","overall_accuracy")
            r=V(f"results/{rr}/dp/{m}/eps_{e}_delta_{d}/adaptive_attacker/seed_*/metrics.pkl","overall_accuracy")
            if not len(c) or not len(r): continue
            print(f"    {lab:<12}{e:>6}{c.mean():>9.3f}{r.mean():>9.3f}{r.mean()-c.mean():>+9.3f}")
print("\n  Positiv => ResNet ist staerker => unsere Zahlen sind eine untere Schranke (belegt).")
