"""
Experiment 2 -- Second utility classifier (feature-based Random Forest).

Robustness check: is the L-shaped utility curve a property of the one deep
architecture (Mousavi & Afghah seq2seq), or does a generic, architecturally
distant model show the same smooth decline and the same ceiling? This trains a
classical Random Forest on hand-crafted beat-morphology features over the SAME
epsilon sweep, mechanisms, seeds, and data.

Consistency with the main model -- everything except the architecture is
identical:
  * same MIT-BIH inter-patient DS1/DS2 split (de Chazal), subject-disjoint,
  * same qsPeaks beats, same N/S/V classes, same macro-F1 metric,
  * same DP applied to the same z-normalized training beats (test set clean),
  * same SMOTE oversampling (S->7000, V->6000) on the TRAIN beats only, with the
    per-config seed,
  * features computed from the SAME DP-protected beat (see src/models/ecg_features.py).

No tuning: a fixed, reasonable Random Forest. The point is that a generic second
model reproduces the shape, not that it is optimised -- visible tuning would
invite the suspicion that the L-shape was engineered.

Output (mirrors the seq2seq utility layout, separate root so nothing collides):
  results/utility_rf/baseline/seed_<S>/metrics.pkl
  results/utility_rf/<mech>/eps_<e>_delta_<d>/seed_<S>/metrics.pkl
with macro_f1 + per_class_f1, directly comparable to results/dp/.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from configs import DP
from src.dp import perturb_encoder_inputs
from src.dp.mechanisms import is_valid_config
from src.models import (
    apply_smote, build_vocab, confusion_matrix_per_class,
    labels_to_targets, metrics_from_cm, read_mitbih_wfdb,
)
from src.models.ecg_features import extract_beat_features
from scripts.train_with_dp import derived_seed, setup_logging

CLASSES = ("N", "S", "V")
SMOTE_TARGETS = {"N": None, "S": 7000, "V": 6000}

# Fixed, untuned Random Forest configuration.
RF_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_leaf=1,
                 max_features="sqrt", n_jobs=-1)


def _flatten(seq_X, seq_y):
    """(n_seqs, T, L), (n_seqs, T) -> (n_seqs*T, L), (n_seqs*T,)."""
    n, T, L = seq_X.shape
    return seq_X.reshape(n * T, L), seq_y.reshape(n * T)


def train_eval_rf(X_train_beats, y_train, X_test_feats, y_test, n_classes, seed,
                  logger, log_prefix="") -> dict:
    """Extract features, fit RF, return macro-F1 metrics (N/S/V)."""
    t0 = time.time()
    F_train = extract_beat_features(X_train_beats)
    clf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    clf.fit(F_train, y_train)
    y_pred = clf.predict(X_test_feats)

    cm = confusion_matrix_per_class(y_test.astype(np.int64),
                                    y_pred.astype(np.int64), n_classes)
    sens, ppv, f1 = metrics_from_cm(cm)
    logger.info(f"{log_prefix}macro_F1={float(f1.mean()):.4f} "
                f"per-class={[f'{x:.3f}' for x in f1]} ({time.time()-t0:.1f}s)")
    return {
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "sensitivity_per_class": sens.tolist(),
        "precision": ppv.tolist(),
        "confusion": cm.tolist(),
        "classes": list(CLASSES),
        "model": "random_forest",
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mechanisms", nargs="+", default=list(DP.mechanisms))
    p.add_argument("--epsilons", type=float, nargs="+", default=None)
    p.add_argument("--deltas", type=float, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--sensitivity", type=float, default=None)
    p.add_argument("--beat-extraction", choices=["qspeaks", "rr_resample"],
                   default="qspeaks")
    p.add_argument("--out-dir", type=str, default="results/utility_rf")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "rf_log.txt")
    logger = logging.getLogger()

    sensitivity = args.sensitivity if args.sensitivity is not None else DP.sensitivity
    seeds = args.seeds if args.seeds is not None else list(DP.seeds)
    epsilons = args.epsilons if args.epsilons is not None else list(DP.epsilons)
    n_classes = len(CLASSES)
    char2num, _ = build_vocab(CLASSES)

    logger.info("Loading MIT-BIH DS1/DS2 (same inter-patient split as seq2seq)...")
    Xtr_seq, ytr_chars = read_mitbih_wfdb(trainset=1, classes=CLASSES,
                                          beat_extraction=args.beat_extraction,
                                          seed=seeds[0])
    Xte_seq, yte_chars = read_mitbih_wfdb(trainset=0, classes=CLASSES,
                                          beat_extraction=args.beat_extraction,
                                          seed=seeds[0])
    ytr = labels_to_targets(ytr_chars, char2num)
    yte = labels_to_targets(yte_chars, char2num)

    # Clean test features computed once (test set is never perturbed).
    Xte_beats, yte_beats = _flatten(Xte_seq, yte)
    Fte = extract_beat_features(Xte_beats)
    logger.info(f"Test beats: {len(Xte_beats)}  features: {Fte.shape[1]}")

    # ---- No-DP baseline per seed ----
    for seed in seeds:
        bdir = out_root / "baseline" / f"seed_{seed}"
        bdir.mkdir(parents=True, exist_ok=True)
        bpath = bdir / "metrics.pkl"
        if args.skip_existing and bpath.exists():
            continue
        # SMOTE on clean train beats (same targets/seed mechanism as seq2seq).
        Xs, ys = apply_smote(Xtr_seq, ytr, char2num, CLASSES, SMOTE_TARGETS,
                             smote_seed=seed)
        Xb, yb = _flatten(Xs, ys)
        res = train_eval_rf(Xb, yb, Fte, yte_beats, n_classes, seed, logger,
                            log_prefix=f"  base[s{seed}] ")
        res["seed"] = seed
        tmp = bpath.with_suffix(".pkl.tmp"); pickle.dump(res, open(tmp, "wb")); tmp.replace(bpath)
        logger.info(f"Baseline seed={seed}: macro_F1={res['macro_f1']:.4f}")

    # ---- DP sweep ----
    def deltas_for(m):
        return list(args.deltas) if args.deltas is not None else [DP.delta_for(m)]

    total = sum(len(seeds) for m in args.mechanisms for e in epsilons
                for d in deltas_for(m) if is_valid_config(m, e, d))
    done = 0
    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for(mech):
                if not is_valid_config(mech, eps, delta):
                    continue
                for seed in seeds:
                    done += 1
                    rdir = out_root / mech / f"eps_{eps}_delta_{delta}" / f"seed_{seed}"
                    rdir.mkdir(parents=True, exist_ok=True)
                    mpath = rdir / "metrics.pkl"
                    if args.skip_existing and mpath.exists():
                        logger.info(f"[{done}/{total}] SKIP {mpath}")
                        continue
                    logger.info(f"[{done}/{total}] {mech} eps={eps} delta={delta} seed={seed}")
                    cfg_seed = derived_seed(seed, mech, eps, delta)
                    try:
                        # DP on the same z-normalized train beats (encoder inputs).
                        Xtr_dp = perturb_encoder_inputs(
                            Xtr_seq, mechanism=mech, eps=eps, delta=delta,
                            sensitivity=sensitivity, seed=cfg_seed)
                        Xs, ys = apply_smote(Xtr_dp, ytr, char2num, CLASSES,
                                             SMOTE_TARGETS, smote_seed=cfg_seed)
                        Xb, yb = _flatten(Xs, ys)
                        res = train_eval_rf(Xb, yb, Fte, yte_beats, n_classes,
                                            cfg_seed, logger,
                                            log_prefix=f"  rf[s{seed}] ")
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"  FAILED: {type(e).__name__}: {e}")
                        continue
                    res.update(mechanism=mech, eps=eps, delta=delta,
                               sensitivity=sensitivity, seed=seed)
                    tmp = mpath.with_suffix(".pkl.tmp"); pickle.dump(res, open(tmp, "wb")); tmp.replace(mpath)
                    logger.info(f"  Saved: macro_F1={res['macro_f1']:.4f}")

    logger.info("Second-utility (RF) sweep complete")


if __name__ == "__main__":
    main()
