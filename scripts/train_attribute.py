"""
Experiment 1 -- Attribute inference as a SECOND leakage axis.

A record counted as re-identification-protected may still leak demographic
attributes (sex, age band). This trains a second classifier that predicts the
attribute from the DP-protected signal, over exactly the same epsilon sweep,
the same three mechanisms, and the same three seeds as the re-identification
attacker -- so the two leakage curves can be laid side by side.

Architecture: the SAME convolutional identifier used for the re-ID attacker
(``build_reid_model``), only the output layer changes to
the attribute classes. Using the same model avoids opening a new
"why this architecture" question.

CRITICAL -- subject-disjoint evaluation. The attribute is constant across a
subject's windows, so the train/test split is on the SUBJECT level (no subject
in both). Otherwise the model memorises identity and reads the attribute off it,
which is disguised re-identification. This is a different split from the
within-record Re-ID split and is built in ``src/data/attributes.py``.

Baselines reported with every run:
  * majority-class floor (the real class balance on the test set, not 1/n),
  * the no-privacy anchor (attribute accuracy on the UNPROTECTED signal).

Decisions (verify with scripts/diagnose/verify_attribute_labels.py):
  * MIT-BIH: sex only.   * ECG-ID: sex + age (2 bands at the cohort median).

Output:
  results/attr/<dataset>/<attribute>/baseline/seed_<S>.pkl
  results/attr/<dataset>/<attribute>/dp/<mech>/eps_<e>_delta_<d>/<threat>/seed_<S>/metrics.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _peek_gpu_memory_arg() -> "float | None":
    for i, a in enumerate(sys.argv):
        if a == "--gpu-memory-gb" and i + 1 < len(sys.argv):
            try:
                return float(sys.argv[i + 1])
            except ValueError:
                return None
        if a.startswith("--gpu-memory-gb="):
            try:
                return float(a.split("=", 1)[1])
            except ValueError:
                return None
    return None

_gpu_mem = _peek_gpu_memory_arg()
if _gpu_mem is not None:
    from src.utils import limit_gpu_memory
    limit_gpu_memory(memory_gb=_gpu_mem)

import numpy as np
import tensorflow as tf

from configs import DP, REID
from src.data.attributes import (
    build_attribute_split_ecgid,
    build_attribute_split_mitbih,
    majority_class_floor,
)
from src.dp import perturb_encoder_inputs
from src.dp.mechanisms import is_valid_config
from src.models import build_reid_model
from scripts.train_reid import derived_seed, setup_logging


def train_and_eval_attribute(
    X_train, y_train, X_test, y_test, subj_test,
    n_classes, window_samples, epochs, batch_size, learning_rate, seed,
    logger, log_prefix="",
) -> dict:
    """
    Train the attribute classifier and return metrics + per-subject vote.

    Faithful mirror of scripts.train_reid.train_and_eval_reid (same model,
    same optimiser, same best-by-test-accuracy selection) so the attribute
    attacker is identical to the re-ID attacker except for the output head and
    the subject-level aggregation added here.
    """
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = build_reid_model(
        window_samples=window_samples, n_patients=n_classes,
        conv_filters=REID.conv_filters, conv_kernel=REID.conv_kernel,
        pool_size=REID.pool_size, dense_units=REID.dense_units, dropout=REID.dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .cache().shuffle(len(X_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )

    best_acc, best_weights = -1.0, None
    history = []
    for ep in range(1, epochs + 1):
        h = model.fit(train_ds, epochs=1, verbose=0)
        if ep % 5 == 0 or ep == epochs:
            ev = model.evaluate(test_ds, verbose=0)
            test_acc = float(ev[1])
            logger.info(f"{log_prefix}[ep {ep:>3}] "
                        f"train acc={float(h.history['sparse_categorical_accuracy'][0]):.3f}, "
                        f"test acc={test_acc:.3f}")
            history.append({"epoch": ep, "test_acc": test_acc})
            if test_acc > best_acc:
                best_acc = test_acc
                best_weights = [w.copy() for w in model.get_weights()]
    if best_weights is not None:
        model.set_weights(best_weights)

    # Collect per-window softmax probabilities (for both window- and
    # subject-level metrics).
    probs = []
    for Xb, _ in test_ds:
        logits = model(Xb, training=False)
        probs.append(tf.nn.softmax(logits, axis=-1).numpy())
    P = np.concatenate(probs, axis=0)            # (n_test_windows, n_classes)
    y_pred = P.argmax(axis=1)

    window_acc = float((y_pred == y_test).mean())
    per_class_acc = []
    for c in range(n_classes):
        m = (y_test == c)
        per_class_acc.append(float((y_pred[m] == c).mean()) if m.sum() else 0.0)

    # Subject-level metrics (the honest number for a per-subject attribute):
    #   * mean-probability aggregation (headline; averages softmax over a
    #     subject's windows, then argmax) -- usually stronger than a hard vote,
    #   * majority vote (kept for transparency).
    subj_prob_correct, subj_vote_correct, subj_total = 0, 0, 0
    for s in np.unique(subj_test):
        m = (subj_test == s)
        if not m.any():
            continue
        true = y_test[m][0]
        pred_prob = int(P[m].mean(axis=0).argmax())
        pred_vote = int(np.bincount(y_pred[m], minlength=n_classes).argmax())
        subj_prob_correct += int(pred_prob == true)
        subj_vote_correct += int(pred_vote == true)
        subj_total += 1
    subject_acc = subj_prob_correct / subj_total if subj_total else float("nan")
    subject_acc_vote = subj_vote_correct / subj_total if subj_total else float("nan")

    return {
        "n_classes": n_classes,
        "window_accuracy": window_acc,
        "subject_accuracy": subject_acc,             # mean-probability aggregation
        "subject_accuracy_vote": subject_acc_vote,   # hard majority vote
        "n_test_subjects": int(subj_total),
        "per_class_accuracy": per_class_acc,
        "majority_floor": majority_class_floor(y_test, n_classes),
        "history": history,
    }


def balance_classes(X: np.ndarray, y: np.ndarray, seed: int):
    """
    Oversample minority-class windows (with replacement) so every class matches
    the majority count. Prevents the attacker from collapsing onto the majority
    class — a legitimate adversary-side strengthening, applied to TRAIN only.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_max = int(counts.max())
    rng = np.random.default_rng(seed)
    idx_parts = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < n_max:
            extra = rng.choice(idx, size=n_max - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        idx_parts.append(idx)
    out = np.concatenate(idx_parts)
    rng.shuffle(out)
    return X[out], y[out]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=["mitbih", "ecgid"], default="ecgid")
    p.add_argument("--attribute", choices=["sex", "age"], default="sex")
    p.add_argument("--train-frac", type=float, default=0.6,
                   help="Subject-level train fraction (subjects, not windows).")
    p.add_argument("--split-seed", type=int, default=12345,
                   help="Fixed subject-split seed (same across model seeds).")
    p.add_argument("--n-age-bands", type=int, default=2)
    p.add_argument("--epochs", type=int, default=REID.epochs)
    p.add_argument("--batch-size", type=int, default=REID.batch_size)
    p.add_argument("--learning-rate", type=float, default=REID.learning_rate)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--balance-classes", action="store_true",
                   help="Oversample minority-class TRAIN windows to balance the "
                        "classes (stops the attacker collapsing to the majority "
                        "class). Use the no-privacy anchor to decide whether it "
                        "raises the attacker's ceiling, then keep it fixed.")

    p.add_argument("--dp-sweep", action="store_true",
                   help="Run the DP sweep (otherwise: no-privacy baseline only).")
    p.add_argument("--threat-model", choices=["clean_attacker", "adaptive_attacker"],
                   default="adaptive_attacker")
    p.add_argument("--mechanisms", nargs="+", default=list(DP.mechanisms))
    p.add_argument("--epsilons", type=float, nargs="+", default=None)
    p.add_argument("--deltas", type=float, nargs="+", default=None)
    p.add_argument("--sensitivity", type=float, default=None)

    p.add_argument("--out-dir", type=str, default="results/attr")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--gpu-memory-gb", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "mitbih" and args.attribute == "age":
        print("MIT-BIH age was judged too thin/skewed (47 subjects). "
              "Run --attribute sex on MIT-BIH; use ECG-ID for age.")
        return

    out_root = Path(args.out_dir) / args.dataset / args.attribute
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "attr_log.txt")
    logger = logging.getLogger()

    # ---- Load subject-disjoint attribute data ----
    logger.info(f"Loading {args.dataset} attribute={args.attribute} "
                f"(subject-disjoint split, seed={args.split_seed})...")
    if args.dataset == "mitbih":
        train_ds, test_ds = build_attribute_split_mitbih(
            attribute=args.attribute, train_frac=args.train_frac,
            split_seed=args.split_seed, n_age_bands=args.n_age_bands,
        )
    else:
        train_ds, test_ds = build_attribute_split_ecgid(
            attribute=args.attribute, train_frac=args.train_frac,
            split_seed=args.split_seed, n_age_bands=args.n_age_bands,
        )
    n_classes = len(train_ds.class_names)
    window_samples = train_ds.X.shape[1]
    n_train_subj = len(np.unique(train_ds.subject_per_window))
    n_test_subj = len(np.unique(test_ds.subject_per_window))
    logger.info(
        f"classes={train_ds.class_names}  "
        f"train: {len(train_ds.X)} win / {n_train_subj} subj   "
        f"test: {len(test_ds.X)} win / {n_test_subj} subj   "
        f"floor(test)={majority_class_floor(test_ds.y, n_classes):.3f}"
    )

    sensitivity = args.sensitivity if args.sensitivity is not None else DP.sensitivity
    seeds = args.seeds if args.seeds is not None else list(DP.seeds)

    # ---- No-privacy baseline (anchor) per seed ----
    base_dir = out_root / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        bpath = base_dir / f"seed_{seed}.pkl"
        if args.skip_existing and bpath.exists():
            continue
        logger.info(f"BASELINE (no DP) seed={seed}")
        Xb, yb = ((balance_classes(train_ds.X, train_ds.y, seed)
                   if args.balance_classes else (train_ds.X, train_ds.y)))
        res = train_and_eval_attribute(
            Xb, yb, test_ds.X, test_ds.y, test_ds.subject_per_window,
            n_classes, window_samples, args.epochs, args.batch_size,
            args.learning_rate, seed, logger, log_prefix=f"  base[s{seed}] ",
        )
        res.update(dataset=args.dataset, attribute=args.attribute,
                   class_names=train_ds.class_names, seed=seed,
                   balanced=args.balance_classes)
        tmp = bpath.with_suffix(".pkl.tmp")
        pickle.dump(res, open(tmp, "wb")); tmp.replace(bpath)
        logger.info(f"  no-privacy anchor: window={res['window_accuracy']:.3f} "
                    f"subject={res['subject_accuracy']:.3f} "
                    f"(floor={res['majority_floor']:.3f})")

    if not args.dp_sweep:
        return

    # ---- DP sweep ----
    epsilons = args.epsilons if args.epsilons is not None else list(DP.epsilons)
    tmode = args.threat_model

    def deltas_for(mech):
        return list(args.deltas) if args.deltas is not None else [DP.delta_for(mech)]

    total = sum(
        len(seeds)
        for mech in args.mechanisms for eps in epsilons for delta in deltas_for(mech)
        if is_valid_config(mech, eps, delta)
    )
    done = 0
    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for(mech):
                if not is_valid_config(mech, eps, delta):
                    continue
                for seed in seeds:
                    done += 1
                    odir = (out_root / "dp" / mech / f"eps_{eps}_delta_{delta}"
                            / tmode / f"seed_{seed}")
                    odir.mkdir(parents=True, exist_ok=True)
                    mpath = odir / "metrics.pkl"
                    if args.skip_existing and mpath.exists():
                        logger.info(f"[{done}/{total}] SKIP {mpath}")
                        continue
                    logger.info("-" * 70)
                    logger.info(f"[{done}/{total}] {mech} eps={eps} delta={delta} "
                                f"{tmode} seed={seed}")
                    cfg_seed = derived_seed(seed, mech, eps, delta, tmode)
                    try:
                        Xte = perturb_encoder_inputs(
                            test_ds.X, mechanism=mech, eps=eps, delta=delta,
                            sensitivity=sensitivity, seed=cfg_seed)
                        if tmode == "clean_attacker":
                            Xtr = train_ds.X
                        else:
                            Xtr = perturb_encoder_inputs(
                                train_ds.X, mechanism=mech, eps=eps, delta=delta,
                                sensitivity=sensitivity, seed=cfg_seed + 1)
                        ytr = train_ds.y
                        if args.balance_classes:
                            Xtr, ytr = balance_classes(Xtr, ytr, cfg_seed)
                        res = train_and_eval_attribute(
                            Xtr, ytr, Xte, test_ds.y, test_ds.subject_per_window,
                            n_classes, window_samples, args.epochs, args.batch_size,
                            args.learning_rate, cfg_seed, logger,
                            log_prefix=f"  {args.attribute[:3]}[s{seed}] ",
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"  FAILED: {type(e).__name__}: {e}")
                        continue
                    res.update(dataset=args.dataset, attribute=args.attribute,
                               class_names=train_ds.class_names, mechanism=mech,
                               eps=eps, delta=delta, sensitivity=sensitivity,
                               threat_model=tmode, seed=seed,
                               balanced=args.balance_classes)
                    tmp = mpath.with_suffix(".pkl.tmp")
                    pickle.dump(res, open(tmp, "wb")); tmp.replace(mpath)
                    logger.info(f"  Saved: window={res['window_accuracy']:.3f} "
                                f"subject={res['subject_accuracy']:.3f} "
                                f"(floor={res['majority_floor']:.3f})")

    logger.info("Attribute-inference sweep complete")


if __name__ == "__main__":
    main()
