"""
Train the Re-Identification CNN attacker on MIT-BIH.

Two evaluation modes ('threat models'):

  * 'clean_attacker': attacker is trained on CLEAN data, evaluated on
                      DP-perturbed data. Asks: 'How well does DP protect
                      against an attacker that didn't anticipate DP?'

  * 'adaptive_attacker': attacker is trained on DP-PERTURBED data with the
                         SAME DP config that will be evaluated. Asks: 'How
                         well does DP protect against a worst-case adaptive
                         attacker?'

For the baseline (no DP), both modes are equivalent.

Usage:
    # Baseline only (no DP)
    python scripts/train_reid_baseline.py

    # Sweep with adaptive attacker (default, most conservative)
    python scripts/train_reid_baseline.py --dp-sweep

    # With specific epsilons
    python scripts/train_reid_baseline.py --dp-sweep --epsilons 0.5 1.0 2.0

Output:
    results/reid/baseline/metrics.pkl
    results/reid/dp/<mechanism>/eps_<e>_delta_<d>/<threat_model>/metrics.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- GPU memory limit BEFORE TF import ----
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
from src.data import (
    build_ecgid_split,
    build_reid_all_mitbih, build_reid_ds1,
)
from src.dp import perturb_encoder_inputs
from src.dp.mechanisms import is_valid_config
from src.models import build_reid_model, build_reid_resnet


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
        force=True,
    )


def derived_seed(base: int, mech: str, eps: float, delta: float, mode: str) -> int:
    """
    Per-config seed. Uses hashlib instead of Python's hash() because the
    latter is process-randomized for strings, breaking reproducibility.
    """
    import hashlib
    payload = f"{base}|{mech}|{eps}|{delta}|{mode}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def train_and_eval_reid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_patients: int,
    window_samples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    logger: logging.Logger,
    log_prefix: str = "",
    arch: str = "cnn",
) -> dict:
    """Train a Re-ID model and return metrics."""
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if arch == "resnet":
        model = build_reid_resnet(
            window_samples=window_samples, n_patients=n_patients,
        )
    else:
        model = build_reid_model(
            window_samples=window_samples,
            n_patients=n_patients,
            conv_filters=REID.conv_filters,
            conv_kernel=REID.conv_kernel,
            pool_size=REID.pool_size,
            dense_units=REID.dense_units,
            dropout=REID.dropout,
        )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .cache()
        .shuffle(len(X_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    best_acc = -1.0
    best_weights = None
    history = []

    for ep in range(1, epochs + 1):
        h = model.fit(train_ds, epochs=1, verbose=0)
        train_loss = float(h.history["loss"][0])
        train_acc = float(h.history["sparse_categorical_accuracy"][0])

        # Eval every 5 epochs
        if ep % 5 == 0 or ep == epochs:
            ev = model.evaluate(test_ds, verbose=0)
            test_loss, test_acc = float(ev[0]), float(ev[1])
            logger.info(
                f"{log_prefix}[ep {ep:>3}] train acc={train_acc:.4f}, "
                f"test acc={test_acc:.4f}"
            )
            history.append({
                "epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc,
            })
            if test_acc > best_acc:
                best_acc = test_acc
                best_weights = [w.copy() for w in model.get_weights()]

    if best_weights is not None:
        model.set_weights(best_weights)

    # Final per-patient accuracy
    preds_all = []
    for X_batch, _ in test_ds:
        logits = model(X_batch, training=False)
        preds_all.append(tf.argmax(logits, axis=-1).numpy())
    y_pred = np.concatenate(preds_all)

    overall_acc = float((y_pred == y_test).mean())
    # Per-patient sensitivity (recall)
    per_patient_acc = []
    for pid in range(n_patients):
        mask = (y_test == pid)
        if mask.sum() > 0:
            per_patient_acc.append(float((y_pred[mask] == pid).mean()))
        else:
            per_patient_acc.append(0.0)

    # Top-1 random baseline = 1/n_patients
    random_baseline = 1.0 / n_patients

    return {
        "n_patients": n_patients,
        "overall_accuracy": overall_acc,
        "random_baseline": random_baseline,
        "per_patient_accuracy": per_patient_acc,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=["mitbih", "ecgid"], default="mitbih",
                   help="Which database to use for Re-ID.")
    p.add_argument("--data-split", choices=["ds1", "all"], default="ds1",
                   help="MIT-BIH only: ds1 = 22 patients, all = 44 patients.")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="MIT-BIH only: temporal fraction per record used for training.")
    p.add_argument("--epochs", type=int, default=REID.epochs)
    p.add_argument("--batch-size", type=int, default=REID.batch_size)
    p.add_argument("--learning-rate", type=float, default=REID.learning_rate)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="One run per seed (default: configs.DP.seeds)")

    # DP sweep
    p.add_argument("--dp-sweep", action="store_true",
                   help="Also run DP sweep (otherwise: baseline only)")
    p.add_argument("--threat-model", choices=["clean_attacker", "adaptive_attacker", "both"],
                   default="adaptive_attacker",
                   help="Which threat model(s) to evaluate")
    p.add_argument("--mechanisms", nargs="+", default=list(DP.mechanisms))
    p.add_argument("--epsilons", type=float, nargs="+", default=None)
    p.add_argument("--deltas", type=float, nargs="+", default=None,
                   help="If omitted, uses DP.delta_for(mech) per mechanism.")
    p.add_argument("--sensitivity", type=float, default=None)
    p.add_argument("--arch", choices=["cnn", "resnet"], default="cnn",
                   help="Attacker architecture. 'cnn' is the generic primary "
                        "attacker; 'resnet' is the stronger second architecture.")
    p.add_argument("--clip", type=float, default=None,
                   help="Clamp signal values to [-CLIP, +CLIP] before adding noise "
                        "(unbounded-adjacency variant; pass --sensitivity 2*CLIP).")

    p.add_argument("--out-dir", type=str, default="results/reid")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--gpu-memory-gb", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "reid_log.txt")
    logger = logging.getLogger()

    # ---- Load Re-ID data (once) ----
    if args.dataset == "mitbih":
        logger.info(f"Loading MIT-BIH for Re-ID (split={args.data_split})...")
        if args.data_split == "ds1":
            train_ds, test_ds = build_reid_ds1(train_frac=args.train_frac)
        else:
            train_ds, test_ds = build_reid_all_mitbih(train_frac=args.train_frac)
        n_subjects = len(train_ds.patient_ids)
    else:  # ecgid
        logger.info("Loading ECG-ID for Re-ID (session-based split)...")
        train_ds, test_ds = build_ecgid_split()
        n_subjects = len(train_ds.subject_ids)

    window_samples = train_ds.X.shape[1]
    logger.info(
        f"Loaded: {len(train_ds.X)} train windows, {len(test_ds.X)} test windows, "
        f"{n_subjects} subjects, {window_samples} samples per window"
    )

    sensitivity = args.sensitivity if args.sensitivity is not None else DP.sensitivity
    seeds = args.seeds if args.seeds is not None else list(DP.seeds)

    # ---- Baseline (no DP), per seed ----
    baseline_dir = out_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        baseline_path = baseline_dir / f"seed_{seed}.pkl"
        if args.skip_existing and baseline_path.exists():
            logger.info(f"Baseline seed={seed} exists, skipping")
            continue
        logger.info("=" * 70)
        logger.info(f"BASELINE: clean training, clean evaluation (seed={seed})")
        logger.info("=" * 70)
        baseline_result = train_and_eval_reid(
            train_ds.X, train_ds.y, test_ds.X, test_ds.y,
            n_patients=n_subjects, window_samples=window_samples,
            arch=args.arch,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=seed,
            logger=logger, log_prefix=f"  baseline[s{seed}] ",
        )
        baseline_result["seed"] = seed
        tmp_path = baseline_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(baseline_result, f)
        tmp_path.replace(baseline_path)
        logger.info(
            f"Baseline seed={seed}: accuracy = {baseline_result['overall_accuracy']:.4f} "
            f"(random = {baseline_result['random_baseline']:.4f})"
        )

    if not args.dp_sweep:
        return

    # ---- DP sweep ----
    epsilons = args.epsilons if args.epsilons is not None else list(DP.epsilons)
    deltas_override = args.deltas
    threat_modes = (["clean_attacker", "adaptive_attacker"]
                    if args.threat_model == "both" else [args.threat_model])

    def deltas_for_mech(mech):
        if deltas_override is not None:
            return list(deltas_override)
        return [DP.delta_for(mech)]

    # Compute total
    total = 0
    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for_mech(mech):
                if is_valid_config(mech, eps, delta):
                    total += len(threat_modes) * len(seeds)
    done = 0

    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for_mech(mech):
                if not is_valid_config(mech, eps, delta):
                    logger.info(
                        f"SKIP invalid: {mech} eps={eps} delta={delta}"
                    )
                    continue

                for tmode in threat_modes:
                    for seed in seeds:
                        done += 1

                        out_dir = (
                            out_root / "dp" / mech
                            / f"eps_{eps}_delta_{delta}" / tmode
                            / f"seed_{seed}"
                        )
                        out_dir.mkdir(parents=True, exist_ok=True)
                        metrics_path = out_dir / "metrics.pkl"

                        if args.skip_existing and metrics_path.exists():
                            logger.info(f"[{done}/{total}] SKIP existing: {metrics_path}")
                            continue

                        logger.info("-" * 70)
                        logger.info(
                            f"[{done}/{total}] {mech} eps={eps} delta={delta} "
                            f"mode={tmode} seed={seed}"
                        )

                        cfg_seed = derived_seed(seed, mech, eps, delta, tmode)

                        t0 = time.time()
                        try:
                            # Test set ALWAYS DP-perturbed
                            X_test_dp = perturb_encoder_inputs(
                                test_ds.X, mechanism=mech, eps=eps, delta=delta,
                                sensitivity=sensitivity, seed=cfg_seed,
                                clip=args.clip,
                            )

                            if tmode == "clean_attacker":
                                X_train_used = train_ds.X
                            else:  # adaptive_attacker
                                X_train_used = perturb_encoder_inputs(
                                    train_ds.X, mechanism=mech, eps=eps, delta=delta,
                                    sensitivity=sensitivity, seed=cfg_seed + 1,
                                    clip=args.clip,
                                )

                            result = train_and_eval_reid(
                                X_train_used, train_ds.y, X_test_dp, test_ds.y,
                                n_patients=n_subjects, window_samples=window_samples,
                                arch=args.arch,
                                epochs=args.epochs, batch_size=args.batch_size,
                                learning_rate=args.learning_rate, seed=cfg_seed,
                                logger=logger, log_prefix=f"  {tmode[:8]}[s{seed}] ",
                            )
                        except Exception as e:
                            logger.error(f"  FAILED: {type(e).__name__}: {e}")
                            continue

                        result["mechanism"] = mech
                        result["eps"] = eps
                        result["delta"] = delta
                        result["sensitivity"] = sensitivity
                        result["clip"] = args.clip
                        result["arch"] = args.arch
                        result["threat_model"] = tmode
                        result["seed"] = seed

                        tmp_path = metrics_path.with_suffix(".pkl.tmp")
                        with open(tmp_path, "wb") as f:
                            pickle.dump(result, f)
                        tmp_path.replace(metrics_path)
                        logger.info(
                            f"  Saved: accuracy={result['overall_accuracy']:.4f} "
                            f"(took {time.time()-t0:.1f}s)"
                        )

    logger.info("=" * 70)
    logger.info("Re-ID sweep complete")


if __name__ == "__main__":
    main()
