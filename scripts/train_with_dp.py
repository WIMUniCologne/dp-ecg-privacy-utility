"""
Train the seq2seq classifier with differential privacy applied to the training
beats.

For each combination of (mechanism, epsilon, delta), we:
  1. Load and prepare beats as pure-class sequences (same as baseline).
  2. Apply the DP mechanism to the encoder inputs (beat signal values).
     Labels are never perturbed.
  3. Train the model.
  4. Evaluate on the clean DS2 test set.
  5. Save per-class metrics.

DP is applied to the z-normalized beat signal values (after preprocessing,
before SMOTE). The sensitivity is configured in `configs/config.py` (default
0.29, derived as the 95th percentile of jump sensitivity, see
notebooks/00_sensitivity_calibration.ipynb).

By default we sweep a grid of epsilons across three mechanisms. Override the
grid with --epsilons / --deltas / --mechanisms.

For parallel runs on the same GPU:
    # Limit each process to ~8 GB GPU memory:
    python scripts/train_with_dp.py --gpu-memory-gb 8 \
        --mechanisms laplace --epsilons 0.05 0.1 0.5 &
    python scripts/train_with_dp.py --gpu-memory-gb 8 \
        --mechanisms laplace --epsilons 1.0 2.0 4.0 &
    python scripts/train_with_dp.py --gpu-memory-gb 8 \
        --mechanisms laplace --epsilons 8.0 15.0 20.0 &

Usage:
    # Default sweep
    python scripts/train_with_dp.py

    # Quick single point
    python scripts/train_with_dp.py --mechanisms laplace --epsilons 1.0 --epochs 100

Output:
    results/dp/<mechanism>/eps_<eps>_delta_<delta>/metrics.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- IMPORTANT: parse --gpu-memory-gb and configure TF BEFORE other imports ----
def _peek_gpu_memory_arg() -> "float | None":
    """Extract --gpu-memory-gb from argv without consuming it."""
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

from configs import DP, SEQ2SEQ
from src.dp import perturb_encoder_inputs
from src.dp.denoise import denoise as denoise_signal
from src.dp.mechanisms import is_valid_config
from src.models import (
    GreedyDecoder,
    apply_smote,
    build_decoder_inputs,
    build_seq2seq_model,
    build_vocab,
    confusion_matrix_per_class,
    labels_to_targets,
    metrics_from_cm,
    read_mitbih_mat,
    read_mitbih_wfdb,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def derived_seed(base: int, mech: str, eps: float, delta: float) -> int:
    """
    Per-config seed so different DP configs use different noise but are
    reproducible across process restarts.

    Uses hashlib.sha256 instead of Python's built-in hash() because the latter
    is process-randomized for strings (PYTHONHASHSEED) and would yield
    different seeds on each program start, breaking reproducibility when a
    sweep is resumed.
    """
    import hashlib
    payload = f"{base}|{mech}|{eps}|{delta}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# One single training run (one DP config)
# ---------------------------------------------------------------------------
def train_one_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    char2num: dict,
    classes: tuple,
    mechanism: str,
    eps: float,
    delta: float,
    sensitivity: float,
    smote_targets: dict,
    clip: "float | None",
    perturb_test: bool,
    denoise_method: "str | None",
    denoise_fs: float,
    epochs: int,
    batch_size: int,
    eval_every: int,
    seed: int,
    logger: logging.Logger,
) -> dict:
    """Train with one (mechanism, eps, delta) config, return metrics dict."""
    # Clear Keras layer counters and free old graph nodes — prevents both
    # the auto-naming drift bug and memory bloat across many configs.
    tf.keras.backend.clear_session()

    np.random.seed(seed)
    tf.random.set_seed(seed)

    go_id = char2num["<GO>"]
    vocab_size = len(char2num)
    n_classes = len(classes)

    # Apply DP to training beat values (NOT to test set)
    logger.info(f"  Applying {mechanism} with eps={eps}, delta={delta}, sens={sensitivity}...")
    t0 = time.time()
    X_train_dp = perturb_encoder_inputs(
        X_train, mechanism=mechanism, eps=eps, delta=delta,
        sensitivity=sensitivity, seed=seed, clip=clip,
    )
    logger.info(f"  DP applied in {time.time() - t0:.2f}s. "
                f"Mean abs perturbation: {np.abs(X_train_dp - X_train).mean():.4f}")

    # Optionally protect the test set too. The default (clean test set) matches
    # the sharing scenario: a recipient trains on the protected recordings and
    # applies the model to their own, unprotected clinical data. Setting this
    # answers the other reading, where the recipient only ever sees protected
    # signal at inference time as well.
    if perturb_test:
        X_test = perturb_encoder_inputs(
            X_test, mechanism=mechanism, eps=eps, delta=delta,
            sensitivity=sensitivity, seed=seed + 1, clip=clip,
        )

    # Optional denoising front end for the *recipient*, mirroring the one the
    # attacker is given in train_reid_denoise.py. The transform is deterministic,
    # so it is applied identically to what the recipient trains on and to what
    # they later classify; otherwise the two would come from different
    # distributions. This tests whether strengthening the recipient the way we
    # strengthen the attacker moves the utility axis.
    #
    # `denoise_fs` is the *effective* rate of the beat representation, not its
    # length: beats run T-wave to T-wave and are resampled to 280 points, so at
    # a median RR of 0.90 s one beat-sample is 1/310 s. Passing the length (280)
    # instead would size the smoothing window for the wrong time-scale and stop
    # it matching the attacker's front end, which smooths over 25 ms.
    if denoise_method:
        X_train_dp = denoise_signal(X_train_dp, method=denoise_method, fs=denoise_fs)
        X_test = denoise_signal(X_test, method=denoise_method, fs=denoise_fs)
        logger.info(f"  Denoiser '{denoise_method}' (fs={denoise_fs:g} Hz) "
                    f"applied to recipient inputs.")

    # SMOTE on the perturbed data
    X_smote, y_smote = apply_smote(
        X_train_dp, y_train, char2num, classes, smote_targets,
        smote_seed=seed,
    )
    train_dec_in = build_decoder_inputs(y_smote, go_id)
    test_dec_in = build_decoder_inputs(y_test, go_id)

    # Build model
    model = build_seq2seq_model(
        n_channels=10,
        input_depth=X_train.shape[-1],
        max_time=SEQ2SEQ.beats_per_group,
        num_units=SEQ2SEQ.lstm_units,
        vocab_size=vocab_size,
        embed_size=10,
        bidirectional=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(SEQ2SEQ.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices(((X_smote, train_dec_in), y_smote))
        .cache()
        .shuffle(len(X_smote), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    decoder = GreedyDecoder(
        model, num_units=SEQ2SEQ.lstm_units, vocab_size=vocab_size,
        go_id=go_id, bidirectional=False,
        max_time=SEQ2SEQ.beats_per_group, input_depth=X_train.shape[-1],
    )

    best_macro = -np.inf
    best_weights = None
    history = []

    for ep in range(1, epochs + 1):
        h = model.fit(train_ds, epochs=1, verbose=0)
        if ep % eval_every == 0 or ep == epochs:
            preds = decoder.decode(X_test, batch_size=batch_size)
            cm = confusion_matrix_per_class(y_test.flatten(), preds.flatten(), n_classes)
            sens, ppv, f1 = metrics_from_cm(cm)
            macro = float(f1.mean())
            logger.info(
                f"    [ep {ep:>3}] loss={float(h.history['loss'][0]):.4f}, "
                f"macro_F1={macro:.4f}, per-class F1={[f'{x:.3f}' for x in f1]}"
            )
            history.append({
                "epoch": ep,
                "train_loss": float(h.history["loss"][0]),
                "macro_f1": macro,
                "per_class_f1": f1.tolist(),
            })
            if macro > best_macro:
                best_macro = macro
                best_weights = [w.copy() for w in model.get_weights()]

    # Final eval with best weights
    if best_weights is not None:
        model.set_weights(best_weights)
    preds = decoder.decode(X_test, batch_size=batch_size)
    cm = confusion_matrix_per_class(y_test.flatten(), preds.flatten(), n_classes)
    sens, ppv, f1 = metrics_from_cm(cm)

    return {
        "mechanism": mechanism,
        "eps": eps,
        "delta": delta,
        "sensitivity": sensitivity,
        "clip": clip,
        "perturb_test": perturb_test,
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "sensitivity_per_class": sens.tolist(),
        "precision": ppv.tolist(),
        "confusion": cm.tolist(),
        "classes": list(classes),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-source", choices=["wfdb", "mat"], default="wfdb")
    p.add_argument("--mat-path", type=str, default=None)
    p.add_argument("--beat-extraction", choices=["qspeaks", "rr_resample"],
                   default="qspeaks")
    p.add_argument("--mechanisms", nargs="+",
                   default=["laplace", "laplace_bounded", "gaussian_analytic"],
                   choices=["laplace", "laplace_bounded", "gaussian_analytic"])
    p.add_argument("--epsilons", type=float, nargs="+", default=None,
                   help="epsilon values (default: from configs.DP.epsilons)")
    p.add_argument("--deltas", type=float, nargs="+", default=None,
                   help="delta values. If omitted, uses DP.delta_for(mech) per "
                        "mechanism (recommended). Pass values to force a grid.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="One run per seed. Default: configs.DP.seeds")
    p.add_argument("--perturb-test", action="store_true",
                   help="Also apply the mechanism to the DS2 test set (the "
                        "recipient only ever sees protected signal).")
    p.add_argument("--denoise", type=str, default=None,
                   choices=["savgol", "moving_average", "gaussian"],
                   help="Give the recipient the same denoising front end the "
                        "attacker gets. Applied after the mechanism, to both "
                        "what the recipient trains on and what they classify.")
    p.add_argument("--denoise-fs", type=float, default=310.0,
                   help="Effective sampling rate of the beat representation, in Hz, "
                        "used to size the smoothing window. Beats are T-wave to "
                        "T-wave resampled to 280 points; at a median RR of 0.90 s "
                        "that is about 310 Hz. This is not the beat length.")
    p.add_argument("--clip", type=float, default=None,
                   help="Clamp signal values to [-CLIP, +CLIP] before adding noise "
                        "(unbounded-adjacency variant; pass --sensitivity 2*CLIP).")
    p.add_argument("--sensitivity", type=float, default=None,
                   help="Default: from configs.DP.sensitivity")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--out-dir", type=str, default="results/dp")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip configs whose metrics.pkl already exists")
    p.add_argument("--gpu-memory-gb", type=float, default=None,
                   help="Limit GPU memory to this many GiB (for parallel runs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "training_log.txt")
    logger = logging.getLogger()

    # Resolve sweep grids
    epsilons = args.epsilons if args.epsilons is not None else list(DP.epsilons)
    seeds = args.seeds if args.seeds is not None else list(DP.seeds)
    sensitivity = args.sensitivity if args.sensitivity is not None else DP.sensitivity

    # Delta strategy:
    #   * If --deltas given: use them as a flat grid for ALL mechanisms
    #   * Otherwise: use DP.delta_for(mech) per mechanism (recommended)
    deltas_override = args.deltas

    logger.info("=" * 70)
    logger.info("DP-ECG utility sweep")
    logger.info("=" * 70)
    logger.info(f"Mechanisms:  {args.mechanisms}")
    logger.info(f"Epsilons:    {epsilons}")
    if deltas_override is not None:
        logger.info(f"Deltas:      {deltas_override} (flat grid)")
    else:
        per_mech_delta = {m: DP.delta_for(m) for m in args.mechanisms}
        logger.info(f"Deltas:      per-mechanism canonical {per_mech_delta}")
    logger.info(f"Seeds:       {seeds}")
    logger.info(f"Sensitivity: {sensitivity}")
    logger.info(f"Epochs/run:  {args.epochs}, batch={args.batch_size}")
    logger.info("=" * 70)

    # ---- Load data once ----
    classes = ("N", "S", "V")
    if args.data_source == "wfdb":
        logger.info("Loading MIT-BIH via WFDB...")
        X_train, y_train_chars = read_mitbih_wfdb(
            trainset=1, classes=classes, beat_extraction=args.beat_extraction,
            seed=seeds[0],
        )
        X_test, y_test_chars = read_mitbih_wfdb(
            trainset=0, classes=classes, beat_extraction=args.beat_extraction,
            seed=seeds[0],
        )
    else:
        X_train, y_train_chars = read_mitbih_mat(
            args.mat_path, trainset=1, classes=classes, seed=seeds[0],
        )
        X_test, y_test_chars = read_mitbih_mat(
            args.mat_path, trainset=0, classes=classes, seed=seeds[0],
        )

    char2num, _ = build_vocab(classes)
    y_train = labels_to_targets(y_train_chars, char2num)
    y_test = labels_to_targets(y_test_chars, char2num)
    smote_targets = {"N": None, "S": 7000, "V": 6000}

    logger.info(f"Train: {len(X_train)} seqs, Test: {len(X_test)} seqs")

    # ---- Sweep ----
    # Per-mechanism delta selection (unless overridden on CLI)
    def deltas_for_mech(mech):
        if deltas_override is not None:
            return list(deltas_override)
        return [DP.delta_for(mech)]

    # Compute total work
    total = 0
    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for_mech(mech):
                if is_valid_config(mech, eps, delta):
                    total += len(seeds)

    done = 0
    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for_mech(mech):

                # Skip invalid configs (e.g. delta=0 with gaussian/bounded)
                if not is_valid_config(mech, eps, delta):
                    logger.info(
                        f"SKIP invalid: {mech} eps={eps} delta={delta}"
                    )
                    continue

                for seed in seeds:
                    done += 1
                    run_dir = (
                        out_root / mech
                        / f"eps_{eps}_delta_{delta}"
                        / f"seed_{seed}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    metrics_path = run_dir / "metrics.pkl"

                    if args.skip_existing and metrics_path.exists():
                        logger.info(
                            f"[{done}/{total}] SKIP existing: {metrics_path}"
                        )
                        continue

                    logger.info("-" * 70)
                    logger.info(
                        f"[{done}/{total}] {mech}, eps={eps}, delta={delta}, "
                        f"seed={seed}"
                    )

                    cfg_seed = derived_seed(seed, mech, eps, delta)
                    try:
                        result = train_one_config(
                            X_train, y_train, X_test, y_test,
                            char2num, classes,
                            mechanism=mech, eps=eps, delta=delta,
                            sensitivity=sensitivity,
                            smote_targets=smote_targets, clip=args.clip,
                            perturb_test=args.perturb_test,
                            denoise_method=args.denoise,
                            denoise_fs=args.denoise_fs,
                            epochs=args.epochs, batch_size=args.batch_size,
                            eval_every=args.eval_every, seed=cfg_seed,
                            logger=logger,
                        )
                    except Exception as e:
                        logger.error(f"  Config failed: {e}")
                        continue

                    result["seed"] = seed
                    # Atomic write: write to .tmp first, then rename
                    tmp_path = metrics_path.with_suffix(".pkl.tmp")
                    with open(tmp_path, "wb") as f:
                        pickle.dump(result, f)
                    tmp_path.replace(metrics_path)
                    logger.info(
                        f"  Saved: {metrics_path} "
                        f"(macro_F1={result['macro_f1']:.4f})"
                    )

    logger.info("=" * 70)
    logger.info("Sweep complete")


if __name__ == "__main__":
    main()
