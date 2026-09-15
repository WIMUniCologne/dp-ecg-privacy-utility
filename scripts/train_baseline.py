"""
Train the seq2seq classifier (no DP).

This is the baseline utility experiment: train the Mousavi seq2seq architecture
on MIT-BIH DS1, evaluate on DS2. Establishes the reference accuracy that the
DP experiments will be compared against.

Usage:
    # Default: load from MIT-BIH WFDB (PhysioNet) via qsPeaks
    python scripts/train_baseline.py

    # Or use Mousavi's pre-processed .mat file (for reproduction check)
    python scripts/train_baseline.py --data-source mat --mat-path data/s2s_mitbih_aami_DS1DS2.mat

Output:
    results/baseline/metrics.pkl
    results/baseline/training_log.txt
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- Configure GPU memory before importing TF (if requested) ----
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

from configs import SEQ2SEQ
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
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--data-source", choices=["wfdb", "mat"], default="wfdb",
        help="'wfdb' loads from raw MIT-BIH via PhysioNet+qsPeaks (default). "
             "'mat' loads from Mousavi's preprocessed .mat file.",
    )
    p.add_argument(
        "--mat-path", type=str, default=None,
        help="Path to .mat file (required if --data-source=mat)",
    )
    p.add_argument(
        "--beat-extraction", choices=["qspeaks", "rr_resample"], default="qspeaks",
        help="Beat extraction method for WFDB (ignored for mat)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=654)
    p.add_argument("--out-dir", type=str, default="results/baseline")
    p.add_argument("--gpu-memory-gb", type=float, default=None,
                   help="Limit GPU memory to this many GiB (for parallel runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "training_log.txt")
    logger = logging.getLogger()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---- Load data ----
    classes = ("N", "S", "V")
    if args.data_source == "wfdb":
        logger.info("Loading MIT-BIH via WFDB (DS1 + DS2)...")
        X_train, y_train_chars = read_mitbih_wfdb(
            trainset=1, classes=classes, beat_extraction=args.beat_extraction,
            seed=args.seed,
        )
        X_test, y_test_chars = read_mitbih_wfdb(
            trainset=0, classes=classes, beat_extraction=args.beat_extraction,
            seed=args.seed,
        )
    else:
        if args.mat_path is None:
            raise SystemExit("--mat-path is required when --data-source=mat")
        logger.info(f"Loading from {args.mat_path}...")
        X_train, y_train_chars = read_mitbih_mat(
            args.mat_path, trainset=1, classes=classes, seed=args.seed,
        )
        X_test, y_test_chars = read_mitbih_mat(
            args.mat_path, trainset=0, classes=classes, seed=args.seed,
        )

    logger.info(f"Train: {len(X_train)} sequences, Test: {len(X_test)} sequences")

    # ---- Vocab + label encoding ----
    char2num, go_id = build_vocab(classes)
    vocab_size = len(char2num)
    n_classes = len(classes)
    logger.info(f"Vocab: {char2num}")

    y_train = labels_to_targets(y_train_chars, char2num)
    y_test = labels_to_targets(y_test_chars, char2num)

    # ---- SMOTE oversampling ----
    smote_targets = {"N": None, "S": 7000, "V": 6000}  # Mousavi's targets
    logger.info(f"SMOTE targets: {smote_targets}")
    X_train_smote, y_train_smote = apply_smote(
        X_train, y_train, char2num, classes, smote_targets,
        smote_seed=args.seed,
    )
    train_dec_in = build_decoder_inputs(y_train_smote, go_id)
    test_dec_in = build_decoder_inputs(y_test, go_id)
    logger.info(f"Post-SMOTE: {len(X_train_smote)} sequences")

    # ---- Build model ----
    model = build_seq2seq_model(
        n_channels=10,
        input_depth=X_train_smote.shape[-1],
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
    logger.info(f"Model built: {model.count_params():,} parameters")

    # ---- tf.data pipeline ----
    train_ds = (
        tf.data.Dataset.from_tensor_slices(((X_train_smote, train_dec_in), y_train_smote))
        .cache()
        .shuffle(buffer_size=len(X_train_smote), seed=args.seed, reshuffle_each_iteration=True)
        .batch(args.batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(f"Pipeline: batch={args.batch_size}, ~{len(X_train_smote)//args.batch_size} steps/epoch")

    # ---- Greedy decoder for eval ----
    decoder = GreedyDecoder(
        model,
        num_units=SEQ2SEQ.lstm_units, vocab_size=vocab_size,
        go_id=go_id, bidirectional=False,
        max_time=SEQ2SEQ.beats_per_group,
        input_depth=X_train_smote.shape[-1],
    )

    # ---- Train ----
    best_macro = -np.inf
    best_weights = None
    history = []

    for ep in range(1, args.epochs + 1):
        h = model.fit(train_ds, epochs=1, verbose=0)
        train_loss = float(h.history["loss"][0])
        train_acc = float(h.history["sparse_categorical_accuracy"][0])

        if ep % args.eval_every == 0 or ep == args.epochs:
            preds = decoder.decode(X_test, batch_size=args.batch_size)
            cm = confusion_matrix_per_class(y_test.flatten(), preds.flatten(), n_classes)
            sens, ppv, f1 = metrics_from_cm(cm)
            macro = float(f1.mean())
            logger.info(
                f"[ep {ep:>3}] loss={train_loss:.4f}, acc={train_acc:.4f}, "
                f"macro_F1={macro:.4f}, per-class F1={[f'{x:.3f}' for x in f1]}"
            )
            history.append({
                "epoch": ep,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "macro_f1": macro,
                "per_class_f1": f1.tolist(),
                "sensitivity": sens.tolist(),
                "precision": ppv.tolist(),
            })
            if macro > best_macro:
                best_macro = macro
                best_weights = [w.copy() for w in model.get_weights()]

    # ---- Final eval with best weights ----
    if best_weights is not None:
        model.set_weights(best_weights)
    preds = decoder.decode(X_test, batch_size=args.batch_size)
    cm = confusion_matrix_per_class(y_test.flatten(), preds.flatten(), n_classes)
    sens, ppv, f1 = metrics_from_cm(cm)

    logger.info("=" * 70)
    logger.info(f"FINAL (best weights): macro F1 = {f1.mean():.4f}")
    for i, cl in enumerate(classes):
        logger.info(f"  {cl}: Sens={sens[i]:.4f}, PPV={ppv[i]:.4f}, F1={f1[i]:.4f}")

    # ---- Save results ----
    out = {
        "classes": list(classes),
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "sensitivity": sens.tolist(),
        "precision": ppv.tolist(),
        "confusion": cm.tolist(),
        "history": history,
        "args": vars(args),
    }
    metrics_path = out_dir / "metrics.pkl"
    tmp_path = metrics_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(out, f)
    tmp_path.replace(metrics_path)
    logger.info(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
