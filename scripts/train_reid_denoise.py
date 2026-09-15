"""
Temporal-correlation stress test: the *denoising* re-identification attacker.

Why
---
The DP mechanisms add i.i.d. noise to each sample of a z-normalized ECG window.
The signal is autocorrelated; the noise is not. A temporally-aware adversary can
low-pass / denoise the released window to average the noise away while keeping
the signal, then re-identify. This script measures exactly that, WITHOUT
changing the privacy mechanism: the released (noised) data is held fixed and the
attacker simply gains a denoising front end. Same data, smarter adversary.

The single most important design decision (from the handover brief): the
denoiser is trained/evaluated end to end as part of the *adaptive* protocol --
it is applied identically to the attacker's TRAIN and TEST inputs, not bolted on
at evaluation only. So the comparison is:

    adaptive               : train re-ID on DP windows,            eval on DP windows
    adaptive_denoise_<...>  : train re-ID on denoise(DP windows),  eval on denoise(DP windows)

Everything else (architecture, seeds, splits, identities) is identical to
``scripts/train_reid.py`` so the numbers line up. The gap is the leakage
recovered by exploiting temporal correlation.

Two attacker fronts (the brief's "minimal defensible attack" + the optional
stronger one):
  * --attack lowpass     : fixed Savitzky-Golay / moving-average / Gaussian
                           filter (src/dp/denoise.py).
  * --attack autoencoder : a small 1D denoising autoencoder trained to map
                           noised -> clean windows (src/models/denoise_ae.py).

This script does NOT modify ``scripts/train_reid.py``; it reuses its training
loop so results are directly comparable.

Output (same layout as train_reid.py, new threat-model folders so nothing
collides):
    results/reid[/_ecgid]/dp/<mech>/eps_<e>_delta_<d>/<threat_name>/seed_<s>/metrics.pkl

Examples
--------
    # Recommended representative sweep, Savitzky-Golay, MIT-BIH:
    python scripts/train_reid_denoise.py --dataset mitbih

    # ECG-ID, all three low-pass filters, also store a matched no-denoise number:
    python scripts/train_reid_denoise.py --dataset ecgid \
        --denoise-methods savgol moving_average gaussian \
        --compare-adaptive --out-dir results/reid_ecgid

    # Stronger learned attacker:
    python scripts/train_reid_denoise.py --dataset mitbih --attack autoencoder
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- GPU memory limit BEFORE TF import (mirror train_reid.py) ----
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

from configs import DP, REID
from src.data import build_ecgid_split, build_reid_all_mitbih, build_reid_ds1
from src.dp import describe_denoise, denoise, perturb_encoder_inputs
from src.dp.denoise import DENOISE_METHODS
from src.dp.mechanisms import is_valid_config

# Reuse the EXACT training/eval loop and seeding from train_reid.py so the
# denoise numbers are directly comparable to the existing adaptive numbers.
from scripts.train_reid import (
    derived_seed,
    setup_logging,
    train_and_eval_reid,
)

# Representative epsilon subset from the brief: two contested middle-band points
# plus a strong-privacy and a weak-privacy anchor. Enough to show the trend
# without re-running the full 15-point grid. All four are points on DP.epsilons,
# so they line up with the existing adaptive results.
REPRESENTATIVE_EPS = (0.05, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.35, 0.4, 0.5,
        0.75, 1.0, 2.0, 4.0,  20.0)


def _build_lowpass_denoiser(method: str, fs: float, args):
    """Return (apply_fn, param_record) for a fixed low-pass denoiser."""
    params = dict(
        fs=fs,
        window_length=args.savgol_window,
        polyorder=args.savgol_polyorder,
        savgol_ms=args.savgol_ms,
        gaussian_ms=args.gaussian_ms,
        moving_average_ms=args.moving_average_ms,
    )

    def apply_fn(X_clean, X_noised):
        # Low-pass attacker ignores the clean signal (no training); it only
        # filters the noised input it actually observes.
        return denoise(X_noised, method=method, **params)

    record = describe_denoise(method, **params)
    record["attack"] = "lowpass"
    return apply_fn, record


def _build_ae_denoiser(fs: float, args, cfg_seed: int, logger, log_prefix: str):
    """Return (apply_fn, param_record) for the denoising-autoencoder attacker.

    The AE is fit on the attacker's TRAIN windows only (clean targets, noised
    inputs); the test windows never touch the AE fit. The same fitted AE is then
    applied to both noised train and noised test.
    """
    # Imported lazily so the low-pass path has no hard dependency on the AE
    # module (and to keep TF import cost off the lowpass attacker).
    from src.models import ae_denoise, train_denoising_ae

    state: dict = {"model": None}

    def apply_train_then_test(X_clean, X_noised):
        if state["model"] is None:
            state["model"] = train_denoising_ae(
                clean_X=X_clean, noised_X=X_noised,
                epochs=args.ae_epochs, batch_size=args.ae_batch_size,
                learning_rate=args.ae_lr, base_filters=args.ae_filters,
                depth=args.ae_depth, kernel_size=args.ae_kernel,
                seed=cfg_seed + 7, logger=logger, log_prefix=log_prefix,
            )
        return ae_denoise(state["model"], X_noised, batch_size=args.ae_batch_size)

    record = dict(
        attack="autoencoder", method="autoencoder", fs=fs,
        ae_epochs=args.ae_epochs, ae_filters=args.ae_filters,
        ae_depth=args.ae_depth, ae_kernel=args.ae_kernel,
    )
    return apply_train_then_test, record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=["mitbih", "ecgid"], default="mitbih")
    p.add_argument("--data-split", choices=["ds1", "all"], default="ds1",
                   help="MIT-BIH only: ds1 = 22 patients, all = 44 patients.")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="MIT-BIH only: temporal fraction per record for training.")
    p.add_argument("--epochs", type=int, default=REID.epochs)
    p.add_argument("--batch-size", type=int, default=REID.batch_size)
    p.add_argument("--learning-rate", type=float, default=REID.learning_rate)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="One run per seed (default: configs.DP.seeds).")

    # Attacker front end
    p.add_argument("--attack", choices=["lowpass", "autoencoder"], default="lowpass",
                   help="Denoising attacker type.")
    p.add_argument("--denoise-methods", nargs="+", default=["savgol"],
                   choices=list(DENOISE_METHODS),
                   help="Low-pass filters to run (one threat-model folder each).")
    p.add_argument("--threat-name", type=str, default=None,
                   help="Override the output threat-model folder name.")
    p.add_argument("--compare-adaptive", action="store_true",
                   help="Also compute the matched NO-denoise adaptive number on "
                        "the identical perturbed inputs, stored next to the "
                        "denoise result for an airtight gap.")

    # Savitzky-Golay params (window derived from fs if not given)
    p.add_argument("--savgol-window", type=int, default=None)
    p.add_argument("--savgol-polyorder", type=int, default=3)
    p.add_argument("--savgol-ms", type=float, default=25.0)
    # Moving average / Gaussian params
    p.add_argument("--moving-average-ms", type=float, default=25.0)
    p.add_argument("--gaussian-ms", type=float, default=8.0)

    # Autoencoder params
    p.add_argument("--ae-epochs", type=int, default=30)
    p.add_argument("--ae-batch-size", type=int, default=128)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--ae-filters", type=int, default=32)
    p.add_argument("--ae-depth", type=int, default=2)
    p.add_argument("--ae-kernel", type=int, default=9)

    # DP grid
    p.add_argument("--mechanisms", nargs="+", default=list(DP.mechanisms))
    p.add_argument("--epsilons", type=float, nargs="+", default=None,
                   help=f"Default: representative subset {REPRESENTATIVE_EPS}.")
    p.add_argument("--full-grid", action="store_true",
                   help="Use the full configs.DP.epsilons grid instead of the "
                        "representative subset.")
    p.add_argument("--deltas", type=float, nargs="+", default=None)
    p.add_argument("--sensitivity", type=float, default=None)
    p.add_argument("--arch", choices=["cnn", "resnet"], default="cnn",
                   help="Attacker architecture behind the denoising front end.")
    p.add_argument("--clip", type=float, default=None,
                   help="Clamp signal values to [-CLIP, +CLIP] before adding noise "
                        "(unbounded-adjacency variant; pass --sensitivity 2*CLIP).")

    p.add_argument("--out-dir", type=str, default=None,
                   help="Default: results/reid (mitbih) or results/reid_ecgid (ecgid).")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--gpu-memory-gb", type=float, default=None)
    return p.parse_args()


def _threat_name(args, method: str) -> str:
    if args.threat_name is not None:
        return args.threat_name
    if args.attack == "autoencoder":
        return "adaptive_denoise_ae"
    return f"adaptive_denoise_{method}"


def main() -> None:
    args = parse_args()

    out_root = Path(
        args.out_dir if args.out_dir is not None
        else ("results/reid_ecgid" if args.dataset == "ecgid" else "results/reid")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "reid_denoise_log.txt")
    logger = logging.getLogger()

    # ---- Load Re-ID data (once) ----
    if args.dataset == "mitbih":
        logger.info(f"Loading MIT-BIH for Re-ID (split={args.data_split})...")
        if args.data_split == "ds1":
            train_ds, test_ds = build_reid_ds1(train_frac=args.train_frac)
        else:
            train_ds, test_ds = build_reid_all_mitbih(train_frac=args.train_frac)
        n_subjects = len(train_ds.patient_ids)
    else:
        logger.info("Loading ECG-ID for Re-ID (session-based split)...")
        train_ds, test_ds = build_ecgid_split()
        n_subjects = len(train_ds.subject_ids)

    window_samples = train_ds.X.shape[1]
    fs = float(train_ds.fs)
    logger.info(
        f"Loaded: {len(train_ds.X)} train windows, {len(test_ds.X)} test windows, "
        f"{n_subjects} subjects, {window_samples} samples/window, fs={fs:g} Hz"
    )

    sensitivity = args.sensitivity if args.sensitivity is not None else DP.sensitivity
    seeds = args.seeds if args.seeds is not None else list(DP.seeds)

    if args.epsilons is not None:
        epsilons = list(args.epsilons)
    elif args.full_grid:
        epsilons = list(DP.epsilons)
    else:
        epsilons = list(REPRESENTATIVE_EPS)

    # Which low-pass methods to iterate. The autoencoder ignores this list.
    methods = args.denoise_methods if args.attack == "lowpass" else ["autoencoder"]

    def deltas_for_mech(mech):
        if args.deltas is not None:
            return list(args.deltas)
        return [DP.delta_for(mech)]

    # ---- Count total runs for progress ----
    total = 0
    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for_mech(mech):
                if is_valid_config(mech, eps, delta):
                    total += len(methods) * len(seeds)
    done = 0

    logger.info("=" * 70)
    logger.info(
        f"DENOISING ATTACKER STRESS TEST  |  attack={args.attack}  "
        f"methods={methods}  eps={epsilons}  mechs={list(args.mechanisms)}  "
        f"seeds={seeds}  ({total} runs)"
    )
    logger.info("=" * 70)

    for mech in args.mechanisms:
        for eps in epsilons:
            for delta in deltas_for_mech(mech):
                if not is_valid_config(mech, eps, delta):
                    logger.info(f"SKIP invalid: {mech} eps={eps} delta={delta}")
                    continue

                for method in methods:
                    tname = _threat_name(args, method)
                    for seed in seeds:
                        done += 1
                        out_dir = (
                            out_root / "dp" / mech
                            / f"eps_{eps}_delta_{delta}" / tname
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
                            f"attack={args.attack} method={method} seed={seed}"
                        )

                        # Same per-config seed as train_reid.py's adaptive path,
                        # using the BASE 'adaptive_attacker' mode string so the DP
                        # noise draws are IDENTICAL to the existing adaptive run.
                        cfg_seed = derived_seed(
                            seed, mech, eps, delta, "adaptive_attacker"
                        )

                        t0 = time.time()
                        try:
                            # Identical perturbation to train_reid.py adaptive:
                            #   test  noised with cfg_seed
                            #   train noised with cfg_seed + 1
                            X_test_dp = perturb_encoder_inputs(
                                test_ds.X, mechanism=mech, eps=eps, delta=delta,
                                sensitivity=sensitivity, seed=cfg_seed,
                                clip=args.clip,
                            )
                            X_train_dp = perturb_encoder_inputs(
                                train_ds.X, mechanism=mech, eps=eps, delta=delta,
                                sensitivity=sensitivity, seed=cfg_seed + 1,
                                clip=args.clip,
                            )

                            # Build the denoiser for this config.
                            if args.attack == "lowpass":
                                apply_fn, denoise_record = _build_lowpass_denoiser(
                                    method, fs, args
                                )
                            else:
                                apply_fn, denoise_record = _build_ae_denoiser(
                                    fs, args, cfg_seed, logger,
                                    log_prefix=f"  ae[s{seed}] ",
                                )

                            # Denoise BOTH train and test, identically.
                            # (clean train windows passed so the AE can learn
                            #  noised->clean; the low-pass attacker ignores them.)
                            X_train_dn = apply_fn(train_ds.X, X_train_dp)
                            X_test_dn = apply_fn(test_ds.X, X_test_dp)

                            result = train_and_eval_reid(
                                X_train_dn, train_ds.y, X_test_dn, test_ds.y,
                                n_patients=n_subjects, window_samples=window_samples,
                                epochs=args.epochs, batch_size=args.batch_size,
                                learning_rate=args.learning_rate, seed=cfg_seed,
                                logger=logger,
                                log_prefix=f"  dn-{method[:6]}[s{seed}] ",
                                arch=args.arch,
                            )

                            # Optional matched no-denoise adaptive number on the
                            # SAME perturbed inputs, for an airtight gap.
                            if args.compare_adaptive:
                                matched = train_and_eval_reid(
                                    X_train_dp, train_ds.y, X_test_dp, test_ds.y,
                                    n_patients=n_subjects,
                                    window_samples=window_samples,
                                    epochs=args.epochs, batch_size=args.batch_size,
                                    learning_rate=args.learning_rate, seed=cfg_seed,
                                    logger=logger,
                                    log_prefix=f"  adapt[s{seed}] ",
                                    arch=args.arch,
                                )
                                result["matched_adaptive_accuracy"] = (
                                    matched["overall_accuracy"]
                                )
                                result["denoise_gain"] = (
                                    result["overall_accuracy"]
                                    - matched["overall_accuracy"]
                                )

                        except Exception as e:
                            logger.error(f"  FAILED: {type(e).__name__}: {e}")
                            continue

                        result["arch"] = args.arch
                        result["mechanism"] = mech
                        result["eps"] = eps
                        result["delta"] = delta
                        result["sensitivity"] = sensitivity
                        result["threat_model"] = tname
                        result["base_threat_model"] = "adaptive_attacker"
                        result["attack"] = args.attack
                        result["denoise"] = denoise_record
                        result["seed"] = seed

                        tmp_path = metrics_path.with_suffix(".pkl.tmp")
                        with open(tmp_path, "wb") as f:
                            pickle.dump(result, f)
                        tmp_path.replace(metrics_path)

                        gap_msg = ""
                        if "denoise_gain" in result:
                            gap_msg = (
                                f" | matched adaptive="
                                f"{result['matched_adaptive_accuracy']:.4f} "
                                f"(gain {result['denoise_gain']:+.4f})"
                            )
                        logger.info(
                            f"  Saved: denoise accuracy={result['overall_accuracy']:.4f}"
                            f"{gap_msg} (took {time.time()-t0:.1f}s)"
                        )

    logger.info("=" * 70)
    logger.info("Denoising-attacker stress test complete")


if __name__ == "__main__":
    main()
