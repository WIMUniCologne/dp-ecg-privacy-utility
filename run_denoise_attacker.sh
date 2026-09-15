#!/usr/bin/env bash
#
# Temporal-correlation stress test: the DENOISING re-identification attacker.
#
# Adds a temporally-aware adversary that low-pass / denoises the released
# (noised) windows before re-identifying. The privacy mechanism is unchanged --
# only the attacker is stronger. Compares, at matched (mechanism, eps, seed):
#
#     adaptive                 -> results/.../adaptive_attacker/...
#     adaptive_denoise_savgol  -> results/.../adaptive_denoise_savgol/...
#
# The gap is the leakage recovered by exploiting temporal correlation.
#
# Output paths (no collision with existing results):
#   results/reid/dp/<mech>/eps_<e>_delta_<d>/adaptive_denoise_savgol/seed_<S>/metrics.pkl
#   results/reid_ecgid/dp/<mech>/eps_<e>_delta_<d>/adaptive_denoise_savgol/seed_<S>/metrics.pkl
#
# Runs a REPRESENTATIVE epsilon subset {0.1, 0.25, 0.5, 1.0} x 3 mechanisms x
# 3 seeds, on both corpora -- enough to show the trend without a full re-sweep.
# --compare-adaptive also computes a matched no-denoise number in the same run,
# so the gap is airtight even without the main pipeline's adaptive results.
#
# Usage:
#   bash run_denoise_attacker.sh 2>&1 | tee patch_denoise.log
#
# Stronger learned attacker (optional, slower): set ATTACK=autoencoder.

set -e
set -u

ATTACK="${ATTACK:-lowpass}"            # lowpass | autoencoder
METHODS="${METHODS:-savgol}"          # space-separated: savgol moving_average gaussian

banner() {
    echo
    echo "================================================================"
    echo " $1"
    echo "================================================================"
    date
    echo
}

banner "Step 1: Denoising attacker on MIT-BIH (intra-record)"
python scripts/train_reid_denoise.py \
    --dataset mitbih \
    --data-split ds1 \
    --attack "${ATTACK}" \
    --denoise-methods ${METHODS} \
    --compare-adaptive \
    --out-dir results/reid \
    --skip-existing

banner "Step 2: Denoising attacker on ECG-ID (cross-session)"
python scripts/train_reid_denoise.py \
    --dataset ecgid \
    --attack "${ATTACK}" \
    --denoise-methods ${METHODS} \
    --compare-adaptive \
    --out-dir results/reid_ecgid \
    --skip-existing

banner "Denoising-attacker stress test complete."
echo "Sanity-check the filter first with:"
echo "  python scripts/diagnose/plot_denoise_sanity.py"
echo "Then compare adaptive vs adaptive_denoise_* in the metrics under results/."
