#!/usr/bin/env bash
#
# Add the non-adaptive ("naive") attacker as a complement to the existing
# adaptive attacker results. Produces a privacy band (naive lower bound,
# adaptive upper bound) instead of a single point.
#
# Output paths (no collision with existing adaptive_attacker results):
#   results/reid/dp/<mech>/eps_<e>_delta_<d>/clean_attacker/seed_<S>/metrics.pkl
#   results/reid_ecgid/dp/<mech>/eps_<e>_delta_<d>/clean_attacker/seed_<S>/metrics.pkl
#
# Estimated time on RTX 3090 Ti: ~3-4h per dataset, ~6-8h total.
#
# Usage:
#   bash run_naive_attacker.sh 2>&1 | tee patch_naive.log

set -e
set -u

banner() {
    echo
    echo "================================================================"
    echo " $1"
    echo "================================================================"
    date
    echo
}

banner "Step 1: Re-ID naive attacker on MIT-BIH"
python scripts/run_parallel_sweep.py \
    --target reid \
    --reid-dataset mitbih \
    --reid-data-split ds1 \
    --reid-threat-model clean_attacker \
    --workers 3 \
    --out-dir results/reid \
    --skip-existing

banner "Step 2: Re-ID naive attacker on ECG-ID"
python scripts/run_parallel_sweep.py \
    --target reid \
    --reid-dataset ecgid \
    --reid-threat-model clean_attacker \
    --workers 3 \
    --out-dir results/reid_ecgid \
    --skip-existing

banner "Naive attacker sweep complete."
echo "Open notebooks/04_attacker_band.ipynb to render the band figures."
