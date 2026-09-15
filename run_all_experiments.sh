#!/usr/bin/env bash
#
# Add-on experiments, simplified:
#   (A) Re-ID with clean_attacker on MIT-BIH and ECG-ID.
#   (B) One sweep per target/dataset over the FULL epsilon list.
#       --skip-existing means already-computed configs are skipped,
#       so only the new epsilons actually run.
#
# Choose ONE of the two epsilon strategies below (see EPSILONS).
#
# Usage:
#   bash run_addon.sh
#   bash run_addon.sh 2>&1 | tee run_addon.log

set -e
set -u

# ============================================================================
# Configuration
# ============================================================================
WORKERS=3
RESULTS_ROOT="results"

# Epsilon strategy:
#   - leave EPSILONS empty  -> script omits --epsilons, sweep uses configs.DP.epsilons
#   - set EPSILONS=(...)     -> script passes them explicitly (full list, old + new)
EPSILONS=()                         # e.g. EPSILONS=(0.1 0.175 0.2 0.225 0.25 0.35 ...)

DP_DIR="${RESULTS_ROOT}/dp"
REID_DIR="${RESULTS_ROOT}/reid"
REID_ECGID_DIR="${RESULTS_ROOT}/reid_ecgid"

# Build the optional --epsilons argument array
EPS_ARG=()
if [ "${#EPSILONS[@]}" -gt 0 ]; then
    EPS_ARG=(--epsilons "${EPSILONS[@]}")
fi

# ============================================================================
banner() {
    echo
    echo "================================================================"
    echo " $1"
    echo "================================================================"
    date
    echo
}

# ============================================================================
# Step 1: Re-ID clean_attacker (adaptive results untouched)
# ============================================================================
banner "Step 1a: Re-ID clean_attacker on MIT-BIH (DS1)"

python scripts/run_parallel_sweep.py \
    --target reid \
    --reid-dataset mitbih \
    --reid-data-split ds1 \
    --reid-threat-model clean_attacker \
    --workers "${WORKERS}" \
    --out-dir "${REID_DIR}" \
    "${EPS_ARG[@]}" \
    --skip-existing

banner "Step 1b: Re-ID clean_attacker on ECG-ID"

python scripts/run_parallel_sweep.py \
    --target reid \
    --reid-dataset ecgid \
    --reid-threat-model clean_attacker \
    --workers "${WORKERS}" \
    --out-dir "${REID_ECGID_DIR}" \
    "${EPS_ARG[@]}" \
    --skip-existing

# ============================================================================
# Step 2: Full epsilon sweep (skip-existing fills only the gaps)
# ============================================================================
banner "Step 2a: Utility sweep (all epsilons)"

python scripts/run_parallel_sweep.py \
    --target utility \
    --workers "${WORKERS}" \
    --out-dir "${DP_DIR}" \
    "${EPS_ARG[@]}" \
    --skip-existing

banner "Step 2b: Re-ID sweep on MIT-BIH (all epsilons, adaptive)"

python scripts/run_parallel_sweep.py \
    --target reid \
    --reid-dataset mitbih \
    --reid-data-split ds1 \
    --reid-threat-model adaptive_attacker \
    --workers "${WORKERS}" \
    --out-dir "${REID_DIR}" \
    "${EPS_ARG[@]}" \
    --skip-existing

banner "Step 2c: Re-ID sweep on ECG-ID (all epsilons, adaptive)"

python scripts/run_parallel_sweep.py \
    --target reid \
    --reid-dataset ecgid \
    --reid-threat-model adaptive_attacker \
    --workers "${WORKERS}" \
    --out-dir "${REID_ECGID_DIR}" \
    "${EPS_ARG[@]}" \
    --skip-existing

# ============================================================================
banner "Add-on sweeps complete!"

echo "Result directories:"
echo "  Utility:  ${DP_DIR}/"
echo "  MIT-BIH:  ${REID_DIR}/"
echo "  ECG-ID:   ${REID_ECGID_DIR}/"
echo
echo "Next: re-render figures in notebooks/02_money_figure.ipynb."