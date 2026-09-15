#!/usr/bin/env bash
#
# Experiment 2 -- Second utility classifier (feature-based Random Forest).
#
# Robustness check that the L-shaped utility curve is not an artifact of the
# seq2seq architecture. Same eps sweep / mechanisms / seeds / data / split /
# SMOTE / macro-F1 as scripts/train_with_dp.py -- only the model differs.
# No GPU required.
#
# Output (separate root, no collision with the seq2seq results in results/dp):
#   results/utility_rf/baseline/seed_<S>/metrics.pkl
#   results/utility_rf/<mech>/eps_<e>_delta_<d>/seed_<S>/metrics.pkl
#
# Usage:
#   bash run_second_utility.sh 2>&1 | tee rf.log

set -e
set -u

echo "================================================================"
echo " Experiment 2: Random Forest utility over the full DP sweep"
echo "================================================================"
date

python scripts/train_utility_rf.py --skip-existing

echo
echo "Done. Render: notebooks/07_second_utility.ipynb"
echo "Compares RF macro-F1 vs the seq2seq curve (results/dp) per mechanism."
