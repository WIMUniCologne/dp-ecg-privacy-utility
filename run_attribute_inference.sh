#!/usr/bin/env bash
#
# Experiment 1 -- Attribute inference as a second leakage axis.
#
# Trains an attribute classifier (same generic 1D-CNN as the re-ID attacker,
# attribute output head) on the DP-protected signal, subject-disjoint, over the
# same eps sweep / mechanisms / seeds as the re-ID attacker. The point: at the
# eps that pushes re-ID onto the operating point, demographic accuracy is still
# well above its chance floor -> a second L-shaped leakage curve.
#
# GATE: run the verification first and read the balance before trusting curves.
#   python scripts/diagnose/verify_attribute_labels.py
#
# Decisions (see that report): MIT-BIH sex only; ECG-ID sex + age (2 bands).
#
# Output:
#   results/attr/<dataset>/<attribute>/baseline/seed_<S>.pkl       (no-privacy anchor)
#   results/attr/<dataset>/<attribute>/dp/<mech>/eps_.../<threat>/seed_<S>/metrics.pkl
#
# Usage:
#   bash run_attribute_inference.sh 2>&1 | tee attr.log

set -e
set -u

banner() { echo; echo "================================================================";
           echo " $1"; echo "================================================================"; date; echo; }

banner "GATE: verify attribute-label availability and balance"
python scripts/diagnose/verify_attribute_labels.py || true

# Tip: judge the attacker by its no-privacy ANCHOR, not the eps-curve. If the
# anchor sits on the floor (e.g. age), the attribute is not learnable here.
# To strengthen a working attacker (legitimate: stronger adversary = more
# conservative), add --balance-classes (balanced oversampling of train windows)
# and compare the anchor, e.g.:
#   python scripts/train_attribute.py --dataset ecgid --attribute sex --balance-classes
# Then keep the choice fixed for the full sweep.

banner "ECG-ID -- SEX (subject-disjoint, adaptive attacker, full sweep)"
python scripts/train_attribute.py --dataset ecgid --attribute sex \
    --dp-sweep --threat-model adaptive_attacker --skip-existing

banner "ECG-ID -- AGE (2 bands at cohort median, subject-disjoint)"
python scripts/train_attribute.py --dataset ecgid --attribute age --n-age-bands 2 \
    --dp-sweep --threat-model adaptive_attacker --skip-existing

banner "MIT-BIH -- SEX (subject-disjoint; age dropped as too thin)"
python scripts/train_attribute.py --dataset mitbih --attribute sex \
    --dp-sweep --threat-model adaptive_attacker --skip-existing

banner "Experiment 1 complete."
echo "Render: notebooks/06_attribute_leakage.ipynb"
