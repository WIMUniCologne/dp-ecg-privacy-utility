#!/usr/bin/env bash
# (1) epsB=5 mit 5 zusaetzlichen Seeds festnageln -- der Punkt liegt bei
#     (0.846, 0.196) gegen eine Schranke von (0.850, 0.200), Seed-sd 0.025.
# (2) Temporal-aware (denoising) Angreifer unter Clipping -- der staerkste
#     Angreifer entscheidet, ob der Winkel wirklich leer ist.
set -u
PY="${PY:-python}"
LOG=results/option_b/logs
C=2.0812; S=4.1624
NEW="7 123 2024 31337 99991"

echo "[$(date +%T)] (1) epsB=5, zusaetzliche Seeds: $NEW"
$PY scripts/train_with_dp.py --mechanisms laplace --epsilons 5.0 --seeds $NEW \
    --clip $C --sensitivity $S --epochs 200 --eval-every 20 --gpu-memory-gb 9 \
    --skip-existing --out-dir results/option_b/util/C_p95 > $LOG/pin_util.log 2>&1
$PY scripts/train_reid.py --dataset ecgid --dp-sweep --threat-model adaptive_attacker \
    --mechanisms laplace --epsilons 5.0 --seeds $NEW --clip $C --sensitivity $S \
    --gpu-memory-gb 9 --skip-existing --out-dir results/option_b/reid_ecgid_C_p95 \
    > $LOG/pin_reid.log 2>&1
echo "[$(date +%T)] (2) temporal-aware Angreifer unter Clipping"
$PY scripts/train_reid_denoise.py --dataset ecgid --attack lowpass \
    --denoise-methods savgol --mechanisms laplace --epsilons 4.0 5.0 6.0 \
    --seeds 42 1337 2026 --clip $C --sensitivity $S --compare-adaptive \
    --gpu-memory-gb 9 --skip-existing --out-dir results/option_b/reid_ecgid_C_p95 \
    > $LOG/pin_denoise.log 2>&1
echo "[$(date +%T)] PIN FERTIG"
