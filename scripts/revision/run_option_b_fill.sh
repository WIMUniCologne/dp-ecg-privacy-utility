#!/usr/bin/env bash
# Fuellt die Luecke zwischen epsB=4 (Utility .826 / ECG-ID-Leak .169) und epsB=8
# (Utility .920). Dort entscheidet sich, ob Clipping+DP die strenge Schranke
# (Utility >= 0.85 UND Leak <= 0.20) erreicht -- also ob der "leere Winkel" haelt.
set -u
PY="${PY:-python}"
LOG=results/option_b/logs
C=2.0812; S=4.1624
EPS="3.0 5.0 6.0"
echo "[$(date +%T)] warte auf Ende des naiven Angreifers ..."
until grep -qa "NAIV-2026 FERTIG" $LOG/naive_driver.log 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] GPU frei. Fuelle epsB in {3,5,6}."

$PY scripts/train_with_dp.py --mechanisms laplace --epsilons $EPS --seeds 42 1337 2026 \
    --clip $C --sensitivity $S --epochs 200 --eval-every 20 --gpu-memory-gb 9 \
    --skip-existing --out-dir results/option_b/util/C_p95 > $LOG/fill_util.log 2>&1
echo "[$(date +%T)] Utility fertig, starte Re-ID"

$PY scripts/train_reid.py --dataset ecgid --dp-sweep --threat-model adaptive_attacker \
    --mechanisms laplace --epsilons $EPS 8.0 --seeds 42 1337 2026 \
    --clip $C --sensitivity $S --gpu-memory-gb 9 --skip-existing \
    --out-dir results/option_b/reid_ecgid_C_p95 > $LOG/fill_reid_ecgid.log 2>&1 &
$PY scripts/train_reid.py --dataset mitbih --data-split ds1 --dp-sweep \
    --threat-model adaptive_attacker --mechanisms laplace --epsilons $EPS 8.0 \
    --seeds 42 1337 2026 --clip $C --sensitivity $S --gpu-memory-gb 9 --skip-existing \
    --out-dir results/option_b/reid_mitbih_C_p95 > $LOG/fill_reid_mitbih.log 2>&1 &
wait
echo "[$(date +%T)] FILL FERTIG"
