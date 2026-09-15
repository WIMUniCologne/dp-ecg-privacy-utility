#!/usr/bin/env bash
# Die ECG-ID-Seite von Option B (unbounded adjacency + Clipping) neu, nach der
# Korrektur der Sitzungssortierung. Ohne das vergleicht Punkt 1 des Briefes die
# Clipping-Variante auf dem alten Split gegen eine Grundlinie auf dem neuen.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
SEEDS="42 1337 2026"
EPS="1.0 3.0 4.0 5.0 6.0 8.0 20.0"
C95=$(grep -oP 'C95=\K[0-9.]+' scripts/revision/run_option_b.sh | head -1)
S95=$(grep -oP 'S95=\K[0-9.]+' scripts/revision/run_option_b.sh | head -1)
echo "[$(date +%F' '%T)] Option B ECG-ID, C=$C95 sens=$S95"
$PY scripts/train_reid.py --dataset ecgid --dp-sweep \
    --threat-model adaptive_attacker --mechanisms laplace --epsilons $EPS \
    --seeds $SEEDS --clip $C95 --sensitivity $S95 --gpu-memory-gb 5 --skip-existing \
    --out-dir results/option_b/reid_ecgid_C_p95 > $L/optionb_ecgid_fixed.log 2>&1
echo "[$(date +%F' '%T)] fertig"
