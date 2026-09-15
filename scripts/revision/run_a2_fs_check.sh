#!/usr/bin/env bash
# Kontrolle: der Empfaenger-Entrauscher lief mit fs=280, obwohl die auf 280 Punkte
# normierten Schlaege effektiv ~310 Hz entsprechen (Median-RR 0.90 s). Das ergibt ein
# Savgol-Fenster von 7 statt 9 Samples, also ~20 statt 25 ms -- der Angreifer glaettet
# ueber 25 ms. Drei Konfigurationen, drei Seeds, um zu sehen ob der Unterschied traegt.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
S3="42 1337 2026"
for spec in "laplace 0.25" "laplace_bounded 0.15" "gaussian_analytic 0.5"; do
  set -- $spec
  $PY scripts/train_with_dp.py --mechanisms $1 --epsilons $2 --seeds $S3 \
      --denoise savgol --denoise-fs 310 --epochs 200 --eval-every 20 --gpu-memory-gb 7 \
      --skip-existing --out-dir results/dp_denoise_fs310 \
      >> $L/a2_fs310.log 2>&1
done
echo "[$(date +%F' '%T)] fs-Kontrolle fertig"
