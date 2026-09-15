#!/usr/bin/env bash
# A2-Restlaeufe (eps 0.5 und 0.75) auf acht Seeds. Sequentiell je Mechanismus,
# damit sie sich die GPU nicht streitig machen.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
echo "[$(date +%F' '%T)] A2 Restlaeufe, sequentiell"
for M in laplace laplace_bounded gaussian_analytic; do
  $PY scripts/train_with_dp.py --mechanisms $M --epsilons 0.5 0.75 \
      --seeds 42 1337 2026 7 123 2024 31337 99991 \
      --denoise savgol --epochs 200 --eval-every 20 --gpu-memory-gb 10 \
      --skip-existing --out-dir results/dp_denoise >> $L/finish_a2_seq.log 2>&1
done
echo "[$(date +%F' '%T)] fertig"
