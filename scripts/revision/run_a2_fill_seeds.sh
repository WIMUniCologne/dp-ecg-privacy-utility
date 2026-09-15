#!/usr/bin/env bash
# Fuellt den A2-Lauf (Empfaenger mit Entrauscher) auf acht Seeds auf, damit die
# Zahlen auf demselben Fundament stehen wie der Rest des Betriebsbands.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
NEW="7 123 2024 31337 99991"
BAND="0.15 0.2 0.25 0.3 0.5 0.75"
echo "[$(date +%F' '%T)] A2 +5 Seeds"
for M in laplace laplace_bounded gaussian_analytic; do
  $PY scripts/train_with_dp.py --mechanisms $M --epsilons $BAND --seeds $NEW \
      --denoise savgol --epochs 200 --eval-every 20 --gpu-memory-gb 7 \
      --skip-existing --out-dir results/dp_denoise \
      > $L/a2_fill_$M.log 2>&1 &
done
wait
echo "[$(date +%F' '%T)] fertig"
