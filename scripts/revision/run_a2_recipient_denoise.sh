#!/usr/bin/env bash
# A2: der Empfaenger bekommt denselben Entrauscher wie der Angreifer.
# Das Paper stellt auf, eine Privatheitsaussage sei nur so stark wie der
# haerteste Angreifer, wendet das aber nur auf einer Achse an. Steigt die
# Utility mit Entrauschen kaum, ist die Verschraenkungsthese belegt statt
# plausibilisiert; steigt sie deutlich, muss die Aussage angepasst werden.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
S3="42 1337 2026"
BAND="0.15 0.2 0.25 0.3 0.5 0.75"
echo "[$(date +%F' '%T)] Empfaenger mit savgol-Entrauscher, Betriebsband"
for M in laplace laplace_bounded gaussian_analytic; do
  $PY scripts/train_with_dp.py --mechanisms $M --epsilons $BAND --seeds $S3 \
      --denoise savgol --epochs 200 --eval-every 20 --gpu-memory-gb 7 \
      --skip-existing --out-dir results/dp_denoise \
      > $L/a2_denoise_$M.log 2>&1 &
done
wait
echo "[$(date +%F' '%T)] fertig"
