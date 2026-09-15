#!/usr/bin/env bash
# Restlaeufe: A2 auf acht Seeds vervollstaendigen und die drei naiven
# Ein-Seed-Punkte auffuellen. Beides mit --skip-existing.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
echo "[$(date +%F' '%T)] Restlaeufe"
for M in laplace laplace_bounded gaussian_analytic; do
  $PY scripts/train_with_dp.py --mechanisms $M --epsilons 0.5 0.75 \
      --seeds 42 1337 2026 7 123 2024 31337 99991 \
      --denoise savgol --epochs 200 --eval-every 20 --gpu-memory-gb 6 \
      --skip-existing --out-dir results/dp_denoise > $L/finish_a2_$M.log 2>&1 &
done
$PY scripts/train_reid.py --dataset ecgid --dp-sweep \
    --threat-model clean_attacker --epsilons 0.175 0.225 0.35 \
    --seeds 42 1337 2026 7 123 --gpu-memory-gb 6 --skip-existing \
    --out-dir results/reid_ecgid > $L/finish_naive.log 2>&1 &
wait
echo "[$(date +%F' '%T)] fertig"
