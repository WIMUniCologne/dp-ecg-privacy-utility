#!/usr/bin/env bash
# Nachtlauf: F1 Seed-Robustheit ueber das Betriebsband, F2 geschuetztes Testset,
# F3 zweite Angreifer-Architektur. Ueberall --skip-existing, damit ein Neustart billig ist.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
NEW="7 123 2024 31337 99991"
S3="42 1337 2026"
BAND="0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.75"
banner(){ echo "[$(date +%F' '%T)] === $1 ==="; }

# ---------------------------------------------------------- F1 (der grosse Block)
banner "F1a Utility, Betriebsband, +5 Seeds (3 Mechanismen x 8 eps x 5 Seeds = 120)"
for M in laplace laplace_bounded gaussian_analytic; do
  $PY scripts/train_with_dp.py --mechanisms $M --epsilons $BAND --seeds $NEW \
      --epochs 200 --eval-every 20 --gpu-memory-gb 7 --skip-existing \
      > $L/f1_util_$M.log 2>&1 &
done
wait

banner "F1b Re-ID adaptiv, Betriebsband, +5 Seeds (240 Laeufe, beide Korpora parallel)"
$PY scripts/train_reid.py --dataset mitbih --data-split ds1 --dp-sweep \
    --threat-model adaptive_attacker --epsilons $BAND --seeds $NEW \
    --gpu-memory-gb 10 --skip-existing --out-dir results/reid > $L/f1_reid_mitbih.log 2>&1 &
$PY scripts/train_reid.py --dataset ecgid --dp-sweep \
    --threat-model adaptive_attacker --epsilons $BAND --seeds $NEW \
    --gpu-memory-gb 10 --skip-existing --out-dir results/reid_ecgid > $L/f1_reid_ecgid.log 2>&1 &
wait

# ---------------------------------------------------------- F2 geschuetztes Testset
banner "F2 Utility mit geschuetztem Testset (54 Laeufe)"
for M in laplace laplace_bounded gaussian_analytic; do
  $PY scripts/train_with_dp.py --mechanisms $M --epsilons 0.15 0.25 0.3 0.5 0.75 1.0 \
      --seeds $S3 --perturb-test --epochs 200 --eval-every 20 --gpu-memory-gb 7 \
      --skip-existing --out-dir results/dp_ptest > $L/f2_ptest_$M.log 2>&1 &
done
wait

# ---------------------------------------------------------- F3 zweite Architektur
banner "F3 ResNet-Angreifer (72 Laeufe, beide Korpora parallel)"
$PY scripts/train_reid.py --dataset mitbih --data-split ds1 --arch resnet --dp-sweep \
    --threat-model adaptive_attacker --epsilons 0.25 0.3 0.5 0.75 --seeds $S3 \
    --gpu-memory-gb 10 --skip-existing --out-dir results/reid_resnet > $L/f3_mitbih.log 2>&1 &
$PY scripts/train_reid.py --dataset ecgid --arch resnet --dp-sweep \
    --threat-model adaptive_attacker --epsilons 0.25 0.3 0.5 0.75 --seeds $S3 \
    --gpu-memory-gb 10 --skip-existing --out-dir results/reid_ecgid_resnet > $L/f3_ecgid.log 2>&1 &
wait

banner "NACHTLAUF FERTIG"
