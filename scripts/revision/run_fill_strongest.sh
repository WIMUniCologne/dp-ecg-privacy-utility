#!/usr/bin/env bash
# Fuellt den staerksten Angreifer (ResNet hinter gelerntem Entrauschen) an den
# Betriebspunkten von Tabelle 1 auf acht Seeds auf. Die uebrigen Spalten der
# Tabelle stehen bereits auf acht; diese stand auf drei.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
NEW="7 123 2024 31337 99991"
EPS="0.25 0.5"
echo "[$(date +%F' '%T)] staerkster Angreifer, +5 Seeds, beide Korpora"
$PY scripts/train_reid_denoise.py --dataset ecgid --attack autoencoder --arch resnet \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons $EPS --seeds $NEW \
    --compare-adaptive --gpu-memory-gb 10 --skip-existing \
    --out-dir results/reid_ecgid_resnet > $L/fill_strong_ecgid.log 2>&1 &
$PY scripts/train_reid_denoise.py --dataset mitbih --attack autoencoder --arch resnet \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons $EPS --seeds $NEW \
    --compare-adaptive --gpu-memory-gb 10 --skip-existing \
    --out-dir results/reid_resnet > $L/fill_strong_mitbih.log 2>&1 &
wait
echo "[$(date +%F' '%T)] fertig"
