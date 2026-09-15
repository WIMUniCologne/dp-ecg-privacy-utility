#!/usr/bin/env bash
# Der staerkste Angreifer (gelernter Entrauscher + ResNet) ueber das VOLLE
# Betriebsband, acht Seeds ueberall. Nach dem Empfaenger-Experiment traegt
# dieser Angreifer die Kernaussage allein; bisher war er nur an vier Budgets
# gemessen, zwei davon mit drei Seeds. Ohne das ist "in the grid" ueberzeichnet.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
S8="42 1337 2026 7 123 2024 31337 99991"
BAND="0.15 0.2 0.25 0.3 0.5 0.75"
echo "[$(date +%F' '%T)] Staerkster Angreifer, volles Band, 8 Seeds (ECG-ID)"
$PY scripts/train_reid_denoise.py --dataset ecgid --attack autoencoder --arch resnet \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons $BAND --seeds $S8 \
    --compare-adaptive --gpu-memory-gb 10 --skip-existing \
    --out-dir results/reid_ecgid_resnet > $L/strongest_band_ecgid.log 2>&1
echo "[$(date +%F' '%T)] ECG-ID fertig"
