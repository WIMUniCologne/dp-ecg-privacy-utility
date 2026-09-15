#!/usr/bin/env bash
# Staerkster Angreifer aus dem Baukasten: ResNet-Architektur HINTER dem gelernten
# Entrauschungs-Frontend. Beide Bausteine sind einzeln belegt staerker; die
# Kombination ist die naheliegende Frage, die ein Reviewer stellen wuerde.
set -u
PY="${PY:-python}"
L=results/option_b/logs
S3="42 1337 2026"
EPS="0.25 0.3 0.5 0.75"
echo "[$(date +%T)] ResNet + Autoencoder-Entrauschen, beide Korpora"
$PY scripts/train_reid_denoise.py --dataset ecgid --attack autoencoder --arch resnet \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons $EPS --seeds $S3 \
    --compare-adaptive --gpu-memory-gb 10 --skip-existing \
    --out-dir results/reid_ecgid_resnet > $L/combo_ecgid.log 2>&1 &
$PY scripts/train_reid_denoise.py --dataset mitbih --attack autoencoder --arch resnet \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons $EPS --seeds $S3 \
    --compare-adaptive --gpu-memory-gb 10 --skip-existing \
    --out-dir results/reid_resnet > $L/combo_mitbih.log 2>&1 &
wait
echo "[$(date +%T)] KOMBI FERTIG"
