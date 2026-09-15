#!/usr/bin/env bash
# Alle ECG-ID-Re-ID-Laeufe neu, nach der Korrektur der Sitzungssortierung in
# src/data/ecgid.py. sorted(recs) war lexikografisch, also rec_1, rec_10, rec_11,
# ..., rec_2; drei Personen mit >=10 Sitzungen (zusammen 16 % der Test-Fenster)
# bekamen dadurch einen verschachtelten statt chronologischen Split. Gemessen an
# einer Konfiguration ueber drei Seeds verschiebt das die adaptive Leakage um
# +0.039 -- zu gross, um es stehen zu lassen.
#
# Nur zwei Prozesse parallel: die GPU teilen wir uns, ~9.5 GB sind belegt.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
S8="42 1337 2026 7 123 2024 31337 99991"
S3="42 1337 2026"
BAND="0.15 0.175 0.2 0.225 0.25 0.3 0.35 0.4 0.5 0.75 1.0"
ALL="0.05 0.1 0.15 0.175 0.2 0.225 0.25 0.3 0.35 0.4 0.5 0.75 1.0 1.5 2.0 4.0 8.0 20.0"

banner(){ echo; echo "[$(date +%F' '%T)] $*"; }

banner "1/4 naiver Angreifer, volles Gitter"
$PY scripts/train_reid.py --dataset ecgid --dp-sweep --threat-model clean_attacker \
    --epsilons $ALL --seeds $S8 --gpu-memory-gb 5 --skip-existing \
    --out-dir results/reid_ecgid > $L/fix_naive.log 2>&1 &
banner "2/4 adaptiver Angreifer, volles Gitter"
$PY scripts/train_reid.py --dataset ecgid --dp-sweep --threat-model adaptive_attacker \
    --epsilons $ALL --seeds $S8 --gpu-memory-gb 5 --skip-existing \
    --out-dir results/reid_ecgid > $L/fix_adaptive.log 2>&1 &
wait

banner "3/4 Entrauscher-Stufen der Leiter (savgol, Autoencoder)"
$PY scripts/train_reid_denoise.py --dataset ecgid --attack lowpass --arch cnn \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons $BAND --seeds $S3 \
    --compare-adaptive --gpu-memory-gb 5 --skip-existing \
    --out-dir results/reid_ecgid > $L/fix_savgol.log 2>&1 &
$PY scripts/train_reid_denoise.py --dataset ecgid --attack autoencoder --arch cnn \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons 0.25 0.3 0.5 0.75 --seeds $S3 \
    --compare-adaptive --gpu-memory-gb 5 --skip-existing \
    --out-dir results/reid_ecgid > $L/fix_ae.log 2>&1 &
wait

banner "4/4 ResNet allein und staerkster Angreifer ueber das Betriebsband"
$PY scripts/train_reid.py --dataset ecgid --arch resnet --dp-sweep \
    --threat-model adaptive_attacker --epsilons 0.25 0.3 0.5 0.75 --seeds $S3 \
    --gpu-memory-gb 5 --skip-existing \
    --out-dir results/reid_ecgid_resnet > $L/fix_resnet.log 2>&1 &
$PY scripts/train_reid_denoise.py --dataset ecgid --attack autoencoder --arch resnet \
    --mechanisms laplace laplace_bounded gaussian_analytic \
    --epsilons 0.15 0.2 0.25 0.3 0.5 0.75 --seeds $S8 \
    --compare-adaptive --gpu-memory-gb 5 --skip-existing \
    --out-dir results/reid_ecgid_resnet > $L/fix_strongest.log 2>&1 &
wait
banner "ECG-ID MIT KORRIGIERTEM SPLIT FERTIG"
