#!/usr/bin/env bash
# Vollstaendiger Durchlauf aller noch offenen, reviewer-relevanten Experimente.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
NEW="7 123 2024 31337 99991"      # 5 zusaetzliche Seeds
S3="42 1337 2026"
G=7                                # GPU-GB je Worker

banner(){ echo "[$(date +%T)] === $1 ==="; }

# ---------------------------------------------------------------- E6 (CPU, parallel im Hintergrund)
banner "E6 RF bei hohem eps (CPU, laeuft nebenher)"
$PY scripts/train_utility_rf.py --mechanisms laplace laplace_bounded gaussian_analytic \
    --epsilons 50.0 100.0 --seeds $S3 --skip-existing \
    --out-dir results/utility_rf > $L/e6_rf.log 2>&1 &
RF=$!

# ---------------------------------------------------------------- E1 Seed-Robustheit
banner "E1 Seed-Robustheit an den Betriebspunkten (+5 Seeds)"
$PY scripts/train_with_dp.py --mechanisms laplace --epsilons 0.25 --seeds $NEW \
    --epochs 200 --eval-every 20 --gpu-memory-gb $G --skip-existing > $L/e1_u_lap.log 2>&1 &
$PY scripts/train_with_dp.py --mechanisms laplace_bounded --epsilons 0.3 --seeds $NEW \
    --epochs 200 --eval-every 20 --gpu-memory-gb $G --skip-existing > $L/e1_u_lb.log 2>&1 &
$PY scripts/train_with_dp.py --mechanisms gaussian_analytic --epsilons 0.25 0.75 --seeds $NEW \
    --epochs 200 --eval-every 20 --gpu-memory-gb $G --skip-existing > $L/e1_u_ga.log 2>&1 &
wait
for TM in adaptive_attacker clean_attacker; do
  for D in mitbih:reid ecgid:reid_ecgid; do
    DS=${D%%:*}; OUT=results/${D##*:}
    $PY scripts/train_reid.py --dataset $DS --dp-sweep --threat-model $TM \
        --mechanisms laplace --epsilons 0.25 --seeds $NEW --gpu-memory-gb $G \
        --skip-existing --out-dir $OUT > $L/e1_r_${DS}_${TM}_lap.log 2>&1
    $PY scripts/train_reid.py --dataset $DS --dp-sweep --threat-model $TM \
        --mechanisms laplace_bounded --epsilons 0.3 --seeds $NEW --gpu-memory-gb $G \
        --skip-existing --out-dir $OUT > $L/e1_r_${DS}_${TM}_lb.log 2>&1
    $PY scripts/train_reid.py --dataset $DS --dp-sweep --threat-model $TM \
        --mechanisms gaussian_analytic --epsilons 0.25 0.75 --seeds $NEW --gpu-memory-gb $G \
        --skip-existing --out-dir $OUT > $L/e1_r_${DS}_${TM}_ga.log 2>&1
  done
done

# ---------------------------------------------------------------- E2 Autoencoder-Angreifer
banner "E2 Lernender Entrauschungs-Angreifer (Autoencoder) -- staerkster Angreifer, nie gelaufen"
$PY scripts/train_reid_denoise.py --dataset ecgid --attack autoencoder \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons 0.25 0.3 0.5 0.75 \
    --seeds $S3 --compare-adaptive --gpu-memory-gb 10 --skip-existing \
    --out-dir results/reid_ecgid > $L/e2_ae_ecgid.log 2>&1 &
$PY scripts/train_reid_denoise.py --dataset mitbih --attack autoencoder \
    --mechanisms laplace laplace_bounded gaussian_analytic --epsilons 0.25 0.3 0.5 0.75 \
    --seeds $S3 --compare-adaptive --gpu-memory-gb 10 --skip-existing \
    --out-dir results/reid > $L/e2_ae_mitbih.log 2>&1 &
wait

# ---------------------------------------------------------------- E3 ECG-ID mit eigenem Delta
banner "E3 ECG-ID mit eigener Sensitivitaet 0.262 statt geliehener 0.29"
$PY scripts/train_reid.py --dataset ecgid --dp-sweep --threat-model adaptive_attacker \
    --mechanisms laplace laplace_bounded gaussian_analytic \
    --epsilons 0.15 0.2 0.25 0.3 0.5 0.75 --seeds $S3 --sensitivity 0.262 \
    --gpu-memory-gb 10 --skip-existing --out-dir results/reid_ecgid_delta262 \
    > $L/e3_delta262.log 2>&1

# ---------------------------------------------------------------- E4 Attribut balanciert
banner "E4 Attribut-Inferenz, klassenbalanciert, volle DP-Sweeps"
$PY scripts/train_attribute.py --dataset ecgid --attribute sex --balance-classes --dp-sweep \
    --epsilons 0.15 0.25 0.5 0.75 1.0 2.0 --seeds $S3 --gpu-memory-gb $G --skip-existing \
    --out-dir results/attr_balanced > $L/e4_ecgid_sex.log 2>&1 &
$PY scripts/train_attribute.py --dataset ecgid --attribute age --balance-classes --dp-sweep \
    --epsilons 0.15 0.25 0.5 0.75 1.0 2.0 --seeds $S3 --gpu-memory-gb $G --skip-existing \
    --out-dir results/attr_balanced > $L/e4_ecgid_age.log 2>&1 &
$PY scripts/train_attribute.py --dataset mitbih --attribute sex --balance-classes --dp-sweep \
    --epsilons 0.15 0.25 0.5 0.75 1.0 2.0 --seeds $S3 --gpu-memory-gb $G --skip-existing \
    --out-dir results/attr_balanced > $L/e4_mitbih_sex.log 2>&1 &
wait

# ---------------------------------------------------------------- E5 C-Optimum
banner "E5 Engeres Clipping: ist p95 wirklich optimal?"
for CC in "0.3753:0.7506:p50" "0.7023:1.4045:p75" "1.2991:2.5982:p90"; do
  C=${CC%%:*}; REST=${CC#*:}; S=${REST%%:*}; T=${REST##*:}
  $PY scripts/train_with_dp.py --mechanisms laplace --epsilons 1e9 1.0 2.0 4.0 --seeds $S3 \
      --clip $C --sensitivity $S --epochs 200 --eval-every 20 --gpu-memory-gb $G \
      --skip-existing --out-dir results/option_b/util/C_$T > $L/e5_u_$T.log 2>&1 &
done
wait
for CC in "0.3753:0.7506:p50" "0.7023:1.4045:p75" "1.2991:2.5982:p90"; do
  C=${CC%%:*}; REST=${CC#*:}; S=${REST%%:*}; T=${REST##*:}
  $PY scripts/train_reid.py --dataset ecgid --dp-sweep --threat-model adaptive_attacker \
      --mechanisms laplace --epsilons 1.0 2.0 4.0 --seeds $S3 --clip $C --sensitivity $S \
      --gpu-memory-gb $G --skip-existing --out-dir results/option_b/reid_ecgid_C_$T \
      > $L/e5_r_$T.log 2>&1 &
done
wait

wait $RF 2>/dev/null
banner "ALLE EXPERIMENTE FERTIG"
