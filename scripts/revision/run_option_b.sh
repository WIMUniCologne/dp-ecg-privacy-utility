#!/usr/bin/env bash
# Option B (unbounded adjacency + clipping) — Gegenrechnung zur metrischen DP.
# C in {p95, p99, max} von |x| auf dem z-normalisierten MIT-BIH-Signal.
# Globale L1-Sensitivitaet unter dieser Adjazenz = 2C.
set -u
PY="${PY:-python}"
ROOT=results/option_b
LOG=$ROOT/logs
SEEDS="42 1337 2026"
EPS="1.0 2.0 4.0 8.0 20.0 50.0"
mkdir -p $LOG

# C -> 2C
C95=2.0812;  S95=4.1624
C99=4.3178;  S99=8.6356
CMX=13.3111; SMX=26.6222

util () {  # $1=C $2=2C $3=eps-liste $4=tag
  $PY scripts/train_with_dp.py --mechanisms laplace --epsilons $3 --seeds $SEEDS \
      --clip $1 --sensitivity $2 --epochs 200 --eval-every 20 \
      --gpu-memory-gb 7 --skip-existing \
      --out-dir $ROOT/util/C_$4 > $LOG/util_$4.log 2>&1
}

echo "[$(date +%T)] B0: Clipping allein (eps=1e9, Rauschen vernachlaessigbar)"
util $C95 $S95 1e9 p95_cliponly &
util $C99 $S99 1e9 p99_cliponly &
util $CMX $SMX 1e9 max_cliponly &
wait
echo "[$(date +%T)] B0 fertig"

echo "[$(date +%T)] B1-B3: Clipping + Laplace bei sens=2C"
util $C95 $S95 "$EPS" p95 &
util $C99 $S99 "$EPS" p99 &
util $CMX $SMX "$EPS" max &
wait
echo "[$(date +%T)] B1-B3 fertig"

echo "[$(date +%T)] B4: Re-ID unter Option B (C=p95), adaptiv, beide Korpora"
$PY scripts/train_reid.py --dataset mitbih --data-split ds1 --dp-sweep \
    --threat-model adaptive_attacker --mechanisms laplace --epsilons 1.0 4.0 20.0 \
    --seeds $SEEDS --clip $C95 --sensitivity $S95 --gpu-memory-gb 7 --skip-existing \
    --out-dir $ROOT/reid_mitbih_C_p95 > $LOG/reid_mitbih.log 2>&1 &
$PY scripts/train_reid.py --dataset ecgid --dp-sweep \
    --threat-model adaptive_attacker --mechanisms laplace --epsilons 1.0 4.0 20.0 \
    --seeds $SEEDS --clip $C95 --sensitivity $S95 --gpu-memory-gb 7 --skip-existing \
    --out-dir $ROOT/reid_ecgid_C_p95 > $LOG/reid_ecgid.log 2>&1 &
wait
echo "[$(date +%T)] ALLES FERTIG"
