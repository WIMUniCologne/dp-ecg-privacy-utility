#!/usr/bin/env bash
# Fuellt eps=0.225 auf acht Seeds auf. Dieser eine Wert fehlte im Nachtlauf,
# obwohl die Utility-Decke des Papers daran haengt; alle uebrigen Werte des
# Betriebsbands stehen bereits auf acht Seeds.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
NEW="7 123 2024 31337 99991"
E=0.225
banner(){ echo "[$(date +%F' '%T)] === $1 ==="; }

banner "Utility bei eps=$E, +5 Seeds (3 Mechanismen x 5 = 15 Laeufe)"
for M in laplace laplace_bounded gaussian_analytic; do
  $PY scripts/train_with_dp.py --mechanisms $M --epsilons $E --seeds $NEW \
      --epochs 200 --eval-every 20 --gpu-memory-gb 7 --skip-existing \
      > $L/fill_util_$M.log 2>&1 &
done
wait

banner "Re-ID adaptiv bei eps=$E, +5 Seeds (2 Korpora x 3 x 5 = 30 Laeufe)"
$PY scripts/train_reid.py --dataset mitbih --data-split ds1 --dp-sweep \
    --threat-model adaptive_attacker --epsilons $E --seeds $NEW \
    --gpu-memory-gb 10 --skip-existing --out-dir results/reid > $L/fill_reid_mitbih.log 2>&1 &
$PY scripts/train_reid.py --dataset ecgid --dp-sweep \
    --threat-model adaptive_attacker --epsilons $E --seeds $NEW \
    --gpu-memory-gb 10 --skip-existing --out-dir results/reid_ecgid > $L/fill_reid_ecgid.log 2>&1 &
wait

banner "fertig"
