#!/usr/bin/env bash
# Ergaenzt den naiven (clean_attacker) Angreifer um Seed 2026, damit er dieselben
# drei Seeds hat wie der adaptive: {42, 1337, 2026}. Erst dann ist das Attacker-Band
# seed-gepaart und die Angabe im Paper korrekt.
# Wartet, bis der Option-B-Sweep fertig ist (GPU frei).
set -u
PY="${PY:-python}"
EPS="0.05 0.1 0.15 0.175 0.2 0.225 0.25 0.3 0.35 0.4 0.5 0.75 1.0 1.5 2.0 4.0 8.0 20.0"
LOG=results/option_b/logs

echo "[$(date +%T)] warte auf Option-B-Abschluss ..."
until grep -qa "ALLES FERTIG" $LOG/driver.log 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] Option B fertig, GPU frei. Starte naiven Angreifer, Seed 2026."

$PY scripts/train_reid.py --dataset mitbih --data-split ds1 --dp-sweep \
    --threat-model clean_attacker --epsilons $EPS --seeds 2026 \
    --gpu-memory-gb 9 --skip-existing --out-dir results/reid \
    > $LOG/naive_2026_mitbih.log 2>&1 &
$PY scripts/train_reid.py --dataset ecgid --dp-sweep \
    --threat-model clean_attacker --epsilons $EPS --seeds 2026 \
    --gpu-memory-gb 9 --skip-existing --out-dir results/reid_ecgid \
    > $LOG/naive_2026_ecgid.log 2>&1 &
wait
echo "[$(date +%T)] NAIV-2026 FERTIG"
