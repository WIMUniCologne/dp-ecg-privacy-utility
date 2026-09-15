#!/usr/bin/env bash
# A2 neu, mit korrigierter effektiver Abtastrate (310 Hz statt der faelschlich
# uebergebenen Beat-Laenge 280). Die Kontrolle ueber 3 Konfigurationen x 3 Seeds
# ergab systematisch +0.020 Utility (95%: +0.009 bis +0.031, 9/9 positiv), also
# muss der ganze Satz neu. Ein Mechanismus je Prozess, damit die GPU reicht.
set -u
PY="${PY:-python}"
L=results/option_b/logs; mkdir -p $L
S8="42 1337 2026 7 123 2024 31337 99991"
BAND="0.15 0.2 0.25 0.3 0.5 0.75"
# Gaussian zuerst und allein gestartet: er bestimmt die Decke, weil er am
# wenigsten leakt, und entscheidet damit ob die relaxed corner leer bleibt.
# Hoechstens zwei gleichzeitig: ein anderer Nutzer haelt 9.5 der 24 GB, und drei
# Prozesse a 6 GB sind zweimal zuverlaessig ins OOM gelaufen.
run(){ $PY scripts/train_with_dp.py --mechanisms "$1" --epsilons $BAND --seeds $S8 \
      --denoise savgol --denoise-fs 310 --epochs 200 --eval-every 20 \
      --gpu-memory-gb 5 --skip-existing --out-dir results/dp_denoise_fs310 \
      > "$L/a2fs310_$1.log" 2>&1; }
run gaussian_analytic &
run laplace &
wait
run laplace_bounded
echo "[$(date +%F' '%T)] A2 mit korrigierter Rate fertig"
