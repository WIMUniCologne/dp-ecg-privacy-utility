# Revision experiments

The runs behind the revision of *The Missing Sweet Spot*. Each script is invoked from
the repository root and writes into `results/`; all of them pass `--skip-existing`, so a
restart is cheap and no completed run is repeated.

Set `PY` to the interpreter of your environment (defaults to `python`):

    PY=/path/to/python bash scripts/revision/run_overnight.sh

| Script | What it produces |
|---|---|
| `run_option_b.sh`, `run_option_b_fill.sh` | Strict-adjacency variant: clipping at C, calibration to 2C, for three values of C |
| `run_overnight.sh` | Seed robustness over the operating band (8 seeds), utility on a protected test set, ResNet attacker |
| `run_all_remaining.sh` | Learned denoiser, balanced attribute inference, second utility model |
| `run_combo.sh` | Learned denoiser combined with the ResNet identifier — the strongest attacker reported |
| `run_naive_seed2026.sh`, `run_pin_down.sh` | Fills gaps in the seed grid so every operating point rests on the same seed set |
| `run_fill_eps0225.sh`, `run_fill_strongest.sh`, `run_finish_gaps.sh` | Further gap fills: the skipped budget in the operating band, the strongest attacker, the naive attacker |
| `run_a2_recipient_denoise.sh`, `run_a2_fill_seeds.sh`, `run_finish_a2.sh` | Gives the recipient the attacker's denoising front end, then raises it to eight seeds |
| `analyze_overnight.py` | Aggregates the above into the operating-point table and its confidence intervals |
| `analyze_option_b.py` | Compares the strict-adjacency variant against the reported curves |
| `rebuild_attacker_band.py` | Regenerates `results/attacker_band_summary.csv` from the raw metrics, headless |
| `run_strongest_full_band.sh` | The strongest attacker over the whole operating band at eight seeds |
| `run_ecgid_fixed_split.sh` | Every ECG-ID re-identification run again, after the session-ordering fix |
| `run_a2_fs_check.sh`, `run_a2_fs_corrected.sh` | The recipient denoiser at the correct effective rate: first a three-configuration control, then the whole band |
