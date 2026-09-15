# DP-ECG: Differential Privacy on Raw ECG Signals

Code and aggregated results for the paper

> **The Missing Sweet Spot: Quantifying the Privacy and Utility Cost of Sharing
> Raw ECG.** Proceedings of the Hawaii International Conference on System
> Sciences (HICSS-60), Advancing Mobile Health (m-Health) Technologies minitrack.

The manuscript is not distributed here; see [Citation](#citation) for where to
find it. Repository: <https://github.com/WIMUniCologne/dp-ecg-privacy-utility>.

Empirical study of the privacy-utility trade-off when data-level differential
privacy is applied directly to raw electrocardiogram signals, rather than to
model gradients. The question: if a custodian shares ECG perturbed at the source,
so that the guarantee attaches to the signal itself rather than to a system the
recipient has to trust, how much clinical utility survives and how much identity
still leaks? We map that across three mechanisms, fifteen privacy budgets, two
corpora, two classifiers and a ladder of attackers of increasing strength, with
eight seeds behind every reported operating point.

Leakage is probed along two axes, **identity** (re-identification) and
**demographics** (attribute inference), and the utility side is cross-checked
with a second, architecturally distant classifier.

## What this package measures

Core privacy–utility trade-off:

1. **Utility** — A CNN/seq2seq classifier (re-implementation of Mousavi &
   Afghah, 2019) trained on MIT-BIH DS1 and evaluated on DS2 under the
   inter-patient split. No-DP baseline: macro-F1 ≈ 0.95.
2. **Privacy mechanisms** — Three additive DP mechanisms (Laplace, bounded
   Laplace, analytic Gaussian) applied per sample to z-normalized beats across
   an epsilon grid. Sensitivity Δ = 0.29 (95th-percentile jump sensitivity;
   see `notebooks/00_sensitivity_calibration.ipynb`).
3. **Re-identification attacker** — A deliberately generic 1D-CNN identifier
   (FCN-style backbone, Wang et al. 2017; see `src/models/reid_cnn.py` for the
   lineage and why an untuned model is the right choice here), trained per
   configuration and evaluated as
   top-1 identification accuracy. Two attacker strengths bracket the leak:
   *naive* (trains on clean signals, tested on DP) and *adaptive* (trains on the
   same DP distribution it attacks — the worst case).
4. **Trade-off map** — Utility (macro-F1) against re-identification accuracy,
   the privacy–utility plane that is the paper's central figure.
5. **Cross-database robustness check** — The whole map is computed on two
   corpora with deliberately different identification structure:
   MIT-BIH (intra-record: enrollment and probe beats from the same recording)
   and ECG-ID (cross-session: enrollment and probe from different sessions,
   the harder and more realistic test). This checks whether the trade-off is an
   artifact of the easy intra-record setting or survives a genuine session
   split.

Stress tests and robustness checks (this package, reproduced in notebooks 05–07):

6. **Temporal-correlation stress test** — A temporally-aware *denoising*
   adversary low-pass-filters the released (noised) window before
   re-identifying, exploiting the fact that the ECG is autocorrelated while the
   i.i.d. DP noise is not. The privacy mechanism is unchanged; only the attacker
   is stronger. Quantifies how much identity leakage smarter post-processing
   recovers.
7. **Attribute inference — the second leakage axis** — A second classifier
   predicts demographic attributes (sex; a 2-band age on ECG-ID) from the
   protected signal, **subject-disjoint**, to test whether records counted as
   re-identification-protected still leak demographics, and whether that leakage
   falls slower than identity.
8. **Second utility model** — A classical Random Forest on beat-morphology
   features re-runs the utility sweep, to check that the L-shaped utility curve
   is a property of the data and the mechanism, not of the one seq2seq
   architecture.

## Key findings

- **The strict corner is empty.** Against the strongest attacker we test, no
  configuration is at once clinically strong and safe under the strict criterion
  (macro-F1 >= 0.85 and cross-session re-identification <= 0.20). The best point
  that holds leakage at or below 0.20 reaches 0.774, falling 0.076 short on
  utility.
- **The relaxed corner is grazed, not empty.** Give the recipient the same
  denoising front end the attacker gets, and two configurations meet the relaxed
  criterion (>= 0.80, <= 0.25) on their point estimates: Laplace at eps = 0.15
  holds macro-F1 0.820 (0.800-0.840) against 0.245 leakage (0.237-0.254), with
  bounded Laplace beside it at 0.813 against 0.246. Both intervals touch their
  bar, so we read this as a boundary being grazed rather than a sweet spot found.
- **Strengthening one side only flatters the result.** A recipient given the
  attacker's denoiser gains 0.071 macro-F1 on average across all eighteen
  configurations in the operating band. Any study that hardens the attacker but
  not the recipient overstates the cost of sharing.
- **Attacker strength dominates the measured risk.** On ECG-ID the ladder runs
  naive 0.05-0.06 -> adaptive 0.18-0.22 -> low-pass denoising 0.24-0.30 ->
  learned denoising 0.27-0.36 -> a stronger identifier 0.21-0.33 -> the last two
  combined 0.31-0.42, which is 4.7 to 6.5 times the empirical floor. Reported
  leakage is a *lower* bound on adversary capability.
- **What differs between settings is the ceiling, not whether the attacker
  matters.** The naive attacker never rises more than one point above the
  majority-class floor anywhere in the operating band, while the adaptive
  attacker reaches three to four times its accuracy. A capable adversary climbs
  to 0.77 within a recording and 0.21 across sessions.
- **Three seeds are not enough.** Moving the operating points from three seeds to
  eight shifts leakage by up to 0.06, more than several differences worth
  interpreting. Every headline number carries a confidence interval.
- **Chance is the majority-class rate, not 1/n.** ECG-ID subjects contribute
  between 19 and 209 windows each, so the window-weighted floor is 0.065 rather
  than 1/89 = 0.011; subject-averaging restores 1/n. At maximum noise the
  attacker collapses onto each of them (0.0650 and 0.0113 measured).
- **The demographic channel closes before identity does.** With class balancing,
  sex is read from unprotected ECG-ID at 0.64 against a 0.50 chance level. Noise
  in the operating band closes it (0.47 at eps = 0.25) while identity stays open
  at three times its floor, which is why identity is the primary axis.
- **Amplitude clipping is free in utility but is not a defence.** Clipping
  affects 96.6 % of beats yet lifts macro-F1 slightly, to 0.968 from 0.950, and
  it does reduce re-identification against the adaptive attacker -- but a
  temporally aware attacker recovers as much under clipping as without it.

## Repository layout

    configs/
      config.py            Central config: Δ=0.29, epsilon grid, seeds. Operating
                           points additionally use seeds 7/123/2024/31337/99991.
    src/
      data/                MIT-BIH + ECG-ID loaders, preprocessing, qsPeaks
                           fiducial detection, beat extraction, Re-ID splits,
                           attributes.py (subject-disjoint sex/age labels).
      dp/                  DP mechanisms: reference + vectorized (KS-verified),
                           apply.py (perturbation, optional amplitude clipping),
                           denoise.py (temporal-correlation denoisers).
      models/              Seq2seq classifier, training pipeline, Re-ID CNN,
                           reid_resnet.py (stronger second attacker architecture),
                           denoise_ae.py (denoising autoencoder),
                           ecg_features.py (beat-morphology features for the RF).
      utils/               GPU helpers, atomic I/O.
    scripts/
      train_baseline.py    Train the classifier without DP.
      train_with_dp.py     Multi-seed DP sweep (per-mechanism delta); --clip and
                           --perturb-test for the revision variants.
      train_reid.py        Re-ID attacker; --dataset {mitbih,ecgid}, naive/adaptive,
                           --arch {cnn,resnet}, --clip for unbounded adjacency.
      train_reid_denoise.py  Denoising (temporally-aware) Re-ID attacker.
      train_attribute.py   Subject-disjoint attribute-inference attacker.
      train_utility_rf.py  Second utility model (Random Forest on features).
      run_parallel_sweep.py  Shard the sweep across GPU workers.
      diagnose/            Sanity-check scripts (DP distributions, beat counts,
                           denoiser sanity, attribute-label verification).
    notebooks/
      _loader.py                       Shared multi-seed result loader.
      00_sensitivity_calibration.ipynb  Δ=0.29 derivation.
      01_dp_sweep_results.ipynb         Utility vs ε (mean ± std, ghost dots).
      02_money_figure.ipynb             Privacy–utility trade-off, MIT-BIH.
      03_reid_robustness_ecgid.ipynb    Cross-database Re-ID (MIT-BIH vs ECG-ID).
      04_attacker_band.ipynb            Naive vs adaptive attacker band.
      05_denoise_attacker.ipynb         Adaptive vs denoising attacker.
      06_attribute_leakage.ipynb        Attribute inference vs re-ID (second axis).
      07_second_utility.ipynb           seq2seq vs Random Forest utility.
      paper_figures.ipynb               Figures as they appear in the paper.
    tests/                 TF-free unit tests (denoisers, features, label parsing,
                           subject-disjoint split).
    data/                  Empty. See data/README.md for how to obtain the two
                           corpora from PhysioNet; no recordings are distributed
                           with this repository.
    results/               Aggregated CSV summaries and figures. Per-run metrics
                           land here too but are not versioned.
    run_all_experiments.sh   Full pipeline orchestrator (utility + adaptive Re-ID).
    run_naive_attacker.sh    Naive-attacker pass for the attacker band.
    run_denoise_attacker.sh  Temporal-correlation stress test (section 6).
    run_attribute_inference.sh  Attribute-inference sweep (section 7).
    run_second_utility.sh    Random Forest utility sweep (section 8).
    scripts/revision/        The runs behind the revision, with a README of
                             their own; each is invoked from the repository
                             root and passes --skip-existing.
    environment.yml          conda environment, pinned to the versions used.
    requirements.txt         pip equivalent.
    CITATION.cff             machine-readable citation.

    New CLI flags added during revision (all optional, all additive):
      --clip C          clamp samples to [-C, C] before the mechanism (section 10)
      --sensitivity S   override Δ (sections 10 and 13)
      --arch resnet     stronger second attacker architecture (section 11)
      --perturb-test    apply the mechanism to the test set too (section 12)
      --denoise M       give the recipient the attacker's denoising front end
      --denoise-fs F    effective rate of the beat representation in Hz; beats
                        are time-normalised to 280 points, so this is not the
                        beat length (default 310)

## Setup

With conda, which is how the reported results were produced:

    conda env create -f environment.yml
    conda activate dp-ecg

Or with pip, on Python 3.12:

    pip install -r requirements.txt

Both files pin the exact versions used: TensorFlow 2.18, NumPy 2.0.2,
scikit-learn 1.6.1, wfdb 4.3.0, diffprivlib 0.6.5. A GPU is needed only for the
seq2seq classifier and the deep attackers; the feature-based second utility
model and the sanity checks run on CPU.

MIT-BIH and ECG-ID are fetched from PhysioNet on first use, so no manual
download is required. See [`data/README.md`](data/README.md) for the corpora,
their licence terms and how to cite them.

## Reproducing the experiments

The core pipeline can be run end to end:

    bash run_all_experiments.sh    # utility sweep + adaptive Re-ID on both corpora
    bash run_naive_attacker.sh     # naive-attacker pass (for the attacker band)

The stress tests and robustness checks are added by:

    bash run_denoise_attacker.sh      # section 6: temporally-aware attacker
    bash run_attribute_inference.sh   # section 7: attribute inference (run the gate first)
    bash run_second_utility.sh        # section 8: Random Forest utility (no GPU needed)

Or step by step, as below.

### Core privacy–utility trade-off

#### 1. Baseline (no DP)

    python scripts/train_baseline.py

Results in `results/baseline/metrics.pkl`. Expected macro-F1 ≈ 0.95, with
per-class F1 of N ≈ 0.99, V ≈ 0.99, and the harder S class ≈ 0.86.

Optional — reproduce with Mousavi's preprocessed `.mat` file:

    python scripts/train_baseline.py \
        --data-source mat --mat-path data/s2s_mitbih_aami_DS1DS2.mat

#### 2. DP utility sweep

One model per (mechanism, ε, seed):

    python scripts/train_with_dp.py

Defaults sweep {laplace, laplace_bounded, gaussian_analytic} over the epsilon
grid (0.05–20) × 3 seeds, with per-mechanism delta (Laplace δ=0; bounded Laplace
and Gaussian δ=1e-5). Override for a quick run:

    python scripts/train_with_dp.py --mechanisms laplace \
        --epsilons 0.5 1.0 2.0 --seeds 42

Results at `results/dp/<mechanism>/eps_<e>_delta_<d>/seed_<s>/metrics.pkl`.
Resumable with `--skip-existing`.

#### 3. Parallel sweep (faster)

Each training run uses only a fraction of a high-end GPU, so several share one
device. Shard the grid across workers:

    python scripts/run_parallel_sweep.py --workers 3 --epochs 200

Logs go to `results/dp/parallel_logs/`.

#### 4. Re-identification attacker

The same 1D-CNN attacker is run at **two strengths**, which together bracket how
much identity an adversary can recover. They differ only in what the attacker
trains on:

  - **Naive attacker** — trains on clean (un-noised) signals, then is tested on
    the DP-perturbed signals it has never seen. This models an adversary who
    does not know the data was protected, or how. It is the *lower* bound on
    leakage.
  - **Adaptive attacker** — trains on data drawn from the same DP-perturbed
    distribution it will attack. This models the worst case, where the adversary
    fully anticipates the defense. It is the *upper* bound on leakage.

The gap between the two is the *attacker band*. It is wide in the intra-record
setting (MIT-BIH) — at ε ≈ 0.25 the naive attacker sits near the floor while the
adaptive one is near total recovery — and narrower in absolute terms
cross-session (ECG-ID). The narrowness is absolute, not relative: on ECG-ID the
naive attacker never rises more than one point above the majority-class floor
anywhere in the operating band while the adaptive attacker reaches three to four
times its accuracy, so
attacker strength dominates the measured risk in *both* settings. What differs
is the ceiling a capable adversary reaches. This is why a privacy claim must
state which attacker it was measured against.

The adaptive pass is produced by the main pipeline; the naive pass is a separate
run. Both are wrapped by the shell scripts; the underlying entry point is
`scripts/train_reid.py` (run with `--help` for the current flags).

    bash run_all_experiments.sh   # includes adaptive Re-ID on MIT-BIH and ECG-ID
    bash run_naive_attacker.sh    # adds the naive-attacker pass

Floors: use the **majority-class rate** an attacker reaches by naming the most
frequent subject — 0.045 on MIT-BIH and 0.065 on ECG-ID — rather than a uniform
1/n. On MIT-BIH the two coincide because records are equally long; on ECG-ID
they do not, because subjects contribute between 19 and 209 windows.
Results are saved per configuration under
`results/reid/<dataset>/<mechanism>/eps_<e>_delta_<d>/<threat_model>/`.

#### 5. Figures and analysis

    jupyter notebook notebooks/01_dp_sweep_results.ipynb       # utility curves, heatmap, sanity checks
    jupyter notebook notebooks/02_money_figure.ipynb           # privacy–utility trade-off (MIT-BIH)
    jupyter notebook notebooks/03_reid_robustness_ecgid.ipynb  # cross-database check: MIT-BIH (intra-record) vs ECG-ID (cross-session)
    jupyter notebook notebooks/04_attacker_band.ipynb          # naive vs adaptive attacker band, both corpora

Notebook `03` is the cross-database robustness check: it re-runs the
re-identification sweep on ECG-ID's session split and plots it against MIT-BIH's
intra-record curves, mechanism by mechanism. The point is to show that the
mechanism ordering and the qualitative trade-off hold under a harder
identification protocol, while making the difference in absolute leakage
explicit.

These regenerate the paper figures and the CSV summaries
(`results/dp_sweep_summary.csv`, `results/privacy_utility_summary.csv`,
`results/privacy_utility_summary_ecgid.csv`).

### Robustness and stress tests

#### 6. Temporal-correlation stress test (denoising attacker)

The DP mechanisms add *independent* noise to each sample of a z-normalized
window. An ECG window is smooth and strongly autocorrelated, so a temporally
aware adversary can low-pass / denoise the released window to average much of
that noise away while leaving the signal — the signal is correlated across time,
the i.i.d. noise is not. This stress test measures that leakage **without
changing the privacy mechanism**: the released (noised) data is held fixed and
the attacker simply gains a denoising front end (same data, smarter adversary).

The denoiser is applied identically to the attacker's **train and test** inputs,
so it is part of the adaptive protocol end to end (not bolted on at evaluation).
The comparison, at matched (mechanism, ε, seed), is:

    adaptive                -> train re-ID on DP windows,           eval on DP windows
    adaptive_denoise_<...>  -> train re-ID on denoise(DP windows),  eval on denoise(DP windows)

The gap is the leakage recovered by exploiting temporal correlation. Two
attacker fronts: a fixed low-pass filter (Savitzky-Golay, moving-average, or
Gaussian; `src/dp/denoise.py`) and an optional learned 1D denoising autoencoder
(`src/models/denoise_ae.py`). Window lengths are chosen from the sampling rate
so the QRS complex is preserved — **always run the sanity check first**:

    # original / noised / denoised overlays + noise-removal and QRS-preservation numbers
    python scripts/diagnose/plot_denoise_sanity.py --dataset mitbih --method savgol

    # representative sweep {0.1, 0.25, 0.5, 1.0} x 3 mechanisms x 3 seeds, both corpora,
    # with a matched no-denoise number computed in the same run:
    bash run_denoise_attacker.sh

    # or directly, e.g. the stronger learned attacker on ECG-ID:
    python scripts/train_reid_denoise.py --dataset ecgid --attack autoencoder \
        --compare-adaptive --out-dir results/reid_ecgid

Results land in the existing layout under new threat-model folders that do not
collide with prior runs:
`results/reid[/_ecgid]/dp/<mech>/eps_<e>_delta_<d>/adaptive_denoise_<method>/seed_<s>/metrics.pkl`.
Each record stores `matched_adaptive_accuracy` and `denoise_gain` when
`--compare-adaptive` is set. Seeds, splits, identities, and per-config DP draws
are identical to the adaptive attacker in `train_reid.py`, so the numbers are
directly comparable. Render with `notebooks/05_denoise_attacker.ipynb`.

#### 7. Attribute inference — the second leakage axis

A record counted as re-identification-protected may still leak demographic
attributes. This experiment trains a second classifier to predict **sex** (and,
on ECG-ID, a 2-band **age**) from the DP-protected signal, over the same ε grid,
mechanisms, and seeds as the re-ID attacker, so the two leakage curves can be
laid side by side. The architecture is the same generic 1D-CNN as the re-ID
attacker (only the output head changes), avoiding a new "why this model" point.

The split is **subject-disjoint** (no subject in train and test). This is
essential: the attribute is constant across a subject's windows, so any subject
overlap would let the model read the attribute off memorised identity — disguised
re-identification. This is a different split from the within-record Re-ID split.

Always run the label gate first — it parses the real headers and prints
availability and balance, and is the basis for the per-corpus decisions:

    python scripts/diagnose/verify_attribute_labels.py     # the GATE
    bash run_attribute_inference.sh                        # ECG-ID sex+age, MIT-BIH sex

Decisions encoded after verification: **MIT-BIH** runs sex only (47 subjects;
age too thin/skewed to evaluate honestly); **ECG-ID** runs sex and 2 age bands
(median split). Every run reports the majority-class **chance floor** (the real
test balance, not 1/n) and the **no-privacy anchor** (accuracy on the unprotected
signal). Output: `results/attr/<dataset>/<attribute>/...`. Render with
`notebooks/06_attribute_leakage.ipynb`. The finding to look for: at the ε that
pushes re-ID onto the operating point, demographic accuracy is still well above
its floor — a second, slower-falling leakage. (If it falls just as fast, that is
also reported, as a cleaner closing of the concern.)

#### 8. Second utility model — robustness of the L-shape

Robustness insurance, not a SOTA claim: is the L-shaped utility curve a property
of the one deep architecture (Mousavi & Afghah seq2seq), or does a generic,
architecturally distant model show the same decline and ceiling? A classical
**Random Forest** on beat-morphology features (`src/models/ecg_features.py`) is
run over the same ε grid, mechanisms, seeds, data, inter-patient split, SMOTE,
and macro-F1 — only the architecture differs. The features are computed from the
same DP-protected beat the seq2seq model consumes (raw RR/timing is deliberately
excluded: it lies outside that protected representation and is not touched by the
amplitude DP, so including it would break the identical-protection requirement).
The RF is **untuned** by design — a generic second model reproducing the shape is
the evidence; visible tuning would invite the suspicion that the L-shape was
engineered.

    bash run_second_utility.sh        # RF over the full DP sweep (no GPU needed)

Output: `results/utility_rf/...` (mirrors the seq2seq utility layout). Render
with `notebooks/07_second_utility.ipynb`, which overlays the RF curve on the
seq2seq curve per mechanism. If both trace the same monotone decline under a
shared ceiling, the L-shape is not an architecture artifact; a different shape
would itself be a reportable, more nuanced result.

### Revision experiments

Five further experiments were added during revision. All are additive: they use
new CLI flags and write to separate result roots, so existing results are
untouched.

#### 9. Seed robustness — why three seeds are not enough

Three seeds turned out to be too few for the operating points: at eight seeds
the cross-session leakage moves by up to 0.06, more than several differences the
analysis had previously interpreted. The operating band was therefore re-run
with five additional seeds:

    python scripts/train_with_dp.py --seeds 7 123 2024 31337 99991 \
        --epsilons 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.75 --skip-existing
    python scripts/train_reid.py --dataset ecgid --dp-sweep \
        --threat-model adaptive_attacker --seeds 7 123 2024 31337 99991 \
        --epsilons 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.75 --skip-existing

All headline numbers are reported as means with 95 % confidence intervals over
the eight seeds.

#### 10. Unbounded adjacency — pricing the stricter reading of DP

The pipeline calibrates noise from a jump statistic, which corresponds to a
*metric* DP adjacency (two recordings differ in one sample by at most Δ). The
stricter unbounded reading, where a sample may take any value, requires clipping
each sample to `[-C, C]` and calibrating to `2C`. The new `--clip` flag applies
that clipping before the mechanism:

    # C = 95th percentile of |x| on the z-normalised signal; sensitivity = 2C
    python scripts/train_with_dp.py --mechanisms laplace --clip 2.0812 \
        --sensitivity 4.1624 --epsilons 1 2 4 8 20 50 \
        --out-dir results/option_b/util/C_p95

**Result:** clipping costs no utility — it affects 96.6 % of beats yet leaves
macro-F1 at 0.968, because the diagnostic information is in beat shape rather
than peak amplitude. The stricter reading therefore reproduces the same curves
with the ε axis relabelled by 14.3×: the operating point called ε = 0.25 becomes
ε ≈ 3.6 on identical released data. ε depends on the adjacency one declares;
measured attacker success does not.

#### 11. A second attacker architecture

`--arch resnet` swaps the generic identifier for the ResNet baseline of
Wang et al. (2017), implemented in `src/models/reid_resnet.py`. It exists to
test rather than assert the claim that our figures are a lower bound:

    python scripts/train_reid.py --dataset ecgid --arch resnet --dp-sweep \
        --threat-model adaptive_attacker --epsilons 0.25 0.3 0.5 0.75 \
        --out-dir results/reid_ecgid_resnet

**Result:** on ECG-ID the ResNet gains +0.08 to +0.17 over the generic CNN; on
MIT-BIH it does not, because the CNN is already near ceiling there. Combined
with the learned denoiser (`--attack autoencoder --arch resnet` in
`train_reid_denoise.py`) it reaches 0.31–0.42 across sessions, 4.7 to 6.5 times
the empirical floor. The two gains are
complementary: denoising exploits temporal correlation, the deeper network
representational capacity.

#### 12. Utility with a protected test set

By default the mechanism is applied to the training beats only, and the model is
evaluated on unprotected DS2 — the sharing scenario, where a recipient trains on
released data and applies the model to their own recordings. `--perturb-test`
answers the other reading, in which the recipient only ever sees protected
signal:

    python scripts/train_with_dp.py --perturb-test --out-dir results/dp_ptest

**Result:** across 18 configurations the difference ranges from −0.041 to
+0.047 with no systematic direction, so the utility definition is not
load-bearing.

#### 13. ECG-ID with its own sensitivity

Δ = 0.29 is calibrated on MIT-BIH and reused on ECG-ID, whose own 95th-percentile
jump sensitivity is 0.262:

    python scripts/train_reid.py --dataset ecgid --dp-sweep --sensitivity 0.262 \
        --out-dir results/reid_ecgid_delta262

**Result:** the smaller Δ means less noise and hence *higher* leakage (+0.02 to
+0.11). Reusing 0.29 over-protects ECG-ID slightly, so the reported figures are
conservative in that respect.

### Sanity checks

    # Vectorized mechanisms vs diffprivlib reference distributions (KS test)
    python scripts/diagnose/verify_dp_mechanisms.py

    # Beat extraction vs Mousavi's per-patient MATLAB counts
    python scripts/diagnose/diagnose_qspeaks_counts.py

    # Denoiser preserves the QRS while removing i.i.d. noise (section 6)
    python scripts/diagnose/plot_denoise_sanity.py --dataset mitbih --method savgol

    # Attribute-label availability and class balance (section 7 gate)
    python scripts/diagnose/verify_attribute_labels.py

### Tests

The `tests/` suite runs without TensorFlow or PhysioNet data (synthetic signals
and the real header formats), covering the denoisers, the beat features, the
demographic-label parsing, and the subject-disjoint split invariant:

    python tests/test_denoise.py
    python tests/test_attributes_and_features.py

## Methodology notes

- **Inter-patient split.** AAMI EC57 / de Chazal et al. (2004): DS1 train,
  DS2 test, no patient in both.
- **Beat extraction.** T-wave to T-wave via qsPeaks fiducial detection,
  resampled to 280 samples (MATLAB-exact to Mousavi's pipeline).
- **Sequence structure.** 10-beat pure-class sequences.
- **Class imbalance.** SMOTE oversampling (S → 7000, V → 6000 beats) on the
  training set only, with seed propagated so runs are reproducible.
- **DP application.** Independent noise added to each z-normalized beat sample,
  after preprocessing and beat extraction but before SMOTE. Labels are never
  perturbed.
- **What the guarantee is.** Δ = 0.29 is the 95th percentile of absolute first
  differences over pooled DS1+DS2 (`notebooks/00_sensitivity_calibration.ipynb`).
  This is a *jump* statistic, not a global sensitivity bound. The guarantee is
  stated as **metric differential privacy** (Chatzikokolakis et al., 2013): two
  recordings are adjacent when they differ in one sample by at most Δ, under
  which the L1 sensitivity of releasing the signal is Δ by construction. It is a
  per-sample statement; composing over the 720–650,000 samples of a window or a
  recording gives Tε and is vacuous at recording level, so no per-individual
  bound is claimed. The z-normalisation uses statistics of the *unprotected*
  recording, which are not themselves privatised. See section 10 for what the
  stricter unbounded reading would cost.
- **Chance level.** The floor is the majority-class rate an attacker reaches by
  naming the most frequent subject — 0.065 on ECG-ID and 0.045 on MIT-BIH — not
  a uniform 1/n. ECG-ID subjects contribute between 19 and 209 windows each.
- **Utility semantics.** Utility is measured with the mechanism applied to the
  training beats only and evaluation on the unprotected test half: the value the
  shared data retains *for a recipient*, not the fidelity of the protected
  waveform. Section 12 shows the alternative reading changes nothing. Attacker
  training and evaluation data are both protected, which follows from the threat
  model.
- **Model selection.** Weights are chosen by best test-set score in both the
  utility and the attacker pipelines. This flatters both axes rather than one.
- **Denoising attacker (section 6).** The denoiser is adversary-side
  post-processing on the *released* data; it does not change the privacy
  mechanism, and it is applied identically to the attacker's train and test
  inputs so the comparison is between two adaptive attackers.
- **Attribute inference (section 7).** Evaluated **subject-disjoint** (no
  subject in both halves) so the measurement is attribute leakage and not
  disguised re-identification; chance is the test majority-class rate, with the
  unprotected-signal accuracy as the upper anchor.
- **Second utility model (section 8).** Identical protocol to the seq2seq
  utility sweep (split, DP point, SMOTE, macro-F1); only the architecture
  differs, and it is left untuned to demonstrate robustness rather than best
  performance.
- **Reproducibility.** Seeds control shuffling, oversampling, initialization and
  noise draws, derived deterministically per configuration. Eight seeds stand
  behind the utility sweep across the operating band and behind every ECG-ID
  re-identification run; nine behind the unprotected references; three outside
  the band and at the intermediate rungs of the attacker ladder. We moved from
  three to eight after finding that three could not resolve differences the
  analysis interpreted (section 9). Headline numbers carry 95 % confidence
  intervals.

## Citation

If you use this code, please cite the paper:

> Rahlmeier, T., Wolf, S., Beck, R., Bui, C. M., & Schoder, D. The Missing Sweet
> Spot: Quantifying the Privacy and Utility Cost of Sharing Raw ECG. In
> *Proceedings of the Hawaii International Conference on System Sciences
> (HICSS-60)*.

A machine-readable form is in [`CITATION.cff`](CITATION.cff). HICSS proceedings
are published in the University of Hawai'i ScholarSpace repository, where the
paper is openly available.

Please also cite the two corpora and PhysioNet; the references are in
[`data/README.md`](data/README.md).

## References

- Mousavi, S., & Afghah, F. (2019). Inter- and intra-patient ECG heartbeat
  classification for arrhythmia detection: a sequence to sequence deep learning
  approach. *ICASSP 2019*, 1308–1312.
- Wang, Z., Yan, W., & Oates, T. (2017). Time series classification from
  scratch with deep neural networks: A strong baseline. *IJCNN 2017*, 1578–1585.
- Donida Labati, R., Muñoz, E., Piuri, V., Sassi, R., & Scotti, F. (2019).
  Deep-ECG: Convolutional neural networks for ECG biometric recognition.
  *Pattern Recognition Letters*, 126, 78–85.
- Zhang, Y., Xiao, Z., Guo, Z., & Wang, Z. (2019). ECG-based personal
  recognition using a convolutional neural network. *Pattern Recognition
  Letters*, 125, 668–676.
- Chatzikokolakis, K., Andrés, M. E., Bordenabe, N. E., & Palamidessi, C.
  (2013). Broadening the scope of differential privacy using metrics.
  *PETS 2013*, 82–102.
- de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification
  of heartbeats using ECG morphology and heartbeat interval features.
  *IEEE Transactions on Biomedical Engineering*, 51(7), 1196–1206.
- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia
  Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
- Lugovaya, T. S. (2005). Biometric human identification based on
  electrocardiogram. Master's thesis (ECG-ID Database, PhysioNet).

## License

See `LICENSE`. Note that the MIT-BIH and ECG-ID data are governed separately by
the PhysioNet Credentialed Health Data Use Agreement / license terms; this
repository does not redistribute the raw recordings (they are downloaded from
PhysioNet on first run).
