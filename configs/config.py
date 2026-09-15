"""
Central configuration for DP-ECG experiments.

Edit values here to change defaults. Most scripts also accept CLI overrides
for the experiment-specific values (epochs, batch size, output paths).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

MITBIH_DIR = DATA_DIR / "mit-bih-arrhythmia"
ECGID_DIR = DATA_DIR / "ecg-id-database"


# -----------------------------------------------------------------------------
# DP configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DPConfig:
    """Differential privacy parameters for the utility sweep."""

    # Finer epsilon grid in the critical 0.1-1.0 region where utility and
    # re-identification both transition (as shown by the first sweep).
    epsilons: Tuple[float, ...] = (
        0.05, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.5,
        0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 20.0,
    )

    # One fixed delta per mechanism (standard literature choice).
    # delta << 1/n for n ~ 50k beats; 1e-5 satisfies this with margin.
    # Laplace is (eps, 0)-DP and ignores its delta argument; we keep
    # delta=0.0 for it in the sweep.
    delta_per_mechanism: dict = None  # filled below

    # Backward-compat: full grid (legacy code referenced .deltas directly).
    # Kept so older scripts still work; new sweep code uses delta_per_mechanism.
    deltas: Tuple[float, ...] = (0.0, 1e-5)

    # Jump sensitivity at the 95th percentile of |x_{i+1} - x_i| across
    # MIT-BIH DS1+DS2 z-normalised signal. Derived in
    # notebooks/00_sensitivity_calibration.ipynb.
    # This is a jump statistic, NOT a global sensitivity bound. The guarantee is
    # stated as metric differential privacy (Chatzikokolakis et al., PETS 2013):
    # two recordings count as adjacent when they differ in one sample by at most
    # Delta, under which the L1 sensitivity of releasing the signal is Delta by
    # construction. See Section 3.2.1 of the paper.
    sensitivity: float = 0.29

    mechanisms: Tuple[str, ...] = ("laplace", "laplace_bounded", "gaussian_analytic")

    # Three seeds for robust baseline statistics.
    seeds: Tuple[int, ...] = (42, 1337, 2026)

    use_fast: bool = True

    @staticmethod
    def delta_for(mechanism: str) -> float:
        """Return the canonical delta to use for a given mechanism."""
        return {
            "laplace": 0.0,
            "laplace_bounded": 1e-5,
            "gaussian_analytic": 1e-5,
        }.get(mechanism, 1e-5)

    @property
    def seed(self) -> int:
        """First seed, kept as the legacy attribute for backward compatibility."""
        return self.seeds[0]


# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Data loading and preprocessing."""

    # MIT-BIH Arrhythmia
    mitbih_fs: float = 360.0
    mitbih_channel: str = "MLII"

    # ECG-ID (used later for re-identification experiments)
    ecgid_fs: float = 500.0

    # Signal preprocessing - Mousavi-style whole-record z-norm
    detrend: bool = False
    znormalize: bool = True


# -----------------------------------------------------------------------------
# Seq2Seq classifier configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Seq2SeqConfig:
    """Mousavi CNN-LSTM seq2seq classifier (TF2/Keras reimplementation)."""

    # Architecture
    beat_length: int = 280
    beats_per_group: int = 10
    cnn_n_channels: int = 10
    lstm_units: int = 128

    aami_classes: Tuple[str, ...] = ("N", "S", "V")
    go_token: str = "<GO>"

    # Training defaults (override via CLI)
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3

    # Inter-patient split per Chazal et al. 2004 / AAMI EC57 / Mousavi 2019
    ds1_records: Tuple[str, ...] = (
        "101", "106", "108", "109", "112", "114", "115", "116",
        "118", "119", "122", "124", "201", "203", "205", "207",
        "208", "209", "215", "220", "223", "230",
    )
    ds2_records: Tuple[str, ...] = (
        "100", "103", "105", "111", "113", "117", "121", "123",
        "200", "202", "210", "212", "213", "214", "219", "221",
        "222", "228", "231", "232", "233", "234",
    )

    @property
    def n_classes(self) -> int:
        return len(self.aami_classes)


# -----------------------------------------------------------------------------
# Re-ID configuration (used in step 3 of the pipeline)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ReIDConfig:
    """1D-CNN re-identification model (generic FCN-style backbone, window-based)."""

    window_sec: float = 2.0
    step_sec: float = 1.0
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3

    conv_filters: Tuple[int, ...] = (32, 64, 128)
    conv_kernel: int = 7
    pool_size: int = 2
    dense_units: int = 128
    dropout: float = 0.3


# -----------------------------------------------------------------------------
# Singletons
# -----------------------------------------------------------------------------
DP = DPConfig()
DATA = DataConfig()
SEQ2SEQ = Seq2SeqConfig()
REID = ReIDConfig()


def ensure_dirs() -> None:
    """Create result/data directories if missing."""
    for d in (DATA_DIR, RESULTS_DIR, MITBIH_DIR, ECGID_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"DP epsilon grid: {DP.epsilons}")
    print(f"DP delta grid:   {DP.deltas}")
    print(f"Sensitivity:     {DP.sensitivity}")
    print(f"Mechanisms:      {DP.mechanisms}")
    print(f"DS1 records:     {len(SEQ2SEQ.ds1_records)}")
    print(f"DS2 records:     {len(SEQ2SEQ.ds2_records)}")
    ensure_dirs()
    print("Directories OK.")
