"""
Simple IO helpers. Pickle-based, as requested.

Convention for result paths:
    results/{experiment}/{mechanism}/eps_{eps}_delta_{delta}/{filename}.pkl

Example:
    results/seq2seq/laplace/eps_1.0_delta_0.01/metrics.pkl
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union

from configs import RESULTS_DIR


def save_pickle(obj: Any, path: Union[str, Path]) -> Path:
    """Save object to pickle. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickled object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def result_path(
    experiment: str,
    mechanism: str,
    eps: float,
    delta: float,
    filename: str = "metrics.pkl",
) -> Path:
    """Standard path for an experiment result file."""
    return (
        RESULTS_DIR
        / experiment
        / mechanism
        / f"eps_{eps}_delta_{delta}"
        / filename
    )


def baseline_path(experiment: str, filename: str = "metrics.pkl") -> Path:
    """Path for a no-DP baseline result."""
    return RESULTS_DIR / experiment / "no_dp" / filename
