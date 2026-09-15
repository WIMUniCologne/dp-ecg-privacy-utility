"""
Tests for Experiment 1 (attribute labels / subject-disjoint split) and
Experiment 2 (beat features). No PhysioNet data or TensorFlow required: header
parsing is checked on the real header formats, and the subject-disjoint split is
checked on synthetic signals.

Run:
    python tests/test_attributes_and_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

import numpy as np

from src.data.attributes import (
    _assemble, age_band_edges, age_to_band, majority_class_floor,
    parse_demographics,
)

# Import the leaf feature module by path so the test stays TensorFlow-free
# (src/models/__init__.py imports the TF-based seq2seq model).
_ef_path = Path(__file__).resolve().parent.parent / "src" / "models" / "ecg_features.py"
_spec = importlib.util.spec_from_file_location("ecg_features", _ef_path)
_ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ef)
extract_beat_features = _ef.extract_beat_features
feature_names = _ef.feature_names


# --- Experiment 1: header parsing (real formats) ---
def test_parse_mitbih_header():
    # Real: "# 69 M 1085 1629 x1" -> comments[0] = "69 M 1085 1629 x1"
    assert parse_demographics(["69 M 1085 1629 x1", "Aldomet"], "mitbih") == {"age": 69, "sex": "M"}
    assert parse_demographics(["54 F 1002 x1"], "mitbih") == {"age": 54, "sex": "F"}
    assert parse_demographics(["? ? 0 0"], "mitbih") == {"age": None, "sex": None}


def test_parse_ecgid_header():
    # Real: "# Age: 25" / "# Sex: male"
    assert parse_demographics(["Age: 25", "Sex: male", "ECG date: x"], "ecgid") == {"age": 25, "sex": "M"}
    assert parse_demographics(["Age: 70", "Sex: female"], "ecgid") == {"age": 70, "sex": "F"}


def test_age_bands_and_floor():
    edges = age_band_edges([13, 20, 25, 30, 40, 55, 70, 75], 2)
    assert len(edges) == 1
    assert age_to_band(20, edges) == 0 and age_to_band(70, edges) == 1
    assert abs(majority_class_floor(np.array([0, 0, 0, 1, 1]), 2) - 0.6) < 1e-9


# --- Experiment 1: subject-disjoint split ---
def _synthetic_entry(subject, sex, fs=360.0, secs=8):
    n = int(fs * secs)
    sig = np.random.default_rng(abs(hash(subject)) % 2**32).normal(0, 1, n).astype(np.float32)
    return (subject, sig, fs, {"sex": sex, "age": 40})


def test_subject_disjoint_split():
    entries = [_synthetic_entry(f"s{i}", "M" if i % 2 else "F") for i in range(10)]
    train, test = _assemble(entries, "sex", [], 360.0, train_frac=0.6,
                            split_seed=7, window_sec=2.0, step_sec=1.0)
    tr_subj = set(np.unique(train.subject_per_window))
    te_subj = set(np.unique(test.subject_per_window))
    assert tr_subj and te_subj
    assert tr_subj.isdisjoint(te_subj), "subjects must not appear in both splits"
    assert train.class_names == ["F", "M"]
    assert train.X.shape[1] == int(round(360.0 * 2.0))
    # labels within range
    assert set(np.unique(train.y)).issubset({0, 1})


# --- Experiment 2: beat features ---
def test_feature_shape_and_finiteness():
    rng = np.random.default_rng(0)
    beats = rng.normal(0, 1, size=(50, 280)).astype(np.float32)
    F = extract_beat_features(beats)
    assert F.shape == (50, len(feature_names()))
    assert np.isfinite(F).all()
    # 1D input round-trips
    assert extract_beat_features(beats[0]).shape == (1, len(feature_names()))


def test_features_track_morphology():
    """Different morphologies must yield different feature vectors."""
    t = np.linspace(0, 1, 280)
    a = np.exp(-0.5 * ((t - 0.5) / 0.02) ** 2)            # narrow spike
    b = np.exp(-0.5 * ((t - 0.5) / 0.10) ** 2)            # wide bump
    Fa = extract_beat_features(a[None]); Fb = extract_beat_features(b[None])
    assert np.linalg.norm(Fa - Fb) > 1e-3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
