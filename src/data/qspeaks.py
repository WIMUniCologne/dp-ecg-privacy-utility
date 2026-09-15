"""
Python port of Mousavi's MATLAB qsPeaks.m + onoffset.m.

Detects fiducial points around each R-peak in an ECG signal:
  P, QRSon, Q, R, S, QRSoff, T

Only beats where ALL 7 fiducial points were successfully detected are kept.
This matches the MATLAB pipeline exactly (line 119 in qsPeaks.m:
    if Ind(i) ~= 0   % only keep if prod(fid_pks(i,:)) != 0
        ECGpeaks = [ECGpeaks; fid_pks(i,:)]

Output is a (n_kept, 7) array of integer sample positions, columns:
    [P, QRSon, Q, R, S, QRSoff, T]

Beats:
  - First beat (i=0): no P, no T-before → only QRSon, Q, R, S, QRSoff set
                       → P=0 → product is 0 → DROPPED in the final filter.
  - Last beat: T can't be detected (needs next-ON) → DROPPED.
  - Middle beats: kept only if QRSon/QRSoff AND P AND T were detectable.

This is the fiducial-point pipeline that gives us:
  - tpeaks (column 7 = T-wave positions) for T-to-T beat segmentation
  - the index filter for annotations (only kept beats)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def onoffset(interval: np.ndarray, mode: str) -> int:
    """
    Find onset or offset of QRS complex within a small interval.

    Port of MATLAB onoffset.m. Returns the index *within* `interval`
    (1-based in MATLAB; we return 0-based Python index).

    Args:
        interval: 1D signal segment (typically ~40 ms = ~14 samples at 360 Hz)
        mode: 'on' (find minimum-absolute-slope point) or 'off' (first point
              where |slope| exceeds 20% of max |slope|)

    Returns:
        0-based index within `interval` (caller adjusts to global position).
        If interval is too short, returns 0.
    """
    if len(interval) < 3:
        return 0

    # slope[i] = interval[i+1] - interval[i-1] for i in 1..len-2
    # In MATLAB: slope = interval(i+1) - interval(i-1) for i=2..len-1
    # Resulting slope has length len-2
    slope = interval[2:] - interval[:-2]  # length = len(interval) - 2
    abs_slope = np.abs(slope)

    if mode == 'on':
        # min |slope| → flatten point = onset
        ind_in_slope = int(np.argmin(abs_slope))
    elif mode == 'off':
        # First point where |slope| >= 20% of max
        threshold = 0.2 * abs_slope.max()
        candidates = np.where(abs_slope >= threshold)[0]
        if len(candidates) == 0:
            ind_in_slope = 0
        else:
            ind_in_slope = int(candidates[0])
    else:
        raise ValueError(f"mode must be 'on' or 'off', got {mode!r}")

    # MATLAB returns ind into `slope` 1-based; our ind_in_slope is 0-based.
    # MATLAB then uses this `ind` directly to index back into the original
    # interval (which is fine because slope is offset by 1 from interval start).
    # In MATLAB: thisON = thisQ - (windowOF+1) + ind  (ind is 1-based slope index)
    #   This means the caller adds (start_in_signal - windowOF - 1 + ind_1based).
    #   With our 0-based: start_in_signal - windowOF + (ind_in_slope + 0)
    # We return 0-based slope index. Caller must add appropriate offsets.
    return ind_in_slope


def qs_peaks(
    ecg: np.ndarray,
    r_positions: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Detect fiducial points (P, QRSon, Q, R, S, QRSoff, T) around each R-peak.

    Direct port of MATLAB qsPeaks.m. Returns ONLY beats where all 7 fiducial
    points were successfully detected (i.e., the product over the row is != 0).

    Args:
        ecg: 1D ECG signal
        r_positions: 1D array of R-peak sample indices (0-based, Python convention)
        fs: sampling frequency (Hz)

    Returns:
        (n_kept, 7) integer array, columns: [P, QRSon, Q, R, S, QRSoff, T]
        All positions are 0-based sample indices into `ecg`.

    Notes:
        - First and last beats are systematically dropped (cannot complete the
          fiducial sextet).
        - The MATLAB code is 1-based; we adjust offsets so that result indices
          can be used directly to slice `ecg[start:end]`.
    """
    if len(r_positions) < 3:
        return np.zeros((0, 7), dtype=np.int64)

    ecg = np.asarray(ecg).flatten()
    R = np.asarray(r_positions).astype(np.int64).flatten()
    n_peaks = len(R)
    n_ecg = len(ecg)

    ave_hb = n_ecg / n_peaks
    fid_pks = np.zeros((n_peaks, 7), dtype=np.int64)

    # Search windows (MATLAB: round())
    windowS = int(round(fs * 0.1))    # ~36 samples
    windowQ = int(round(fs * 0.05))   # ~18 samples
    windowP = int(round(ave_hb / 3))
    windowT = int(round(ave_hb * 2 / 3))
    windowOF = int(round(fs * 0.04))  # ~14 samples

    # ---- Step 1: QRS landmarks (Q, R, S, QRSon, QRSoff) ----
    for i in range(n_peaks):
        thisR = R[i]

        if i == 0:
            # First beat: only set R (col 3) and QRSoff (col 5 = R+windowS, 0-indexed)
            # MATLAB cols are 1-indexed: (4)=R, (6)=QRSoff
            # In Python: cols (3)=R, (5)=QRSoff
            fid_pks[i, 3] = thisR
            fid_pks[i, 5] = thisR + windowS
            continue
        if i == n_peaks - 1:
            # Last beat: set R and QRSon
            fid_pks[i, 3] = thisR
            fid_pks[i, 1] = thisR - windowQ
            continue

        # Middle beat: check bounds
        if not ((thisR + windowT) < n_ecg and (thisR - windowP) >= 0):
            continue  # leave row zeros (will be filtered later)

        # R
        fid_pks[i, 3] = thisR

        # S = local min in [R, R+windowS]
        end_S = min(thisR + windowS + 1, n_ecg)
        seg_S = ecg[thisR:end_S]
        if len(seg_S) == 0:
            continue
        Sp = int(np.argmin(seg_S))
        thisS = Sp + thisR
        fid_pks[i, 4] = thisS

        # Q = local min in [R-windowQ, R]
        start_Q = max(thisR - windowQ, 0)
        seg_Q = ecg[start_Q : thisR + 1]
        if len(seg_Q) == 0:
            continue
        Qp = int(np.argmin(seg_Q))
        thisQ = start_Q + Qp
        fid_pks[i, 2] = thisQ

        # QRSon: onset using onoffset() on interval [Q-windowOF, Q]
        start_qon = max(thisQ - windowOF, 0)
        interval_q = ecg[start_qon : thisQ + 1]
        if len(interval_q) < 3:
            continue
        ind_on = onoffset(interval_q, 'on')
        # MATLAB: thisON = thisQ - (windowOF+1) + ind  (1-based ind)
        # Our `ind_on` is 0-based slope index; slope starts at interval[1]
        # → actual sample in original signal:
        #   start_qon + (ind_on + 1)  (because slope[0] corresponds to interval[1])
        # We want the position equivalent to MATLAB's:
        #   thisQ - (windowOF + 1) + ind_1based
        #   = (thisQ - windowOF) - 1 + (ind_0based + 1)
        #   = (thisQ - windowOF) + ind_0based
        thisON = start_qon + ind_on
        # Make sure thisON > 0 so product survives
        if thisON <= 0:
            thisON = 1
        fid_pks[i, 1] = thisON

        # QRSoff: offset using onoffset() on interval [S, S+windowOF]
        end_qoff = min(thisS + windowOF + 1, n_ecg)
        interval_s = ecg[thisS : end_qoff]
        if len(interval_s) < 3:
            continue
        ind_off = onoffset(interval_s, 'off')
        # MATLAB: thisOFF = thisS + ind - 1  (1-based)
        # Our 0-based: thisOFF = thisS + ind_0based
        # But slope[0] corresponds to interval[1], so 0-based equivalent:
        thisOFF = thisS + ind_off + 1
        if thisOFF >= n_ecg:
            thisOFF = n_ecg - 1
        fid_pks[i, 5] = thisOFF

    # ---- Step 2: P and T waves (only for middle beats with valid neighbors) ----
    for i in range(1, n_peaks - 1):
        lastOFF = fid_pks[i - 1, 5]
        thisON = fid_pks[i, 1]
        thisOFF = fid_pks[i, 5]
        nextON = fid_pks[i + 1, 1]

        if not (thisON > lastOFF and thisOFF < nextON):
            continue
        if thisON <= 0 or thisOFF <= 0 or lastOFF <= 0 or nextON <= 0:
            continue

        # Tzone = [thisOFF, nextON - (nextON-thisOFF)/3]
        t_zone_end = nextON - int(round((nextON - thisOFF) / 3))
        if t_zone_end <= thisOFF:
            continue
        Tzone = ecg[thisOFF : t_zone_end + 1]
        if len(Tzone) == 0:
            continue
        thisT_in_zone = int(np.argmax(Tzone))
        thisT = thisOFF + thisT_in_zone
        if thisT <= 0:
            thisT = 1

        # Pzone = [lastOFF + 2*(thisON-lastOFF)/3, thisON]
        p_zone_start = lastOFF + int(round(2 * (thisON - lastOFF) / 3))
        if thisON <= p_zone_start:
            continue
        Pzone = ecg[p_zone_start : thisON + 1]
        if len(Pzone) == 0:
            continue
        thisP_in_zone = int(np.argmax(Pzone))
        thisP = p_zone_start + thisP_in_zone
        if thisP <= 0:
            thisP = 1

        fid_pks[i, 0] = thisP   # P
        fid_pks[i, 6] = thisT   # T

    # ---- Step 3: Filter — keep only beats where ALL 7 fiducials are nonzero ----
    # MATLAB: Ind(i) = prod(fid_pks(i, :))
    products = np.prod(fid_pks, axis=1)
    kept_mask = products != 0
    return fid_pks[kept_mask]
