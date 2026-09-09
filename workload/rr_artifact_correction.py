"""
rr_artifact_correction.py
=========================
Robust artifact correction for Polar RR intervals.

WHY THIS EXISTS
  With raw ECG dropped from the recording, RR is the ONLY cardiac signal, so
  it carries all cardiac workload features. That makes beat-detection artifacts
  more dangerous: with ECG you could verify a suspicious interval against the
  waveform; with RR alone you must detect and correct blind. This module does
  that with the standard method from the HRV literature rather than the crude
  300-2000 ms range filter the pipeline used before.

THE FAILURE MODES IT HANDLES (all seen in the July pilot)
  - MISSED beat  -> one interval ~2x (or 3x) a normal one
                    (pilot: 1290 ms ~ 2x620, 2170 ms ~ 3x620)
  - EXTRA beat   -> two short intervals that sum to one normal one
                    (pilot: 400 ms then 830 ms, summing to ~1230 ~ 2x620)
  - ECTOPIC/noise-> a single interval far from its local neighbours

WHY RMSSD MAKES THIS URGENT
  RMSSD squares successive differences. A single 2170 ms interval next to a
  620 ms neighbour contributes a squared difference of ~2.4 million and can
  inflate RMSSD several-fold within a short window. One bad beat corrupts the
  whole window's headline cardiac feature. So correction is not optional.

METHOD (Malik-style local-median rule, the standard)
  1. Physiological gate: drop intervals outside [300, 2000] ms (impossible).
  2. Local-median test: for each interval, compare it to the median of a small
     window of its neighbours. If it deviates by more than THRESHOLD (default
     20%), flag it as an artifact.
  3. Correct flagged intervals by interpolation from neighbours (default), or
     drop them. Interpolation preserves the sample count and the time base.
  4. Report how many were corrected, so a bad recording is visible rather than
     silently "cleaned" into plausible-looking garbage.

The reporting matters: if 30% of intervals in a window are corrected, that
window's HRV is not trustworthy no matter how clean the output looks. The
function returns the correction fraction so downstream code can flag or drop
such windows.
"""

import numpy as np


def correct_rr(rr_ms, threshold=0.20, local_window=5, method="interpolate",
               min_valid=300.0, max_valid=2000.0):
    """
    Correct artifacts in a sequence of RR intervals (milliseconds).

    Parameters
    ----------
    rr_ms : array-like
        RR intervals in ms, in order.
    threshold : float
        Fractional deviation from the local median above which an interval is
        flagged as an artifact. 0.20 = 20%, the common default.
    local_window : int
        Number of neighbouring intervals used to compute the local median
        (odd number; the interval itself is excluded from its own median).
    method : {"interpolate", "drop"}
        How to handle flagged intervals. "interpolate" replaces them with the
        local median (preserves count and time base); "drop" removes them.
    min_valid, max_valid : float
        Hard physiological bounds in ms. Outside these, an interval is
        impossible (300 ms = 200 bpm, 2000 ms = 30 bpm) and always removed.

    Returns
    -------
    dict with:
        rr_corrected : np.ndarray  the cleaned intervals
        n_input      : int         how many came in
        n_flagged    : int         how many were flagged as artifacts
        frac_flagged : float       n_flagged / n_input  (window-quality signal)
        mask_artifact: np.ndarray  boolean, True where an artifact was flagged
    """
    rr = np.asarray(rr_ms, dtype=float).ravel()
    n_input = len(rr)
    if n_input == 0:
        return {"rr_corrected": rr, "n_input": 0, "n_flagged": 0,
                "frac_flagged": 0.0, "mask_artifact": np.array([], dtype=bool)}

    # 1. hard physiological gate
    valid = (rr >= min_valid) & (rr <= max_valid) & np.isfinite(rr)

    # 2. local-median test on the physiologically-plausible intervals
    artifact = ~valid.copy()          # already-invalid count as artifacts
    half = local_window // 2
    for i in range(n_input):
        if not valid[i]:
            continue
        lo = max(0, i - half)
        hi = min(n_input, i + half + 1)
        neighbours = np.concatenate([rr[lo:i], rr[i+1:hi]])
        neighbours = neighbours[(neighbours >= min_valid) &
                                (neighbours <= max_valid) &
                                np.isfinite(neighbours)]
        if len(neighbours) == 0:
            continue
        local_med = np.median(neighbours)
        if local_med > 0 and abs(rr[i] - local_med) / local_med > threshold:
            artifact[i] = True

    n_flagged = int(artifact.sum())
    frac_flagged = n_flagged / n_input

    # 3. correct
    if method == "drop":
        rr_corrected = rr[~artifact]
    else:  # interpolate: replace each artifact with the local median of good neighbours
        rr_corrected = rr.copy()
        good = ~artifact
        for i in np.where(artifact)[0]:
            lo = max(0, i - half)
            hi = min(n_input, i + half + 1)
            idx = np.arange(lo, hi)
            idx = idx[(idx != i) & good[idx]]
            if len(idx) > 0:
                rr_corrected[i] = np.median(rr[idx])
            else:
                rr_corrected[i] = np.nan
        rr_corrected = rr_corrected[np.isfinite(rr_corrected)]

    return {"rr_corrected": rr_corrected, "n_input": n_input,
            "n_flagged": n_flagged, "frac_flagged": frac_flagged,
            "mask_artifact": artifact}


def hrv_from_rr_corrected(rr_ms, max_artifact_frac=0.30, **kwargs):
    """
    Correct RR artifacts, then compute time-domain HRV. Returns NaNs (and a
    quality flag) if too much of the window had to be corrected to be trusted.

    max_artifact_frac : if more than this fraction of intervals were flagged,
    the window is deemed untrustworthy and HRV is returned as NaN with
    hrv_ok=False. A window that needed a third of its beats corrected is not a
    clean HRV measurement, however plausible the numbers look afterwards.
    """
    res = correct_rr(rr_ms, **kwargs)
    rr = res["rr_corrected"]
    out = {"frac_artifact": res["frac_flagged"], "n_beats": len(rr),
           "hrv_ok": True}

    if res["frac_flagged"] > max_artifact_frac or len(rr) < 5:
        out.update({"hr_mean": np.nan, "sdnn": np.nan, "rmssd": np.nan,
                    "pnn50": np.nan, "hrv_ok": False})
        return out

    d = np.diff(rr)
    out.update({
        "hr_mean": 60000.0 / np.mean(rr),
        "sdnn":    np.std(rr, ddof=1),
        "rmssd":   np.sqrt(np.mean(d**2)),
        "pnn50":   100.0 * np.mean(np.abs(d) > 50),
    })
    return out


# ----------------------------------------------------------------------------
# Self-test with the ACTUAL pilot artifacts
# ----------------------------------------------------------------------------
def _demo():
    print("RR artifact correction — self-test on the July pilot's real artifacts")
    print("=" * 70)

    # a clean run around 620 ms with the pilot's specific artifacts inserted
    rng = np.random.default_rng(0)
    clean = rng.normal(620, 25, 40)                    # ~97 bpm, healthy variation
    dirty = clean.copy()
    dirty[5]  = 2170          # two missed beats (pilot)
    dirty[12] = 1290          # one missed beat (pilot)
    dirty[20] = 400           # extra beat, first half (pilot)
    dirty[21] = 830           # extra beat, second half (pilot)

    print("\nWithout correction (crude, trust everything):")
    d = np.diff(dirty)
    print(f"  RMSSD = {np.sqrt(np.mean(d**2)):.1f} ms   <-- inflated by the artifacts")
    print(f"  SDNN  = {np.std(dirty, ddof=1):.1f} ms")

    print("\nWith correction:")
    res = correct_rr(dirty)
    print(f"  flagged {res['n_flagged']} / {res['n_input']} intervals "
          f"({100*res['frac_flagged']:.0f}%) as artifacts")
    print(f"  flagged positions: {list(np.where(res['mask_artifact'])[0])}")
    rc = res["rr_corrected"]
    dc = np.diff(rc)
    print(f"  RMSSD = {np.sqrt(np.mean(dc**2)):.1f} ms   <-- now reflects real variability")
    print(f"  SDNN  = {np.std(rc, ddof=1):.1f} ms")

    print("\nFor reference, the truly clean signal (no artifacts):")
    dt = np.diff(clean)
    print(f"  RMSSD = {np.sqrt(np.mean(dt**2)):.1f} ms")
    print(f"  SDNN  = {np.std(clean, ddof=1):.1f} ms")

    print("\n" + "=" * 70)
    print("The corrected RMSSD should land close to the clean reference,")
    print("while the uncorrected one is inflated several-fold. That gap is")
    print("why RR-only requires real artifact correction, not a range filter.")

    # window-quality gate demo
    print("\nWindow-quality gate:")
    hrv = hrv_from_rr_corrected(dirty)
    print(f"  frac_artifact={hrv['frac_artifact']:.2f}  hrv_ok={hrv['hrv_ok']}  "
          f"RMSSD={hrv['rmssd']:.1f}")
    mostly_bad = np.array([620, 2000, 400, 1800, 500, 620])
    hrv2 = hrv_from_rr_corrected(mostly_bad)
    print(f"  mostly-bad window: frac_artifact={hrv2['frac_artifact']:.2f}  "
          f"hrv_ok={hrv2['hrv_ok']}  (correctly refuses to report HRV)")


if __name__ == "__main__":
    _demo()
