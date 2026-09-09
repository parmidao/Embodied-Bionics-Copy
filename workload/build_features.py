"""
build_features.py
=================
Step 2 of the workload pipeline: load synthetic (later: real) recordings,
cut CAUSAL 10 s / 5 s-hop windows, compute HRV / EDA / EMG features, and
assemble a feature table.

TWO HARD RULES ENFORCED HERE (from the project brief):

  Rule A — CAUSAL WINDOWS. Each 10 s window uses only samples that fall inside
  it. Nothing from the future enters a window.

  Rule B — BASELINE-ONLY, PER-PARTICIPANT NORMALISATION. The scaling numbers
  (median + IQR) are estimated ONCE from each participant's 120 s baseline
  segment, frozen, and then applied forward to all that person's windows.
  Never whole-session statistics. Never pooled across participants. This is
  the non-causal-normalisation guard from constraint #3.

Every window row carries BOTH label schemes side by side:
  nominal    — absolute n-back level (0,1,2,3)
  calibrated — level relative to this person's n*  (below / at / above)
plus participant id and segment type. Step 4 runs the model twice, once per
label column, on identical features/folds.

⚠️ Synthetic data. No number from it is a result.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "synthetic_data"
WINDOW_SEC = 10.0
HOP_SEC    = 5.0
BASELINE_SEGMENT = "baseline"   # segment type whose stats we freeze


# ----------------------------------------------------------------------------
# Loader. For synthetic .npz. When real .xdf arrives, ONLY this function
# changes (swap np.load for pyxdf.load_xdf and remap fields); everything
# below stays identical.
# ----------------------------------------------------------------------------
def load_recording(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    stream_names = list(z["__stream_names"])
    streams = {}
    for s in stream_names:
        streams[s] = {
            "data": z[f"{s}__data"],
            "ts":   z[f"{s}__ts"],
            "rate": float(z[f"{s}__rate"][0]),
            "n_ch": int(z[f"{s}__n_ch"][0]),
        }
    segment_log = z["__segment_log"]   # rows: (t0, t1, seg_type, nback, nstar)
    nstar = int(z["__nstar"][0])
    return streams, segment_log, nstar


# ----------------------------------------------------------------------------
# Helpers to slice a stream to a time window (causal: [t0, t1) only)
# ----------------------------------------------------------------------------
def slice_stream(stream, t0, t1):
    ts = stream["ts"]
    mask = (ts >= t0) & (ts < t1)
    return stream["data"][mask], ts[mask]


# ----------------------------------------------------------------------------
# Feature functions. Each takes a window's raw samples, returns a dict.
# These mirror the physiology in the real pipeline; on synthetic data they
# just have to be computable and stable.
# ----------------------------------------------------------------------------
def hrv_features(rr_ms):
    """Time-domain HRV from RR intervals (ms) within the window."""
    rr = np.asarray(rr_ms, dtype=float).ravel()
    rr = rr[(rr > 300) & (rr < 2000)]     # plausibility filter (real pipeline
                                          # needs stronger artifact correction)
    if len(rr) < 3:
        return {"hr_mean": np.nan, "sdnn": np.nan, "rmssd": np.nan,
                "pnn50": np.nan, "n_beats": len(rr)}
    d = np.diff(rr)
    return {
        "hr_mean": 60000.0 / np.mean(rr),
        "sdnn":    np.std(rr, ddof=1) if len(rr) > 1 else 0.0,
        "rmssd":   np.sqrt(np.mean(d**2)) if len(d) else 0.0,
        "pnn50":   100.0 * np.mean(np.abs(d) > 50) if len(d) else 0.0,
        "n_beats": len(rr),
    }


def eda_features(sig):
    sig = np.asarray(sig, dtype=float).ravel()
    if len(sig) < 5:
        return {"scl_mean": np.nan, "scl_std": np.nan, "scl_slope": np.nan}
    x = np.arange(len(sig))
    slope = np.polyfit(x, sig, 1)[0] if len(sig) > 1 else 0.0
    return {"scl_mean": float(np.mean(sig)),
            "scl_std":  float(np.std(sig)),
            "scl_slope": float(slope)}


def emg_features(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if len(arr) < 5:
        return {"emg_rms": np.nan, "emg_mav": np.nan}
    ch0 = arr[:, 0]
    return {"emg_rms": float(np.sqrt(np.mean(ch0**2))),
            "emg_mav": float(np.mean(np.abs(ch0)))}


FEATURE_COLUMNS = ["hr_mean", "sdnn", "rmssd", "pnn50",
                   "scl_mean", "scl_std", "scl_slope",
                   "emg_rms", "emg_mav"]


# ----------------------------------------------------------------------------
# Per-window feature extraction over one participant
# ----------------------------------------------------------------------------
def segment_at(segment_log, t):
    """Return (seg_type, nback, nstar) for the segment containing time t."""
    for (t0, t1, seg_type, nback, nstar) in segment_log:
        if t0 <= t < t1:
            return seg_type, nback, nstar
    return None, None, None


def calibrated_label(nback, nstar):
    """below / at / above that person's capacity. 0-back and baseline -> 'low'."""
    if nback is None:
        return None
    if nback == 0:
        return "low"          # 0-back is the low anchor in the main blocks
    if nback < nstar:
        return "below"
    elif nback == nstar:
        return "at"
    else:
        return "above"


def extract_participant(npz_path):
    streams, segment_log, nstar = load_recording(npz_path)
    pid = npz_path.stem.split("_")[0]   # 'P01'

    rr_stream  = streams["Polar_RR"]
    eda_stream = streams["BioRadio_EDA"]
    emg_stream = streams["RawEMG"]

    # session end = min across streams (only where all have data)
    t_end = min(s["ts"][-1] for s in streams.values())

    rows = []
    t = 0.0
    while t + WINDOW_SEC <= t_end:
        t0, t1 = t, t + WINDOW_SEC
        seg_type, nback, _ = segment_at(segment_log, t0)

        rr, _  = slice_stream(rr_stream, t0, t1)
        eda, _ = slice_stream(eda_stream, t0, t1)
        emg, _ = slice_stream(emg_stream, t0, t1)

        feats = {}
        feats.update(hrv_features(rr))
        feats.update(eda_features(eda))
        feats.update(emg_features(emg))

        feats["participant"] = pid
        feats["segment"]     = seg_type
        feats["t_start"]     = t0
        feats["nominal"]     = nback                      # absolute level
        feats["calibrated"]  = calibrated_label(nback, nstar)
        feats["nstar"]       = nstar
        rows.append(feats)
        t += HOP_SEC

    df = pd.DataFrame(rows)

    # ---- Rule B: freeze normalisation from BASELINE ONLY, then apply forward.
    base = df[df["segment"] == BASELINE_SEGMENT]
    med = base[FEATURE_COLUMNS].median()
    iqr = base[FEATURE_COLUMNS].quantile(0.75) - base[FEATURE_COLUMNS].quantile(0.25)
    iqr = iqr.replace(0, 1.0)   # avoid divide-by-zero on flat baseline features
    df[FEATURE_COLUMNS] = (df[FEATURE_COLUMNS] - med) / iqr

    return df


def main():
    files = sorted(DATA_DIR.glob("P*_synthetic.npz"))
    print(f"Found {len(files)} recordings in {DATA_DIR}\n")
    all_rows = []
    for f in files:
        dfp = extract_participant(f)
        n_main = (dfp["segment"] == "main").sum()
        n_stair = (dfp["segment"] == "staircase").sum()
        print(f"  {f.stem.split('_')[0]}  windows: {len(dfp):4d}  "
              f"(staircase {n_stair}, main {n_main})")
        all_rows.append(dfp)

    df = pd.concat(all_rows, ignore_index=True)
    out = DATA_DIR.parent / "features.parquet"
    df.to_parquet(out)

    print(f"\nTotal windows: {len(df)}")
    print(f"Feature table saved to: {out}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nWindows per segment:\n{df['segment'].value_counts()}")
    print(f"\nNominal label counts (staircase only):")
    print(df[df.segment == 'staircase']['nominal'].value_counts().sort_index())
    print(f"\nCalibrated label counts (staircase only):")
    print(df[df.segment == 'staircase']['calibrated'].value_counts())
    print(f"\nAny NaNs in feature columns?")
    print(df[FEATURE_COLUMNS].isna().sum())
    print("\n⚠️  Synthetic. No number here is a result.")


if __name__ == "__main__":
    main()
