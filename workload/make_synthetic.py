"""
make_synthetic.py
==================
Generates realistic FAKE .xdf-shaped recordings so the workload pipeline can be
built and tested before real data exists (real collection: 24-28 Aug 2026).

⚠️  READ THIS:
Synthetic data proves the PLUMBING works. It proves NOTHING about accuracy.
Any number that comes out of the pipeline on this data is meaningless and must
never appear in a report, a figure, or a conversation with a PI. The only
questions this data can answer are:
  - Does the pipeline run end to end without crashing?
  - Does it return CHANCE when SIGNAL_STRENGTH = 0.0 (i.e. it doesn't
    hallucinate patterns)?
  - Can it recover CALIBRATED labels (and blur NOMINAL labels) when a
    signal tied to each person's n* is present?

The workload signal is injected RELATIVE TO EACH PARTICIPANT'S n*, not the
absolute n-back level. This is deliberate: it is the only way the synthetic
data can exercise the paper's actual hypothesis (calibrated labels generalise
across people, nominal labels do not). If the signal were injected by absolute
level, both labelling schemes would succeed and the harness would test nothing.
"""

import numpy as np
from pathlib import Path

# Try to import the XDF writer. pyxdf READS xdf; writing needs a small helper.
# We write a minimal valid XDF ourselves so no extra dependency is required.

# ----------------------------------------------------------------------------
# KNOBS
# ----------------------------------------------------------------------------
SIGNAL_STRENGTH = 0.5     # 0.0 = pure noise (pipeline MUST return chance)
                          # ~0.5 = weak but detectable, tied to each person's n*
RANDOM_SEED     = 7
OUT_DIR         = Path(__file__).parent / "synthetic_data"

# Participant n* distribution — deliberately MIXED so H1 is testable and the
# "missing class" case (n*=1 has no 'below', n*=3 has no 'above') appears.
NSTAR_ASSIGNMENT = (
    [1] * 4 +   # 4 participants cap out at 1-back
    [2] * 8 +   # 8 at 2-back
    [3] * 6     # 6 at 3-back
)   # 18 participants total

# Real stream names + realistic rates (from the actual pilot files)
STREAMS = {
    "Polar_ECG":       {"rate": 130.0,   "n_ch": 1},
    "Polar_HR":        {"rate": 1.0,     "n_ch": 1},   # irregular ~1 Hz
    "Polar_RR":        {"rate": 0.0,     "n_ch": 1},   # irregular, per-beat
    "BioRadio_EDA":    {"rate": 250.0,   "n_ch": 1},
    "RawEMG":          {"rate": 1248.5,  "n_ch": 4},   # TRUE rate, not declared 1111
    "OVR_Gaze_Stream": {"rate": 90.0,    "n_ch": 7},   # present but unused by features
}

# Session structure (seconds)
BASELINE_SEC   = 120.0          # frozen normalisation comes from here ONLY
STAIRCASE_LEVELS = [1, 2, 3]    # 1/2/3-back
STAIRCASE_SEC  = 90.0           # per level
N_MAIN_BLOCKS  = 8
MAIN_BLOCK_SEC = 180.0          # 60 BBT + 120 CRT (collapsed to one label here)

# ----------------------------------------------------------------------------
# Signal model
# ----------------------------------------------------------------------------
# We model workload as a scalar "load" per segment, then let it push a few
# physiological summaries: HR up, HRV down, EDA level up, EMG up. The load is
# defined RELATIVE to n* so calibrated labels carry it and nominal labels don't.

def relative_load(nback_level, nstar):
    """
    Map (absolute nback, this person's nstar) -> a load in {below, at, above}.
    Returns a scalar: below=0.0, at=0.5, above=1.0. This is what the signal
    tracks. Two people at the same ABSOLUTE level can be at different loads.
    """
    if nback_level < nstar:
        return 0.0      # below capacity
    elif nback_level == nstar:
        return 0.5      # at capacity
    else:
        return 1.0      # above capacity


def make_ecg(duration, fs, load, rng):
    """Fake ECG: a periodic-ish spike train whose rate rises with load."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    base_hr = 70 + 25 * load * SIGNAL_STRENGTH     # bpm rises with load
    # add per-beat jitter that SHRINKS with load (HRV down under load)
    hrv = (0.08 - 0.05 * load * SIGNAL_STRENGTH)
    beat_period = 60.0 / base_hr
    sig = np.zeros(n)
    beat_t = 0.0
    while beat_t < duration:
        idx = int(beat_t * fs)
        if idx < n:
            sig[idx] += 1.0   # R-spike
        beat_t += beat_period * (1 + rng.normal(0, hrv))
    # add small baseline noise
    sig += rng.normal(0, 0.02, n)
    return sig.reshape(-1, 1)


def make_rr(duration, load, rng):
    """Fake RR intervals (ms), irregular stream: one value per beat."""
    base_hr = 70 + 25 * load * SIGNAL_STRENGTH
    hrv_ms = (60 - 35 * load * SIGNAL_STRENGTH)     # SDNN-like, shrinks with load
    mean_rr = 60000.0 / base_hr
    rrs, ts, clock = [], [], 0.0
    while clock < duration:
        rr = rng.normal(mean_rr, hrv_ms)
        rr = float(np.clip(rr, 350, 1600))
        rrs.append(rr)
        clock += rr / 1000.0
        ts.append(clock)
    return np.array(rrs).reshape(-1, 1), np.array(ts)


def make_eda(duration, fs, load, rng):
    """Fake EDA: slow tonic level rises with load + slow drift + small noise."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    tonic = 5 + 3 * load * SIGNAL_STRENGTH
    drift = 0.5 * np.sin(2 * np.pi * 0.01 * t)
    sig = tonic + drift + rng.normal(0, 0.05, n)
    return sig.reshape(-1, 1)


def make_emg(duration, fs, load, rng, n_ch=4):
    """Fake EMG: zero-mean noise whose amplitude rises with load."""
    n = int(duration * fs)
    amp = 0.01 + 0.02 * load * SIGNAL_STRENGTH
    sig = rng.normal(0, amp, (n, n_ch))
    return sig


def make_gaze(duration, fs, rng, n_ch=7):
    """Fake gaze: present but carries no workload signal (features ignore it)."""
    n = int(duration * fs)
    return rng.normal(0, 1, (n, n_ch))


# ----------------------------------------------------------------------------
# Minimal XDF writer
# ----------------------------------------------------------------------------
# XDF is a binary chunked format. Rather than hand-roll the binary spec, we
# save each participant as a compressed .npz that mirrors the xdf structure
# (stream name -> data array + timestamps + metadata). The loader in step 2
# will read these. When real .xdf arrives, only the loader's read function
# changes; the rest of the pipeline is identical.
#
# NOTE: this keeps the synthetic path dependency-free and robust. It does mean
# check_recording.py (which expects true .xdf) is exercised separately against
# a couple of true-xdf files we can generate later if needed.

def build_participant(pid, nstar, rng):
    """Return a dict: stream_name -> {'data','ts','rate','n_ch','labels'}."""
    streams = {s: {"data": [], "ts": [], "rate": STREAMS[s]["rate"],
                   "n_ch": STREAMS[s]["n_ch"]} for s in STREAMS}
    # A parallel per-segment label log (what the manipulation check would use).
    segment_log = []   # (t_start, t_end, segment_type, nback_level, nstar)

    clock = 0.0

    def add_segment(duration, nback_level, seg_type):
        nonlocal clock
        load = relative_load(nback_level, nstar) if nback_level is not None else 0.0
        for sname, meta in STREAMS.items():
            fs = meta["rate"]
            if sname == "Polar_ECG":
                d = make_ecg(duration, fs, load, rng)
                ts = clock + np.arange(len(d)) / fs
            elif sname == "Polar_RR":
                d, ts_rel = make_rr(duration, load, rng)
                ts = clock + ts_rel
            elif sname == "Polar_HR":
                nbeats = max(1, int(duration))
                hr = 70 + 25 * load * SIGNAL_STRENGTH + rng.normal(0, 2, nbeats)
                d = hr.reshape(-1, 1); ts = clock + np.arange(nbeats)
            elif sname == "BioRadio_EDA":
                d = make_eda(duration, fs, load, rng)
                ts = clock + np.arange(len(d)) / fs
            elif sname == "RawEMG":
                d = make_emg(duration, fs, load, rng, meta["n_ch"])
                ts = clock + np.arange(len(d)) / fs
            elif sname == "OVR_Gaze_Stream":
                d = make_gaze(duration, fs, rng, meta["n_ch"])
                ts = clock + np.arange(len(d)) / fs
            streams[sname]["data"].append(d)
            streams[sname]["ts"].append(ts)
        segment_log.append((clock, clock + duration, seg_type, nback_level, nstar))
        clock += duration

    # 1. Baseline (no n-back; load = 0). Normalisation will come from here.
    add_segment(BASELINE_SEC, None, "baseline")
    # 2. Staircase: 1,2,3-back, 90 s each (all always presented).
    for lvl in STAIRCASE_LEVELS:
        add_segment(STAIRCASE_SEC, lvl, "staircase")
    # 3. Main blocks: alternate 0-back (low) and n*-back (high).
    for b in range(N_MAIN_BLOCKS):
        lvl = 0 if b % 2 == 0 else nstar
        add_segment(MAIN_BLOCK_SEC, lvl, "main")

    # concatenate per stream
    for sname in streams:
        streams[sname]["data"] = np.concatenate(streams[sname]["data"], axis=0)
        streams[sname]["ts"] = np.concatenate(streams[sname]["ts"], axis=0)

    return streams, segment_log


def save_participant(pid, nstar, streams, segment_log):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for sname, s in streams.items():
        out[f"{sname}__data"] = s["data"]
        out[f"{sname}__ts"]   = s["ts"]
        out[f"{sname}__rate"] = np.array([s["rate"]])
        out[f"{sname}__n_ch"] = np.array([s["n_ch"]])
    seg = np.array(segment_log, dtype=object)
    out["__segment_log"] = seg
    out["__nstar"] = np.array([nstar])
    out["__stream_names"] = np.array(list(streams.keys()))
    fpath = OUT_DIR / f"P{pid:02d}_synthetic.npz"
    np.savez_compressed(fpath, **out)
    return fpath


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"Synthetic data generator")
    print(f"  SIGNAL_STRENGTH = {SIGNAL_STRENGTH}  "
          f"({'PURE NOISE — expect chance' if SIGNAL_STRENGTH == 0 else 'signal present'})")
    print(f"  participants    = {len(NSTAR_ASSIGNMENT)}")
    print(f"  n* distribution = {dict(zip(*np.unique(NSTAR_ASSIGNMENT, return_counts=True)))}")
    print(f"  output          = {OUT_DIR}")
    print()
    for i, nstar in enumerate(NSTAR_ASSIGNMENT, start=1):
        streams, seg = build_participant(i, nstar, rng)
        fpath = save_participant(i, nstar, streams, seg)
        dur = streams["BioRadio_EDA"]["ts"][-1]
        print(f"  P{i:02d}  n*={nstar}  duration={dur:6.1f}s  -> {fpath.name}")
    print(f"\nDone. {len(NSTAR_ASSIGNMENT)} files written to {OUT_DIR}")
    print("\nReminder: these are FAKE. No number from them is a result.")


if __name__ == "__main__":
    main()
