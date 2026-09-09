#!/usr/bin/env python3
"""
Recording QC check
P4 workload study -- HERO Lab

Verifies that every expected LSL stream in an .xdf is not merely PRESENT but
actually CARRYING DATA.

Why this exists
---------------
In the July BBT pilot, Polar_ECG and BioRadio_EDA appeared in three
recordings with the correct stream name and the correct nominal rate, and
zero samples. Polar_HR and Polar_RR kept streaming in the same files, so the
device had not disconnected -- the ECG subscription had dropped while the
LSL outlet stayed open. Separately, OVR_Gaze_Stream ran at a perfect
90.00 Hz while containing no valid data.

A check that asks "is the stream there?" passes in both cases. This script
checks sample counts, effective rates, and whether the values are
degenerate.

Usage
-----
    python check_recording.py <file.xdf>
    python check_recording.py <directory>      # checks every .xdf inside

Exit code is 1 if any required check FAILS, so this can gate a session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyxdf

# ---------------------------------------------------------------------------
# EXPECTED STREAMS
#
# expected_rate is the rate we actually believe, which is not always the
# nominal rate declared by the device. RawEMG declares 1111.11 Hz and
# delivers ~1248 Hz; until that is resolved at the source, 1248 is the
# number to check against and the number features must be computed with.
# ---------------------------------------------------------------------------

EXPECTED = {
    "Polar_ECG":       dict(rate=130.0,  channels=1, required=True,  kind="continuous"),
    "Polar_HR":        dict(rate=1.0,    channels=1, required=True,  kind="continuous"),
    "Polar_RR":        dict(rate=None,   channels=1, required=True,  kind="irregular"),
    "BioRadio_EDA":    dict(rate=250.0,  channels=1, required=True,  kind="continuous"),
    "OVR_Gaze_Stream": dict(rate=90.0,   channels=7, required=False, kind="continuous"),
    "RawEMG":          dict(rate=1248.0, channels=4, required=True,  kind="continuous"),
    "VREventMarkers":  dict(rate=None,   channels=1, required=True,  kind="markers"),
    "ExperimentData":  dict(rate=None,   channels=1, required=False, kind="irregular"),
}

RATE_TOLERANCE = 0.05      # effective rate must be within 5% of expected
MIN_SPAN_S = 20.0          # a usable recording is at least this long
GAP_FACTOR = 10.0          # a gap is an interval > 10x the median interval
MIN_MARKERS = 1            # VREventMarkers must carry at least this many

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Result:
    def __init__(self, stream: str):
        self.stream = stream
        self.status = PASS
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.status = FAIL
        self.notes.append(msg)

    def warn(self, msg: str) -> None:
        if self.status != FAIL:
            self.status = WARN
        self.notes.append(msg)


def is_numeric(series) -> bool:
    """
    Marker and event streams carry strings, not numbers. ExperimentData in
    the July pilot holds entries like 'TIMER_START|BBT|firstTouch=true'.
    Those must not be pushed through the numeric degeneracy checks.
    """
    try:
        np.asarray(series, dtype=float)
        return True
    except (ValueError, TypeError):
        return False


def describe_values(series) -> list[str]:
    """
    Look for the failure modes that a presence check misses: all-zero
    channels, flat channels, and non-finite values.
    """
    problems = []
    data = np.asarray(series, dtype=float)
    if data.ndim == 1:
        data = data[:, None]

    for c in range(data.shape[1]):
        col = data[:, c]
        finite = np.isfinite(col)

        if not finite.any():
            problems.append(f"ch{c}: no finite values")
            continue

        bad_frac = 1.0 - finite.mean()
        if bad_frac > 0.01:
            problems.append(f"ch{c}: {bad_frac:.1%} non-finite")

        vals = col[finite]
        if np.allclose(vals, 0.0):
            problems.append(f"ch{c}: all zero")
        elif vals.std() == 0.0:
            problems.append(f"ch{c}: constant ({vals[0]:g})")

    return problems


def find_gaps(timestamps: np.ndarray) -> list[tuple[float, float]]:
    """Return (position_in_seconds, gap_length) for suspicious dropouts."""
    if len(timestamps) < 10:
        return []
    diffs = np.diff(timestamps)
    median = float(np.median(diffs))
    if median <= 0:
        return []
    idx = np.where(diffs > GAP_FACTOR * median)[0]
    return [(float(timestamps[i] - timestamps[0]), float(diffs[i])) for i in idx]


def check_stream(name: str, spec: dict, stream) -> Result:
    r = Result(name)

    if stream is None:
        if spec["required"]:
            r.fail("stream absent from file")
        else:
            r.warn("stream absent from file")
        return r

    ts = np.asarray(stream["time_stamps"])
    n = len(ts)

    # The July failure mode: present, correctly named, zero samples.
    if n == 0:
        r.fail("stream present but ZERO samples -- outlet open, no data")
        return r

    if spec["kind"] == "markers" or not is_numeric(stream["time_series"]):
        if spec["kind"] == "markers" and n < MIN_MARKERS:
            r.fail(f"only {n} event(s) -- block labels cannot be recovered")
            return r

        span = float(ts[-1] - ts[0]) if n > 1 else 0.0
        r.notes.append(f"{n} events over {span:.1f}s")

        # Show what kinds of events are present -- this is where block
        # boundaries and task labels live.
        try:
            labels = [str(row[0]) if isinstance(row, (list, np.ndarray)) else str(row)
                      for row in stream["time_series"]]
            kinds = sorted({lab.split("|")[0] for lab in labels})
            preview = ", ".join(kinds[:6])
            if len(kinds) > 6:
                preview += f", +{len(kinds) - 6} more"
            r.notes.append(f"event types: {preview}")
        except Exception:
            pass
        return r

    ch = int(stream["info"]["channel_count"][0])
    if ch != spec["channels"]:
        r.fail(f"channel count {ch}, expected {spec['channels']}")

    span = float(ts[-1] - ts[0]) if n > 1 else 0.0
    if span < MIN_SPAN_S:
        r.warn(f"span only {span:.1f}s")

    if spec["rate"] is not None and n > 1 and span > 0:
        eff = (n - 1) / span
        drift = abs(eff - spec["rate"]) / spec["rate"]
        if drift > RATE_TOLERANCE:
            r.fail(f"effective {eff:.1f}Hz vs expected {spec['rate']:.1f}Hz "
                   f"({drift:.1%} off)")

        nominal = float(stream["info"]["nominal_srate"][0])
        if nominal > 0 and abs(nominal - eff) / eff > RATE_TOLERANCE:
            r.warn(f"declared {nominal:.1f}Hz but delivers {eff:.1f}Hz "
                   f"-- features must use the effective rate")

    for gap_at, gap_len in find_gaps(ts):
        r.warn(f"gap of {gap_len:.1f}s at t+{gap_at:.1f}s")

    for problem in describe_values(stream["time_series"]):
        r.fail(problem)

    return r


def check_file(path: Path) -> bool:
    print(f"\n{'=' * 78}")
    print(f"  {path.name}")
    print(f"{'=' * 78}")

    try:
        streams, _ = pyxdf.load_xdf(str(path))
    except Exception as exc:
        print(f"  FAIL  could not load: {exc}")
        return False

    by_name = {s["info"]["name"][0]: s for s in streams}

    for extra in sorted(set(by_name) - set(EXPECTED)):
        print(f"  note  unexpected stream in file: {extra}")

    results = [check_stream(name, spec, by_name.get(name))
               for name, spec in EXPECTED.items()]

    print(f"\n  {'stream':<18} {'status':<7} detail")
    print(f"  {'-' * 74}")
    for r in results:
        detail = r.notes[0] if r.notes else "ok"
        print(f"  {r.stream:<18} {r.status:<7} {detail}")
        for extra in r.notes[1:]:
            print(f"  {'':<18} {'':<7} {extra}")

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]

    print()
    if failures:
        print(f"  RESULT: FAIL -- {len(failures)} stream(s) unusable: "
              f"{', '.join(r.stream for r in failures)}")
    elif warnings:
        print(f"  RESULT: PASS WITH WARNINGS -- "
              f"{', '.join(r.stream for r in warnings)}")
    else:
        print("  RESULT: PASS -- all streams present, sampled and non-degenerate")

    return not failures


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(target.rglob("*.xdf"))
        if not files:
            print(f"No .xdf files under {target}")
            sys.exit(2)
    else:
        files = [target]

    ok = [check_file(f) for f in files]

    print(f"\n{'=' * 78}")
    print(f"  {sum(ok)}/{len(ok)} recording(s) passed")
    print(f"{'=' * 78}\n")

    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
