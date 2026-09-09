#!/usr/bin/env python3
"""
SYNC_TEST -- clock synchronisation check
P4 workload study -- HERO Lab

Freeze-list row 12: synchronisation drift across streams must be <= 5 ms,
with SYNC_TEST verification at session start, middle and end.

What this measures
------------------
Every device keeps its own clock. LSL continuously measures the offset
between each stream's clock and the recording machine's clock and stores
those measurements in the XDF footer. This script reads them.

How to read the result
----------------------
Streams running on a DIFFERENT MACHINE from LabRecorder show a large
constant offset -- in this study the Unity streams sit roughly 83 seconds
from the Python bridges, because the two computers have different clock
origins. That is normal and is NOT a synchronisation failure: pyxdf fits and
removes both the constant offset and any steady drift when the file loads.

What cannot be removed is the SCATTER in those measurements -- the residual
after a linear fit. That is the irreducible uncertainty in aligning two
streams, and that is what the 5 ms tolerance applies to.

The verdict below is therefore based on residual scatter and drift rate, not
on the raw offset between machines.

What this does NOT do
---------------------
This reads the offsets LSL recorded about itself. It does not verify that a
physical event lands at the same timestamp in two streams -- that needs the
manual SYNC_TEST (a hard clench producing an EMG burst alongside a marker,
at session start, middle and end). Both are needed: this one runs
automatically on every recording, the manual one confirms LSL's own numbers
can be trusted.

A note on the tolerance
-----------------------
5 ms is finer than one sample for several streams: ECG at 130 Hz is 7.7 ms
per sample, gaze at 90 Hz is 11.1 ms. The tolerance applies to CLOCK
ALIGNMENT, not to sample-level resolution, which is bounded below by the
sample period. Both are reported.

Usage
-----
    python sync_test.py <file.xdf>
    python sync_test.py <directory>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyxdf

TOLERANCE_MS = 5.0                  # freeze-list row 12
DRIFT_RATE_WARN_MS_PER_MIN = 2.0    # flagged, not failed -- steady drift is removed
MACHINE_GROUP_TOLERANCE_S = 1.0     # offsets within 1 s are treated as one machine

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def offsets_for(stream) -> tuple[np.ndarray, np.ndarray]:
    """Pull LSL's recorded clock offsets from a stream's footer."""
    try:
        clock = stream["footer"]["info"]["clock_offsets"][0]["offset"]
    except (KeyError, IndexError, TypeError):
        return np.array([]), np.array([])

    times, values = [], []
    for entry in clock:
        try:
            times.append(float(entry["time"][0]))
            values.append(float(entry["value"][0]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return np.asarray(times), np.asarray(values)


def linear_residual_ms(times: np.ndarray, offsets: np.ndarray) -> float | None:
    """
    Scatter around the fitted offset line, in ms.

    pyxdf corrects the constant offset and the steady slope. What it cannot
    correct is how far individual measurements sit from that line, so this
    is the number that actually limits cross-stream alignment.
    """
    if len(times) < 3:
        return None
    fit = np.polyfit(times, offsets, 1)
    residual = offsets - np.polyval(fit, times)
    return float(np.std(residual) * 1000.0)


def drift_rate_ms_per_min(times: np.ndarray, offsets: np.ndarray) -> float | None:
    if len(times) < 3 or times[-1] - times[0] <= 0:
        return None
    slope = np.polyfit(times, offsets, 1)[0]      # s of offset per s
    return float(slope * 1000.0 * 60.0)


def sample_period_ms(stream) -> float | None:
    ts = np.asarray(stream["time_stamps"])
    if len(ts) < 2:
        return None
    span = float(ts[-1] - ts[0])
    if span <= 0:
        return None
    return 1000.0 * span / (len(ts) - 1)


def group_by_machine(rows: list[dict]) -> dict[int, list[dict]]:
    """
    Cluster streams by clock base. Streams whose mean offsets sit within
    MACHINE_GROUP_TOLERANCE_S of each other are almost certainly on the same
    computer.
    """
    measured = sorted((r for r in rows if r["n"] > 0), key=lambda r: r["mean_s"])
    groups: dict[int, list[dict]] = {}
    gid = 0
    for r in measured:
        if gid == 0:
            gid = 1
            groups[gid] = [r]
        elif abs(r["mean_s"] - groups[gid][-1]["mean_s"]) <= MACHINE_GROUP_TOLERANCE_S:
            groups[gid].append(r)
        else:
            gid += 1
            groups[gid] = [r]
    return groups


def check_file(path: Path) -> bool:
    print(f"\n{'=' * 78}")
    print(f"  {path.name}")
    print(f"{'=' * 78}")

    try:
        streams, _ = pyxdf.load_xdf(str(path))
    except Exception as exc:
        print(f"  FAIL  could not load: {exc}")
        return False

    rows = []
    for s in streams:
        name = s["info"]["name"][0]
        n_samples = len(np.asarray(s["time_stamps"]))
        times, offsets = offsets_for(s)

        if len(offsets) == 0:
            rows.append(dict(name=name, n=0, n_samples=n_samples, mean_s=None,
                             residual=None, rate=None,
                             period=sample_period_ms(s)))
            continue

        rows.append(dict(
            name=name,
            n=len(offsets),
            n_samples=n_samples,
            mean_s=float(np.mean(offsets)),
            residual=linear_residual_ms(times, offsets),
            rate=drift_rate_ms_per_min(times, offsets),
            period=sample_period_ms(s),
        ))

    print(f"\n  {'stream':<18} {'offs':>5} {'samples':>8}  {'residual':>10}  "
          f"{'drift':>13}  {'1 sample':>9}")
    print(f"  {'-' * 76}")
    for r in rows:
        period = f"{r['period']:.1f}ms" if r["period"] else "--"
        if r["n"] == 0:
            print(f"  {r['name']:<18} {0:>5} {r['n_samples']:>8}  "
                  f"{'no offsets':>10}  {'--':>13}  {period:>9}")
            continue
        resid = f"{r['residual']:.3f}ms" if r["residual"] is not None else "--"
        rate = f"{r['rate']:+.2f} ms/min" if r["rate"] is not None else "--"
        print(f"  {r['name']:<18} {r['n']:>5} {r['n_samples']:>8}  {resid:>10}  "
              f"{rate:>13}  {period:>9}")

    measured = [r for r in rows if r["n"] > 0]
    if not measured:
        print("\n  RESULT: INCONCLUSIVE -- no clock offsets in this file.")
        print("          LabRecorder was likely run with clock sync disabled.")
        return False

    groups = group_by_machine(rows)
    print("\n  Clock groups (streams sharing a clock base):")
    for gid, members in groups.items():
        base = float(np.mean([m["mean_s"] for m in members]))
        names = ", ".join(m["name"] for m in members)
        print(f"    group {gid}: base {base:+.3f}s -- {names}")
    if len(groups) > 1:
        print("    More than one group means more than one machine. The constant")
        print("    offset between them is expected and removed by pyxdf on load.")

    worst_resid = max((r["residual"] for r in measured
                       if r["residual"] is not None), default=0.0)
    worst_rate = max((abs(r["rate"]) for r in measured
                      if r["rate"] is not None), default=0.0)

    print()
    print(f"  Worst residual scatter : {worst_resid:.3f} ms "
          f"(tolerance {TOLERANCE_MS:.1f} ms)")
    print(f"  Fastest drift rate     : {worst_rate:.2f} ms/min")

    # Streams alive for clock sync but delivering nothing: the July signature.
    alive_but_empty = [r for r in measured if r["n_samples"] == 0]

    status = PASS
    if worst_resid > TOLERANCE_MS:
        status = FAIL
    elif worst_rate > DRIFT_RATE_WARN_MS_PER_MIN:
        status = WARN

    print()
    if status == FAIL:
        print(f"  RESULT: FAIL -- residual scatter {worst_resid:.3f} ms exceeds "
              f"the {TOLERANCE_MS:.0f} ms tolerance")
    elif status == WARN:
        print(f"  RESULT: PASS WITH WARNING -- alignment within tolerance, "
              f"drifting at {worst_rate:.2f} ms/min")
    else:
        print(f"  RESULT: PASS -- alignment within {TOLERANCE_MS:.0f} ms "
              f"after correction")

    if alive_but_empty:
        print()
        print("  DATA WARNING (not a sync problem):")
        for r in alive_but_empty:
            print(f"    {r['name']}: exchanged {r['n']} clock offsets, "
                  f"delivered 0 samples.")
        print("    The bridge process was running and synchronising throughout;")
        print("    the device sent nothing. Investigate the device connection,")
        print("    not the bridge script.")

    print()
    print("  Note: the tolerance applies to clock alignment. Sample-level")
    print("  resolution is bounded by each stream's sample period, last column.")

    return status != FAIL


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    target = Path(sys.argv[1])
    files = sorted(target.rglob("*.xdf")) if target.is_dir() else [target]
    if not files:
        print(f"No .xdf files under {target}")
        sys.exit(2)

    ok = [check_file(f) for f in files]

    print(f"\n{'=' * 78}")
    print(f"  {sum(ok)}/{len(ok)} recording(s) within tolerance")
    print(f"{'=' * 78}\n")

    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
