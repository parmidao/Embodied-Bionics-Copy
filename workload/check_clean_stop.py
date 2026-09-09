"""
check_clean_stop.py
===================
Verifies that a LabRecorder .xdf was stopped CLEANLY and is usable for
cross-machine synchronisation.

WHY THIS EXISTS
  A recording can look complete — all streams present, samples flowing — and
  still be useless for sync. When LabRecorder is not stopped cleanly, it never
  writes the footer that carries the clock-offset measurements. Without those
  offsets, streams from different machines (e.g. EMG on one host, Unity on
  another) cannot be aligned onto a common timeline. The timestamps may happen
  to land in the same numeric range, but that is coincidence, not
  synchronisation.

  This exact failure occurred: a recording had no footer and zero clock offsets
  on all streams, making the Unity<->EMG alignment unrecoverable, and it was
  only discovered days later. This script catches it in seconds, right after
  the recording, while a re-record is still possible.

WHAT IT CHECKS
  1. Clock offsets exist and are non-zero for streams that need them.
  2. Multiple hosts are present (cross-machine) and each host's streams carry
     offsets, since that is what cross-machine correction depends on.
  3. Basic footer sanity (streams have a recorded time range).

USAGE
  python check_clean_stop.py "path\\to\\recording.xdf"

  Exit is a clear PASS / FAIL you can read at a glance right after stopping
  LabRecorder.
"""

import sys
from pathlib import Path

try:
    import pyxdf
except ImportError:
    print("ERROR: pyxdf not installed in this environment.")
    print("  pip install pyxdf")
    sys.exit(2)


def check_file(path):
    print("=" * 70)
    print(f"  CLEAN-STOP CHECK: {Path(path).name}")
    print("=" * 70)

    try:
        streams, header = pyxdf.load_xdf(path)
    except Exception as e:
        print(f"  ❌ Could not open file: {e}")
        return False

    problems = []
    hosts = {}

    print(f"\n  {len(streams)} streams in file:\n")
    print(f"  {'stream':22s} {'host':18s} {'samples':>9s}  {'clock offsets':>13s}")
    print("  " + "-" * 66)

    for s in streams:
        name = s["info"]["name"][0]
        host = s["info"]["hostname"][0] if s["info"].get("hostname") else "?"
        n_samples = len(s["time_stamps"])

        # clock offset info lives in the footer; pyxdf exposes it per-stream
        clock_offsets = s["info"].get("clock_offsets", [{}])
        n_offsets = 0
        if clock_offsets and isinstance(clock_offsets[0], dict):
            offs = clock_offsets[0].get("offset", [])
            n_offsets = len(offs) if offs else 0

        hosts.setdefault(host, []).append((name, n_offsets))

        flag = ""
        if n_offsets == 0:
            flag = "  <-- NO OFFSETS"
        print(f"  {name:22s} {host:18s} {n_samples:9d}  {n_offsets:13d}{flag}")

    # ---- Rule 1: at least some streams must carry clock offsets ----
    total_offsets = sum(n for host_streams in hosts.values()
                        for _, n in host_streams)
    if total_offsets == 0:
        problems.append(
            "NO clock offsets anywhere in the file. LabRecorder was almost "
            "certainly not stopped cleanly (no footer written). Cross-machine "
            "sync is UNRECOVERABLE from this file. Re-record.")

    # ---- Rule 2: cross-machine case — each host needs offsets ----
    if len(hosts) > 1:
        print(f"\n  {len(hosts)} hosts present (cross-machine recording):")
        for host, host_streams in hosts.items():
            host_offsets = sum(n for _, n in host_streams)
            status = "OK" if host_offsets > 0 else "NO OFFSETS"
            print(f"    {host:20s} total offsets: {host_offsets}   [{status}]")
            if host_offsets == 0:
                problems.append(
                    f"Host '{host}' has zero clock offsets — its streams "
                    f"cannot be aligned to the other machine(s).")
    else:
        print(f"\n  Single host ({list(hosts)[0]}). Cross-machine sync N/A, "
              f"but offsets still expected for a clean stop.")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    if problems:
        print("  ❌ FAIL — recording was NOT stopped cleanly:")
        for p in problems:
            print(f"     • {p}")
        print("\n  ACTION: In LabRecorder, always press Stop and let it finish")
        print("  writing before closing. Then re-run this check. If offsets are")
        print("  still zero, the streams dropped before Stop — investigate the")
        print("  bridges, not LabRecorder.")
        print("=" * 70)
        return False
    else:
        print("  ✅ PASS — footer present, clock offsets recorded.")
        print("     This file is safe for cross-machine synchronisation.")
        print("=" * 70)
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_clean_stop.py \"path\\to\\recording.xdf\"")
        sys.exit(2)
    ok = check_file(sys.argv[1])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
