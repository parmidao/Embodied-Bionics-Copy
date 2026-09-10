"""
monitor_streams.py — resolve every live LSL stream and print a refreshing
one-line-per-stream readout, so you can confirm they're all publishing at once.
Ctrl+C to stop.
"""
import time
from pylsl import resolve_streams, StreamInlet

print("Resolving LSL streams (2s)...")
infos = resolve_streams(wait_time=2.0)
if not infos:
    raise SystemExit("No LSL streams found. Are the bridges running?")

inlets = [(info, StreamInlet(info, max_buflen=4)) for info in infos]
print(f"Found {len(inlets)} stream(s): " + ", ".join(i.name() for i, _ in inlets))
print("Live readout (observed rate + latest first-channel value) — Ctrl+C to stop.\n")

counts = {i.name(): 0 for i, _ in inlets}
latest = {i.name(): None for i, _ in inlets}
last = time.time()

try:
    while True:
        for info, inlet in inlets:
            samples, _ = inlet.pull_chunk(timeout=0.0)
            if samples:
                counts[info.name()] += len(samples)
                latest[info.name()] = samples[-1]
        now = time.time()
        if now - last >= 1.0:
            dt = now - last
            parts = []
            for info, _ in inlets:
                name = info.name()
                rate = counts[name] / dt
                val = latest[name]
                v0 = f"{val[0]:+8.3f}" if val else "   --   "
                parts.append(f"{name}={rate:6.1f}Hz v0={v0}")
                counts[name] = 0
            print("  ".join(parts))
            last = now
except KeyboardInterrupt:
    print("\nStopped.")