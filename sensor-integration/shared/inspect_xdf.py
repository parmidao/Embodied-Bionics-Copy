import sys
import pyxdf

path = sys.argv[1] if len(sys.argv) > 1 else "recording.xdf"
streams, _ = pyxdf.load_xdf(path)
print(f"\nLoaded {path}  —  {len(streams)} stream(s)\n")
for s in streams:
    info = s["info"]
    name = info["name"][0]
    ch = int(info["channel_count"][0])
    nominal = float(info["nominal_srate"][0])
    ts = s["time_stamps"]
    n = len(ts)
    if n > 1:
        span = ts[-1] - ts[0]
        eff = (n - 1) / span if span > 0 else 0.0
        print(f"{name:12} ch={ch}  nominal={nominal:8.2f}Hz  "
              f"effective={eff:8.2f}Hz  n={n:7d}  span={span:5.1f}s  "
              f"t=[{ts[0]:.2f}..{ts[-1]:.2f}]")
    else:
        print(f"{name:12} ch={ch}  nominal={nominal:8.2f}Hz  n={n}")
print()