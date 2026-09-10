import sys
import numpy as np
import pyxdf
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "recording.xdf"
streams, _ = pyxdf.load_xdf(path)
streams = [s for s in streams if len(s["time_stamps"]) > 1]
t0 = min(s["time_stamps"][0] for s in streams)

fig, axes = plt.subplots(len(streams), 1, sharex=True,
                         figsize=(12, 2.0 * len(streams)))
if len(streams) == 1:
    axes = [axes]
for ax, s in zip(axes, streams):
    name = s["info"]["name"][0]
    t = np.asarray(s["time_stamps"]) - t0
    y = np.asarray(s["time_series"], dtype=float)
    ax.plot(t, y[:, 0], lw=0.6)
    ax.set_ylabel(name, fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("time (s) — shared LSL clock")
fig.suptitle("Multimodal .xdf on a shared time axis")
plt.tight_layout()
plt.show()