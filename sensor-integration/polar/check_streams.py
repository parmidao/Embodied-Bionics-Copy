from pylsl import resolve_streams, StreamInlet

streams = resolve_streams(wait_time=3.0)
print(f"Found {len(streams)} stream(s):")
for s in streams:
    print(f"  {s.name():10} type={s.type():4} ch={s.channel_count()} rate={s.nominal_srate()}")

ecg = [s for s in streams if s.name() == "Polar_ECG"]
if ecg:
    inlet = StreamInlet(ecg[0])
    print("\nFirst ECG samples (uV):")
    for _ in range(5):
        sample, ts = inlet.pull_sample()
        print(f"  t={ts:.3f}  uV={sample[0]:.1f}")