"""
sync_test_markers.py — emits SYNC_TEST markers into an LSL stream so they land
in the .xdf at START / MIDDLE / END. Write side; sync_test.py is the read side.
Covers dry-run items DR-02 (start), DR-03 (middle), DR-04 (end).
"""
import sys, time
try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError:
    print("ERROR: pylsl not installed. Run:")
    print("    .\\.venv\\Scripts\\python.exe -m pip install pylsl")
    sys.exit(1)

STREAM_NAME = "SYNC_TEST"

def make_outlet():
    info = StreamInfo(name=STREAM_NAME, type="Markers", channel_count=1,
                      nominal_srate=0, channel_format="string",
                      source_id="sync_test_marker_emitter_v1")
    return StreamOutlet(info)

def push(outlet, label):
    t = local_clock()
    outlet.push_sample([label])
    print(f"  -> pushed {label!r} at LSL time {t:.4f}")
    return t

def main():
    print("SYNC_TEST marker emitter")
    print("-" * 60)
    print("Creating LSL outlet named 'SYNC_TEST' ...")
    outlet = make_outlet()
    time.sleep(0.5)
    print("Outlet is live.\n")
    print("NOW: in LabRecorder, refresh streams, tick SYNC_TEST, Start recording.")
    print("Then come back here.\n")
    steps = ["SYNC_TEST_START", "SYNC_TEST_MIDDLE", "SYNC_TEST_END"]
    when = {0: "at the START of the run", 1: "at the MIDDLE of the run",
            2: "at the END of the run"}
    pushed = []
    for i, label in enumerate(steps):
        input(f"[{i+1}/3] Press ENTER {when[i]} to push {label} ... ")
        pushed.append((label, push(outlet, label)))
        print()
    print("-" * 60)
    print("All three markers pushed:")
    for label, t in pushed:
        print(f"  {label:20s}  LSL time {t:.4f}")
    print("\nNow STOP the LabRecorder recording, then run:")
    print("  .\\.venv\\Scripts\\python.exe sync_test.py <that>.xdf")
    print("\nKeeping outlet alive 3 s to flush...")
    time.sleep(3)
    print("Done.")

if __name__ == "__main__":
    main()
