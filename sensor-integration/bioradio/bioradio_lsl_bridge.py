"""
bioradio_lsl_bridge.py — stream the BioRadio GSR/EDA channel to LSL.
Confirmed by discovery: one signal "Skin Con" in BioPotentialSignals @ 250 Hz.
Publishes one LSL stream: BioRadio_EDA (1 channel, 250 Hz, float32).
"""
import os, sys, time, math
from pythonnet import load
load("netfx")
import clr

DLL_DIR = r"C:\Program Files (x86)\Great Lakes NeuroTechnologies\BioCapture"
sys.path.append(DLL_DIR)
os.add_dll_directory(DLL_DIR)
clr.AddReference("BioRadioAPI")

from GLNeuroTech.Devices.BioRadio import BioRadioDeviceManager
from pylsl import StreamInfo, StreamOutlet

RATE = 250.0
STREAM_NAME = "BioRadio_EDA"

def connect():
    mgr = BioRadioDeviceManager()
    print("Scanning for BioRadio...")
    devices = list(mgr.DiscoverBluetoothDevices())
    if not devices:
        raise SystemExit("No BioRadio found. Powered on? Dongle in? BioCapture closed?")
    info = devices[0]
    mac_int = int(str(info.MacId).replace(":", "").replace("-", "").strip(), 16)
    device = mgr.GetBluetoothDevice(mac_int)
    print(f"Connected to {info.MacId}.")
    return device

def main():
    device = connect()
    device.StartAcquisition()
    print("Acquisition started.")
    time.sleep(1.0)   # let the channel settle

    # The GSR channel lives in BioPotentialSignals (confirmed by discovery).
    gsr = list(device.BioPotentialSignals)[0]
    label = str(getattr(gsr, "Name", "GSR"))

    info = StreamInfo(STREAM_NAME, "EDA", 1, RATE, "float32", "bioradio_eda_skincon")
    ch = info.desc().append_child("channels").append_child("channel")
    ch.append_child_value("label", label)
    ch.append_child_value("unit", "uS")          # nominal; verify against BioCapture
    ch.append_child_value("type", "EDA")
    outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
    print(f"[LSL] {STREAM_NAME}: 1 ch @ {RATE} Hz (label={label!r}). Streaming. Ctrl+C to stop.")

    last_good = 0.0
    try:
        while True:
            samples = list(gsr.GetScaledValueArray())
            if samples:
                clean = []
                for v in samples:
                    v = float(v)
                    if math.isnan(v) or math.isinf(v):
                        v = last_good            # carry forward over NaN/inf
                    else:
                        last_good = v
                    clean.append([v])            # 1 channel -> one value per sample
                outlet.push_chunk(clean)
            time.sleep(0.04)                     # poll ~25x/sec
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        device.StopAcquisition()
        print("Stopped.")

if __name__ == "__main__":
    main()
    