"""
bioradio_discover.py — connect to the BioRadio, start streaming, and print
the available signal groups, signals, sample rates, and a sample read.
"""
import os, sys, time
from pythonnet import load
load("netfx")
import clr

DLL_DIR = r"C:\Program Files (x86)\Great Lakes NeuroTechnologies\BioCapture"
sys.path.append(DLL_DIR)
os.add_dll_directory(DLL_DIR)
clr.AddReference("BioRadioAPI")

from GLNeuroTech.Devices.BioRadio import BioRadioDeviceManager

mgr = BioRadioDeviceManager()
print("Scanning for BioRadio over the dongle...")
devices = list(mgr.DiscoverBluetoothDevices())
if not devices:
    print("No BioRadio found. Powered on? Dongle in? BioCapture fully closed?")
    raise SystemExit(1)

info = devices[0]
mac_raw = getattr(info, "MacId", None)
print("Device MacId:", repr(mac_raw), "| type:", type(mac_raw).__name__)

# GetBluetoothDevice expects an Int64 MAC, but MacId comes back as a string,
# so convert it (strip any : or - and read it as hex), with a decimal fallback.
device = None
attempts = []
if mac_raw is not None:
    s = str(mac_raw).replace(":", "").replace("-", "").strip()
    for parse in (lambda: int(s, 16), lambda: int(s)):
        try:
            attempts.append(parse())
        except ValueError:
            pass

for val in attempts:
    try:
        device = mgr.GetBluetoothDevice(val)
        print(f"Connected using MAC as int: {val}")
        break
    except Exception as e:
        print(f"  GetBluetoothDevice({val}) failed: {type(e).__name__}")

if device is None:
    device = mgr.GetBluetoothDevice(info)   # last resort: pass the info object
    print("Connected using the device-info object.")

device.StartAcquisition()
print("Acquisition started. Accumulating ~1.5s...\n")
time.sleep(1.5)

group_names = ["BioPotentialSignals", "AuxiliarySignals",
               "PulseOximeterSignals", "AccelerometerSignals", "EventSignals"]

for gname in group_names:
    group = getattr(device, gname, None)
    if group is None:
        continue
    signals = list(group)
    if not signals:
        continue
    print(f"[{gname}] {len(signals)} signal(s):")
    for sig in signals:
        rate = None
        for r in ("SamplesPerSecond", "SampleRate", "SamplingRate"):
            rate = getattr(sig, r, None)
            if rate is not None:
                break
        arr = list(sig.GetScaledValueArray())
        name = getattr(sig, "Name", "?")
        print(f"    name={name!r:16} rate={rate}  new_samples={len(arr)}  first={arr[:3]}")
    print()

device.StopAcquisition()
print("Stopped.")