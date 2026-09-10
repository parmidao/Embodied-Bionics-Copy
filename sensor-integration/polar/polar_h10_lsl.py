"""
polar_h10_lsl.py — streams Polar H10 ECG (130 Hz), HR, and RR intervals to LSL.
Outlets: Polar_ECG (130 Hz, uV) | Polar_HR (irregular, bpm) | Polar_RR (irregular, ms)
"""
import asyncio
from bleak import BleakClient
from pylsl import StreamInfo, StreamOutlet, local_clock

ADDRESS = "A0:9E:1A:EA:73:DB"          # your Polar H10

HR_UUID     = "00002a37-0000-1000-8000-00805f9b34fb"   # Heart Rate Measurement
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"   # PMD control point
PMD_DATA    = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"   # PMD data
ECG_START   = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])
ECG_RATE    = 130

def make_outlet(name, sigtype, n, rate, unit):
    info = StreamInfo(name, sigtype, n, rate, "float32", f"polar_{name}")
    chns = info.desc().append_child("channels")
    c = chns.append_child("channel")
    c.append_child_value("label", sigtype)
    c.append_child_value("unit", unit)
    c.append_child_value("type", sigtype)
    return StreamOutlet(info)

ecg_outlet = make_outlet("Polar_ECG", "ECG", 1, ECG_RATE, "microvolts")
hr_outlet  = make_outlet("Polar_HR",  "HR",  1, 0,        "bpm")
rr_outlet  = make_outlet("Polar_RR",  "RR",  1, 0,        "ms")

_ecg_start_clock = None

def parse_hr(data):
    flags = data[0]
    hr16   = flags & 0x01
    energy = (flags >> 3) & 0x01
    rr_present = (flags >> 4) & 0x01
    i = 1
    if hr16:
        hr = int.from_bytes(data[i:i+2], "little"); i += 2
    else:
        hr = data[i]; i += 1
    if energy:
        i += 2
    rrs = []
    if rr_present:
        while i + 1 < len(data):
            rr_raw = int.from_bytes(data[i:i+2], "little"); i += 2
            rrs.append(rr_raw / 1024.0 * 1000.0)   # 1/1024 s -> ms
    return hr, rrs

def parse_ecg(data):
    return [int.from_bytes(data[i:i+3], "little", signed=True)
            for i in range(10, len(data), 3)]

def on_hr(_, data):
    hr, rrs = parse_hr(data)
    hr_outlet.push_sample([float(hr)])
    for rr in rrs:
        rr_outlet.push_sample([float(rr)])

def on_ecg(_, data):
    global _ecg_start_clock
    if not data or data[0] != 0x00:
        return
    if _ecg_start_clock is None:
        _ecg_start_clock = local_clock()
    if local_clock() - _ecg_start_clock < 1.0:    # drop ~1s settling
        return
    samples = parse_ecg(data)
    if samples:
        ecg_outlet.push_chunk([[float(s)] for s in samples])

async def main():
    async with BleakClient(ADDRESS, timeout=30.0) as client:
        print("Connected:", client.is_connected)
        try:
            await client.pair()                    # Windows: unlocks PMD (fixes 0x05)
        except Exception as e:
            print("pair() note:", e)
        await client.start_notify(HR_UUID, on_hr)
        await client.start_notify(PMD_DATA, on_ecg)
        try:
            await client.write_gatt_char(PMD_CONTROL, ECG_START, response=True)
            print("ECG stream requested.")
        except Exception as e:
            print("ECG start failed (HR/RR still stream):", e)
        print("Streaming Polar_ECG / Polar_HR / Polar_RR. Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")