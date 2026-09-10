# Sensor Integration - Lead: Parmida. Support: Negar
# Sensor Integration — Real-time LSL bridges

Each sensor streams into Lab Streaming Layer (LSL); LabRecorder captures all
streams into one synchronized .xdf.

- polar/    Polar H10 -> LSL (ECG 130 Hz, HR, RR) over BLE via bleak
- delsys/   Delsys Trigno EMG -> LSL via the Delsys .NET API (see delsys/README.md)
- bioradio/ (planned) BioRadio ECG/EDA -> LSL
- shared/   fake_emg_lsl.py (LSL test sender), inspect_emg.py (.xdf viewer)

Install deps:  pip install -r requirements.txt