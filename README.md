# Multimodal Biosignal Integration & Cognitive Workload Modeling

## Context & Attribution
This repository contains standalone copies of my direct engineering and research contributions developed for multimodal biosignal acquisition, hardware synchronization, and real-time workload modeling. 

The modules here represent the sub-pipelines I designed and implemented as part of my undergraduate research work in adaptive human-machine systems. To adhere to research confidentiality protocols, all proprietary lab datasets, third-party vendor SDK binaries, and broader experimental suite code remain private.

## Included Contributions

### 1. Sensor Integration (`sensor-integration/`)
- Real-time synchronization and streaming pipelines using **Lab Streaming Layer (LSL)**.
- Hardware interfacing and acquisition scripts for multimodal sensor streams (EMG, EEG, eye-tracking).
- Robust stream clock synchronization and sample buffer management.

### 2. Cognitive Workload Pipeline (`workload/`)
- Feature extraction and time-series signal processing from real-time data streams.
- Machine learning classification architectures for continuous cognitive workload decoding.
- Pipeline evaluation, calibration routines, and performance metrics.

## Environment & Dependencies
- Python 3.9+
- `pylsl`
- `numpy`, `scipy`, `pandas`, `scikit-learn`

## Note on Confidentiality
Raw participant data files, proprietary hardware calibration keys, and lab-internal dependencies have been removed. Code is made available here strictly for portfolio and technical presentation purposes.
