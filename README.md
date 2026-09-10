# Sensor Integration & Cognitive Workload Modeling

A modular pipeline for multimodal biosignal streaming, real-time synchronization, and cognitive workload estimation for adaptive human-machine interfaces.

## Overview
This repository contains modular implementations for:
- **Sensor Integration**: Biosignal acquisition and streaming pipelines utilizing Lab Streaming Layer (LSL).
- **Workload Estimation**: Feature extraction and classification architectures for real-time cognitive workload decoding.

## Architecture
- `sensor-integration/`: Device drivers, streaming protocols, and synchronization logic.
- `workload/`: Processing scripts, feature engineering, and model evaluation routines.

## Requirements
- Python 3.9+
- `pylsl`
- `numpy`, `scipy`, `pandas`, `scikit-learn`

## Usage
1. Set up the virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
2. Run the integration or workload pipelines directly within their respective modules.

## License & Academic Notice
Code published for academic portfolio presentation. All proprietary research data, raw participant logs, and third-party vendor SDK binaries have been excluded.
