# Delsys EMG -> LSL

`delsys_lsl.py` taps the Delsys Python demo's data loop and streams the EMG
channels to an LSL outlet named `DelsysEMG`.

## Setup (on a fresh clone of the Delsys Example-Applications demo)
1. Copy `delsys_lsl.py` into the demo root (next to `DelsysPythonDemo.py`).
2. Make three edits to `AeroPy/DataManager.py`:
   - Top of file:  `from delsys_lsl import EMGStreamer`
   - In `DataKernel.__init__`:  `self.lsl = EMGStreamer()`
   - In `processData`, first line inside `if outArr is not None:`:
     `self.lsl.push(self.trigno_base, outArr)`
3. Run the demo, uncheck "Stream Time Series Values?", Scan, Start.
   The `[LSL] DelsysEMG outlet created` block confirms it streams.