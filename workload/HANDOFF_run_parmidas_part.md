# HANDOFF CARD — Running Parmida's Part (Physiology + Workload + Sync)

For: Zara (running Parmida's part in her absence)
Machine: white lab laptop (user 14034)
Everything runs from ONE folder in the **(base) Anaconda Prompt** (NOT plain
PowerShell — plain PowerShell grabs the wrong Python and everything fails).

Always start each terminal with:
```
cd C:\Users\14034\Documents\embodied-bionics
```

You'll end up with SEVERAL terminals open at once — one per thing. That's normal.
Each running program = one live stream. Don't close them mid-session.

--------------------------------------------------------------------------
## STEP 1 — Start Parmida's sensors (2 terminals)
--------------------------------------------------------------------------

**Polar H10** (heart) — strap on participant, electrodes moistened, laptop
Bluetooth ON. New (base) terminal:
```
cd C:\Users\14034\Documents\embodied-bionics
python sensor-integration\polar\polar_h10_lsl.py
```
Wait for: `Connected: True` and `Streaming Polar_ECG / Polar_HR / Polar_RR`.
If it hangs: strap must be WORN with damp electrodes; close any Polar phone app.
Leave this terminal open.

**BioRadio** (EDA/sweat) — electrodes on non-dominant hand:
red = index finger, black = middle finger, white = back of hand / wrist bone.
No alcohol on fingers. **Close BioCapture completely first** (check system tray).
New (base) terminal:
```
cd C:\Users\14034\Documents\embodied-bionics
python sensor-integration\bioradio\bioradio_lsl_bridge.py
```
Wait for: `BioRadio_EDA` streaming at 250 Hz. Leave open.
If "device busy": BioCapture is still open somewhere — close it, rerun.

--------------------------------------------------------------------------
## STEP 2 — Start the auto sync-marker emitter (1 terminal)
--------------------------------------------------------------------------

New (base) terminal:
```
cd C:\Users\14034\Documents\embodied-bionics
python workload\sync_test_markers_auto.py
```
It creates the SYNC_TEST stream and waits. You press ENTER ONCE at the start of
each 60-second BBT trial — it then fires START/MIDDLE/END automatically at
0/30/60 s. (You do NOT press three times; just once per trial.)

--------------------------------------------------------------------------
## STEP 3 — Start the n-back task (1 terminal)
--------------------------------------------------------------------------

The n-back plays letters and the participant presses SPACEBAR when the current
letter matches the one N back. It runs CONCURRENTLY during the 60-s BBT trials.

New (base) terminal:
```
cd C:\Users\14034\Downloads\dual_model_0803\dual_model_0727_12pm\work_exact_v13
```
[ n-back launch command goes here — Parmida to fill in the exact command/flags
  before handoff. It should:
  - run the staircase (1-back, 2-back, 3-back — one 60-s BBT trial each) to
    pick the participant's n*, then
  - run main blocks at 0-back (low) or n*-back (high). ]

What to confirm as it runs:
- letters play, spacebar presses register
- after the staircase it prints d' per level and picks n*
- WorkloadMarkers are being emitted (they'll appear in LabRecorder)

--------------------------------------------------------------------------
## STEP 4 — LabRecorder: select EVERYTHING, then record
--------------------------------------------------------------------------

Open LabRecorder:
```
C:\Users\14034\Downloads\LabRecorder-1.17.0-Win_amd64\LabRecorder-1.17.0-Win_amd64\LabRecorder.exe
```
1. Click **Update**.
2. Confirm ALL expected streams appear:
   - Parmida: Polar_HR, Polar_RR, Polar_ECG, BioRadio_EDA
   - EMG (whatever the team's canonical EMG stream is — RawEMG / ClassifierIntent)
   - VR: VREventMarkers, ExperimentData, DecoderControl
   - SYNC_TEST, WorkloadMarkers
3. Click **Select All** — EVERY stream ticked. (Missing this loses streams.)
4. Set a distinct Participant / Run so nothing overwrites.
5. **Start** → run the session → **Stop** (press Stop and let it FINISH writing —
   a dirty stop loses the footer and makes sync unrecoverable).

During the session: press ENTER in the sync terminal once at the start of each
60-s BBT trial.

--------------------------------------------------------------------------
## STEP 5 — Teardown checks (MANDATORY — Dr. Park's rule)
--------------------------------------------------------------------------

"A stream list is not evidence." After stopping, in a (base) terminal, run all
three on the file that was just saved (get the exact path from LabRecorder's
"Saving to..." box):

```
cd C:\Users\14034\Documents\embodied-bionics

python check_recording.py "<the>.xdf"       # every stream present + NON-ZERO samples
python check_clean_stop.py "<the>.xdf"       # footer + clock offsets present (clean stop)
python sync_test.py "<the>.xdf"              # clock alignment / drift
```

PASS criteria:
- check_recording: every stream has non-zero samples. (RawEMG showing 6 channels
  vs "expected 4" is a harmless checker note, not a failure.)
- check_clean_stop: ✅ PASS (footer + offsets). If ❌ FAIL, the recording was not
  stopped cleanly — RE-RECORD.
- sync_test: residual within tolerance.

If any physiology stream has ZERO samples (the Polar_ECG-died problem), it is a
DEAD STREAM — re-seat electrodes / restart that bridge / re-record. Do NOT accept
a file just because the stream is listed.

--------------------------------------------------------------------------
## Quick reference — all commands
--------------------------------------------------------------------------
```
cd C:\Users\14034\Documents\embodied-bionics
python sensor-integration\polar\polar_h10_lsl.py
python sensor-integration\bioradio\bioradio_lsl_bridge.py
python workload\sync_test_markers_auto.py
# n-back: cd to work_exact_v13, run the n-back command
# LabRecorder: Update -> Select All -> Start -> (ENTER per trial) -> Stop
python check_recording.py "<file>.xdf"
python check_clean_stop.py "<file>.xdf"
python sync_test.py "<file>.xdf"
```

--------------------------------------------------------------------------
## If something breaks
--------------------------------------------------------------------------
- "No module named X" → wrong Python; use the (base) Anaconda Prompt, not
  PowerShell. If still failing: `pip install X`.
- A sensor won't connect → check it's worn/powered, its vendor app is CLOSED
  (BioCapture / Polar phone app), Bluetooth on.
- A stream missing in LabRecorder → click Update again; make sure its terminal
  is still running.
- Text Parmida if a check FAILS and it's not obvious — better to pause than
  record a dead session.
```
