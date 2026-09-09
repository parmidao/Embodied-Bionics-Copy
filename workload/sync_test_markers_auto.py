"""
sync_test_markers_auto.py — auto-timed SYNC_TEST emitter for 60-second BBT trials.
=================================================================================
Instead of pressing ENTER three times per trial, you press ENTER ONCE at the
start of each 60-second BBT trial. The emitter then fires three markers by
itself: START at 0 s, MIDDLE at 30 s, END at 60 s. Timing is exact (driven by
the clock, not your reaction), and the only manual presses left in the session
are the participant's n-back responses.

WHY THIS DESIGN
  Manual start/middle/end pressing means three chances per trial to miss a
  marker under time pressure, and the marker times depend on human reaction.
  One press to arm a trial, with the three markers placed automatically 30 s
  apart, removes both problems while keeping a human in control of WHEN each
  trial begins (which the emitter can't know on its own).

MARKER FORMAT
  Emits: SYNC_TEST_START, SYNC_TEST_MIDDLE, SYNC_TEST_END, each tagged with the
  trial number, e.g. "SYNC_TEST_START trial=1". If the team's parser wants a
  different format (e.g. pipe-delimited "SYNC_TEST|01|start"), change MARKER_FMT
  below — that's the only edit needed.

USAGE
  1. Start this BEFORE LabRecorder starts recording:
       python workload\sync_test_markers_auto.py
  2. In LabRecorder: Update, tick SYNC_TEST, Start recording.
  3. For each 60-s BBT trial: press ENTER exactly when the trial starts.
     The emitter fires START now, MIDDLE at +30 s, END at +60 s, printing each.
  4. Repeat for each trial. Type q then ENTER to finish.
  5. Stop LabRecorder cleanly (so the footer/offsets are written).

TRIAL_SECONDS and the marker times are configurable below.
"""

import sys
import time
import threading

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError:
    print("ERROR: pylsl not installed.")
    print("  pip install pylsl")
    sys.exit(1)

STREAM_NAME    = "SYNC_TEST"
SOURCE_ID      = "sync_test_marker_emitter_v2_auto"
TRIAL_SECONDS  = 60.0
MARKER_TIMES   = {"START": 0.0, "MIDDLE": 30.0, "END": 60.0}

# Marker string format. {phase} = START/MIDDLE/END, {trial} = trial number.
# To match a pipe-delimited team format, e.g.: "SYNC_TEST|{trial:02d}|{phase_lc}"
MARKER_FMT = "SYNC_TEST_{phase} trial={trial}"


def make_outlet():
    info = StreamInfo(STREAM_NAME, "Markers", 1, 0, "string", SOURCE_ID)
    return StreamOutlet(info)


def fire_trial_markers(outlet, trial_num):
    """
    Fire START now, then schedule MIDDLE and END on background timers so the
    operator can immediately arm the next trial if needed. Each marker prints
    when it fires.
    """
    def emit(phase):
        label = MARKER_FMT.format(phase=phase, phase_lc=phase.lower(),
                                  trial=trial_num)
        t = local_clock()
        outlet.push_sample([label])
        print(f"    [trial {trial_num}] {phase:6s} -> {label!r}  @ LSL {t:.3f}")

    # START immediately
    emit("START")
    # MIDDLE and END on timers
    t_mid = threading.Timer(MARKER_TIMES["MIDDLE"], emit, args=("MIDDLE",))
    t_end = threading.Timer(MARKER_TIMES["END"], emit, args=("END",))
    t_mid.daemon = True
    t_end.daemon = True
    t_mid.start()
    t_end.start()
    return t_mid, t_end


def main():
    print("SYNC_TEST auto-timed marker emitter (one press per 60-s trial)")
    print("-" * 66)
    print("Creating LSL outlet 'SYNC_TEST' ...")
    outlet = make_outlet()
    time.sleep(0.5)
    print("Outlet is live.\n")
    print("In LabRecorder: Update, tick SYNC_TEST, Start recording.\n")
    print("Then, for EACH 60-second BBT trial: press ENTER when the trial")
    print("starts. START fires now, MIDDLE at +30 s, END at +60 s, all")
    print("automatically. Type q + ENTER to finish.\n")

    trial = 0
    pending = []
    while True:
        cmd = input(f"Press ENTER to arm trial {trial + 1} "
                    f"(or q to quit): ").strip().lower()
        if cmd == "q":
            break
        trial += 1
        print(f"  -> trial {trial} armed; START now, MIDDLE +30 s, END +60 s")
        pending.append(fire_trial_markers(outlet, trial))

    # let any still-pending MIDDLE/END timers fire before exit
    print("\nWaiting for any pending markers to flush...")
    time.sleep(TRIAL_SECONDS + 1)   # ensure the last trial's END has fired
    print(f"Done. Fired markers for {trial} trial(s).")
    print("Now STOP LabRecorder cleanly, then run check_clean_stop.py on the file.")


if __name__ == "__main__":
    main()
