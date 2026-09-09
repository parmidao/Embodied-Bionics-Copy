"""
nback_task.py
=============
The full, runnable n-back task. Ties together:
  - nback_core.py      (sequence generation + d-prime scoring)
  - pyttsx3            (speaks the letters aloud, pre-generated once)
  - keyboard spacebar  (stands in for the VR clicker — swappable, see CLICK INPUT)
  - pylsl              (emits WorkloadMarkers: every stimulus + every response)

WHAT IT DOES
  Runs a staircase: 1-back, 2-back, 3-back in order, each for LEVEL_DURATION_S.
  For each level it speaks letters every 2 s, listens for your spacebar presses,
  scores you with d-prime, and picks n* = highest level with balanced accuracy
  >= PASS_THRESHOLD. Every stimulus and response is timestamped into an LSL
  stream called WorkloadMarkers so it lands in the LabRecorder .xdf.

HOW TO RUN (test mode, spacebar, no VR, no LabRecorder needed):
    .\.venv\Scripts\python.exe workload\nback_task.py

  Put on headphones, and press SPACEBAR whenever the current letter matches the
  one n positions back. At the end it prints your per-level scores and n*.

CLICK INPUT — the one stubbed piece
  Responses currently come from the spacebar (get_clicks_keyboard). When Zara
  says how the VR controller press reaches an external program, implement
  get_clicks_lsl() and switch CLICK_SOURCE. Nothing else changes.
"""

import sys
import time
from pathlib import Path
import numpy as np

# our own core
sys.path.insert(0, str(Path(__file__).parent))
from nback_core import (generate_sequence, score_level,
                        LEVEL_DURATION_S, STIM_INTERVAL_S, RESPONSE_WINDOW_S)

# ---- config ----------------------------------------------------------------
STAIRCASE_LEVELS = [1, 2, 3]
PASS_THRESHOLD   = 0.80        # n* = highest level with balanced_acc >= this
AUDIO_DIR        = Path(__file__).parent / "audio_letters"
CLICK_SOURCE     = "keyboard"  # "keyboard" (test) or "lsl" (real clicker, TODO)
EMIT_MARKERS     = True        # push WorkloadMarkers over LSL

# ----------------------------------------------------------------------------
# AUDIO: pre-generate one clip per letter, then play clips on schedule.
# ----------------------------------------------------------------------------
def ensure_audio(letters):
    """Generate a .wav for each letter once, using pyttsx3 (offline TTS)."""
    try:
        import pyttsx3
    except ImportError:
        print("ERROR: pyttsx3 not installed. Run:")
        print("    .\\.venv\\Scripts\\python.exe -m pip install pyttsx3")
        sys.exit(1)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    made = []
    for L in letters:
        path = AUDIO_DIR / f"{L}.wav"
        if not path.exists():
            engine.save_to_file(L, str(path))
            made.append(L)
    engine.runAndWait()
    if made:
        print(f"  generated audio for: {' '.join(made)}")
    return {L: AUDIO_DIR / f"{L}.wav" for L in letters}


def make_player():
    """Return a function play(path) that plays a wav without blocking long."""
    # Try winsound first (built into Windows, zero install).
    try:
        import winsound
        def play(path):
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return play
    except Exception:
        pass
    # Fallback: playsound if present
    try:
        from playsound import playsound
        def play(path):
            playsound(str(path), block=False)
        return play
    except Exception:
        def play(path):
            pass  # silent fallback; task still runs, just no sound
        print("  (no audio backend found; running SILENT — letters print only)")
        return play


# ----------------------------------------------------------------------------
# CLICK INPUT — stubbed. keyboard now; LSL later for the real clicker.
# ----------------------------------------------------------------------------
class KeyboardClicker:
    """Collect spacebar presses with timestamps, non-blocking, on Windows."""
    def __init__(self):
        import msvcrt
        self.msvcrt = msvcrt
        self.clicks = []
    def poll(self, t_now):
        # drain any pending keypresses
        while self.msvcrt.kbhit():
            ch = self.msvcrt.getch()
            if ch == b' ':
                self.clicks.append(t_now)
    def reset(self):
        self.clicks = []


def get_clicker():
    if CLICK_SOURCE == "keyboard":
        return KeyboardClicker()
    elif CLICK_SOURCE == "lsl":
        raise NotImplementedError(
            "LSL clicker not implemented yet — waiting on Zara for the stream "
            "name/format. Implement an inlet reader here once known.")
    else:
        raise ValueError(CLICK_SOURCE)


# ----------------------------------------------------------------------------
# WORKLOAD MARKERS (LSL) — stamps stimuli and responses into the recording.
# ----------------------------------------------------------------------------
def make_marker_outlet():
    if not EMIT_MARKERS:
        return None
    try:
        from pylsl import StreamInfo, StreamOutlet
    except ImportError:
        print("  (pylsl not available; WorkloadMarkers disabled)")
        return None
    info = StreamInfo("WorkloadMarkers", "Markers", 1, 0, "string",
                      "workload_nback_v1")
    return StreamOutlet(info)


def push_marker(outlet, text):
    if outlet is not None:
        outlet.push_sample([text])


# ----------------------------------------------------------------------------
# RUN ONE LEVEL
# ----------------------------------------------------------------------------
def run_level(n, audio, play, clicker, marker_outlet, rng):
    seq = generate_sequence(n, rng=rng)
    print(f"\n=== {n}-back  ({seq.n_items} letters, "
          f"{seq.n_targets} targets) — press SPACEBAR on matches ===")
    push_marker(marker_outlet, f"LEVEL_START n={n}")

    clicker.reset()
    t0 = time.perf_counter()

    for stim in seq.stimuli:
        # wait until this stimulus's onset
        while time.perf_counter() - t0 < stim.onset_s:
            clicker.poll(time.perf_counter() - t0)
            time.sleep(0.005)
        # present
        play(audio[stim.letter])
        push_marker(marker_outlet,
                    f"STIM n={n} i={stim.index} letter={stim.letter} "
                    f"target={int(stim.is_target)}")
        print(f"  {stim.letter}{'  <target>' if stim.is_target else ''}")

    # keep polling through the final response window
    end_t = seq.stimuli[-1].onset_s + RESPONSE_WINDOW_S
    while time.perf_counter() - t0 < end_t:
        clicker.poll(time.perf_counter() - t0)
        time.sleep(0.005)

    # record responses as markers
    for ct in clicker.clicks:
        push_marker(marker_outlet, f"RESPONSE n={n} t={ct:.3f}")

    result = score_level(seq, clicker.clicks)
    push_marker(marker_outlet, f"LEVEL_END n={n} "
                f"dprime={result.dprime:.3f} bacc={result.balanced_accuracy:.3f}")
    print(f"  -> hits={result.hits} misses={result.misses} "
          f"FA={result.false_alarms} | d'={result.dprime:.2f} "
          f"balanced_acc={result.balanced_accuracy:.2f}")
    return result


# ----------------------------------------------------------------------------
# STAIRCASE: run levels, pick n*
# ----------------------------------------------------------------------------
def run_staircase():
    from nback_core import LETTER_SET
    rng = np.random.default_rng()

    print("Preparing audio...")
    audio = ensure_audio(LETTER_SET)
    play = make_player()
    clicker = get_clicker()
    marker_outlet = make_marker_outlet()
    if marker_outlet:
        print("WorkloadMarkers LSL stream is live.")
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("STAIRCASE — you'll hear letters every 2 s. Press SPACEBAR when the")
    print(f"current letter matches the one N back. Threshold to pass: "
          f"balanced accuracy >= {PASS_THRESHOLD}.")
    print("=" * 60)
    input("Press ENTER to begin...")

    results = {}
    for n in STAIRCASE_LEVELS:
        results[n] = run_level(n, audio, play, clicker, marker_outlet, rng)
        time.sleep(1.0)

    # n* = highest level passed
    passed = [n for n in STAIRCASE_LEVELS
              if results[n].balanced_accuracy >= PASS_THRESHOLD]
    nstar = max(passed) if passed else 1
    flag = "" if passed else "  (FLAG: failed 1-back, n* set to 1)"

    push_marker(marker_outlet, f"NSTAR selected={nstar}")
    print("\n" + "=" * 60)
    print("STAIRCASE COMPLETE")
    for n in STAIRCASE_LEVELS:
        r = results[n]
        mark = " <= n*" if n == nstar else ""
        print(f"  {n}-back: d'={r.dprime:.2f}  "
              f"balanced_acc={r.balanced_accuracy:.2f}{mark}")
    print(f"\n  n* = {nstar}{flag}")
    print("=" * 60)
    if marker_outlet:
        time.sleep(1.0)  # flush
    return nstar, results


if __name__ == "__main__":
    run_staircase()
