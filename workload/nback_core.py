"""
nback_core.py
=============
The scientific core of the n-back task: sequence generation and scoring.
NO audio, NO hardware, NO LSL here — just the logic, so it can be tested
instantly and is easy to reason about. Audio, clicks, and markers live in
separate files that call into this one.

Design decisions locked with Parmida (10 Aug 2026):
  - go/no-go response: participant clicks ONLY on target (n-back match).
  - response window: click counts if within RESPONSE_WINDOW_S of stimulus onset.
  - scoring: d-prime (signal detection theory) as primary; balanced accuracy
    reported alongside. Plain % correct is deliberately NOT the primary metric,
    because with ~30% targets a participant who never clicks scores ~70%
    "correct" while discriminating nothing. d-prime separates true sensitivity
    from response bias, which is what an "at capacity" threshold requires.
  - defaults follow the common n-back conventions in the literature:
    2000 ms stimulus onset asynchrony, ~30% targets.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---- conventions from the n-back literature (all configurable) -------------
STIM_INTERVAL_S    = 2.0     # stimulus onset asynchrony (new item every 2 s)
TARGET_RATE        = 0.30    # ~30% of items are targets (matches)
RESPONSE_WINDOW_S  = 2.0     # a click within this many s of onset counts
LEVEL_DURATION_S   = 60.0    # per-level duration (Parmida: 60 s; set 90 for
                             # a separate, more reliable staircase segment)
LETTER_SET         = ["C", "H", "K", "L", "Q", "R", "S", "T"]  # low-confusability


@dataclass
class Stimulus:
    index: int          # position in the sequence (0-based)
    onset_s: float      # time from level start, seconds
    letter: str
    is_target: bool     # True if this letter matches the one n back


@dataclass
class LevelSequence:
    n: int                          # the n-back level (0,1,2,3)
    stimuli: list                   # list[Stimulus]
    duration_s: float

    @property
    def n_items(self):
        return len(self.stimuli)

    @property
    def n_targets(self):
        return sum(s.is_target for s in self.stimuli)


def generate_sequence(n, duration_s=LEVEL_DURATION_S,
                      interval_s=STIM_INTERVAL_S, target_rate=TARGET_RATE,
                      letters=LETTER_SET, rng=None):
    """
    Build one n-back level's stimulus sequence.

    We first decide WHICH positions are targets (forced matches), then fill the
    sequence so that exactly those positions match n-back and non-targets do
    NOT accidentally match. This gives control over the target rate, which
    random letters alone would not.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_items = int(duration_s // interval_s)
    seq = [None] * n_items

    # positions eligible to be targets: only those with an item n steps back
    eligible = list(range(n, n_items)) if n > 0 else []
    n_targets = int(round(target_rate * n_items)) if n > 0 else 0
    target_positions = set(rng.choice(eligible, size=min(n_targets, len(eligible)),
                                      replace=False)) if eligible else set()

    for i in range(n_items):
        if n > 0 and i in target_positions:
            # forced match: copy the letter from n positions back
            seq[i] = seq[i - n]
        else:
            # pick a letter that does NOT create an unintended n-back match
            forbidden = set()
            if n > 0 and i >= n and seq[i - n] is not None:
                forbidden.add(seq[i - n])
            choices = [l for l in letters if l not in forbidden]
            seq[i] = rng.choice(choices)

    stimuli = []
    for i, letter in enumerate(seq):
        is_target = (n > 0 and i >= n and letter == seq[i - n])
        stimuli.append(Stimulus(index=i, onset_s=i * interval_s,
                                 letter=letter, is_target=is_target))
    return LevelSequence(n=n, stimuli=stimuli, duration_s=duration_s)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
@dataclass
class ScoreResult:
    n: int
    hits: int
    misses: int
    false_alarms: int
    correct_rejections: int
    n_targets: int
    n_nontargets: int
    hit_rate: float
    fa_rate: float
    dprime: float
    balanced_accuracy: float


def _zscore_rate(rate, n_trials):
    """
    Convert a proportion to a z-score for d-prime, with the standard
    log-linear correction so rates of 0 or 1 don't blow up to +/- infinity.
    """
    from scipy.stats import norm
    # log-linear correction: add 0.5 to counts, 1 to n
    corrected = (rate * n_trials + 0.5) / (n_trials + 1)
    return norm.ppf(corrected)


def score_level(sequence: LevelSequence, click_times_s,
                response_window_s=RESPONSE_WINDOW_S):
    """
    Score one level given the stimulus sequence and a list of click times
    (seconds from level start).

    For each stimulus we open a window [onset, onset + response_window]. A click
    in a target's window is a HIT; a click in a non-target's window is a FALSE
    ALARM. Targets with no click are MISSES; non-targets with no click are
    CORRECT REJECTIONS. Each click is matched to at most one stimulus window.
    """
    clicks = sorted(click_times_s)
    used = [False] * len(clicks)

    hits = misses = false_alarms = correct_rejections = 0

    for stim in sequence.stimuli:
        w0, w1 = stim.onset_s, stim.onset_s + response_window_s
        # find the first unused click inside this window
        clicked = False
        for ci, ct in enumerate(clicks):
            if used[ci]:
                continue
            if w0 <= ct < w1:
                clicked = True
                used[ci] = True
                break
        if stim.is_target:
            if clicked:
                hits += 1
            else:
                misses += 1
        else:
            if clicked:
                false_alarms += 1
            else:
                correct_rejections += 1

    n_targets = hits + misses
    n_nontargets = false_alarms + correct_rejections
    hit_rate = hits / n_targets if n_targets else 0.0
    fa_rate = false_alarms / n_nontargets if n_nontargets else 0.0

    # d-prime with log-linear correction
    if n_targets > 0 and n_nontargets > 0:
        zh = _zscore_rate(hit_rate, n_targets)
        zf = _zscore_rate(fa_rate, n_nontargets)
        dprime = zh - zf
    else:
        dprime = float("nan")

    # balanced accuracy = mean of (hit rate, correct-rejection rate)
    cr_rate = correct_rejections / n_nontargets if n_nontargets else 0.0
    balanced_accuracy = 0.5 * (hit_rate + cr_rate)

    return ScoreResult(
        n=sequence.n, hits=hits, misses=misses, false_alarms=false_alarms,
        correct_rejections=correct_rejections, n_targets=n_targets,
        n_nontargets=n_nontargets, hit_rate=hit_rate, fa_rate=fa_rate,
        dprime=dprime, balanced_accuracy=balanced_accuracy,
    )


# ----------------------------------------------------------------------------
# Self-test / demo
# ----------------------------------------------------------------------------
def _demo():
    rng = np.random.default_rng(0)
    print("n-back core self-test")
    print("=" * 60)

    for n in [0, 1, 2, 3]:
        seq = generate_sequence(n, rng=rng)
        print(f"\n{n}-back: {seq.n_items} items, {seq.n_targets} targets "
              f"({100*seq.n_targets/seq.n_items:.0f}%), {seq.duration_s:.0f}s")
        letters = " ".join(s.letter + ("*" if s.is_target else " ")
                           for s in seq.stimuli[:15])
        print(f"  first 15: {letters}   (* = target)")

    # Simulate a 'good' participant on 2-back: clicks every target, no false alarms
    print("\n" + "=" * 60)
    print("Simulated PERFECT participant on 2-back:")
    seq2 = generate_sequence(2, rng=rng)
    perfect_clicks = [s.onset_s + 0.5 for s in seq2.stimuli if s.is_target]
    r = score_level(seq2, perfect_clicks)
    print(f"  hits={r.hits} misses={r.misses} FA={r.false_alarms} "
          f"CR={r.correct_rejections}")
    print(f"  hit_rate={r.hit_rate:.2f} fa_rate={r.fa_rate:.2f} "
          f"d'={r.dprime:.2f} balanced_acc={r.balanced_accuracy:.2f}")

    # Simulate a 'never clicks' participant: shows why % correct is misleading
    print("\nSimulated participant who NEVER clicks on 2-back:")
    r2 = score_level(seq2, [])
    naive_pct = (r2.hits + r2.correct_rejections) / seq2.n_items
    print(f"  naive % correct = {100*naive_pct:.0f}%  "
          f"(looks OK!) but d'={r2.dprime:.2f}, balanced_acc="
          f"{r2.balanced_accuracy:.2f}  (correctly shows no skill)")

    print("\n" + "=" * 60)
    print("This is why d-prime / balanced accuracy is the metric, not % correct.")


if __name__ == "__main__":
    _demo()
