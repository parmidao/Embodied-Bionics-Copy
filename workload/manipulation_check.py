"""
manipulation_check.py
=====================
P-02: the workload manipulation check.

THE QUESTION IT ANSWERS
  "Did the workload manipulation actually work?" i.e. were participants really
  under higher cognitive load in the high condition than the low condition?

WHY IT COMES BEFORE MODEL ACCURACY
  The whole study labels windows "low" (0-back) and "high" (n*-back). If those
  conditions did NOT actually differ in difficulty for participants, the labels
  are meaningless and no classification accuracy — however high — would mean
  anything. So this check is the PRECONDITION for interpreting any model result.
  It is also what tells you, if the model fails, whether the problem is
  "physiology doesn't carry workload" vs "the conditions weren't distinct".

WHAT IT USES
  The n-back performance itself — specifically d-prime (and balanced accuracy),
  computed by nback_core.score_level. Under higher working-memory load,
  performance drops: d-prime falls and/or reaction accuracy falls. So a
  significant drop from low (0-back) to high (n*-back) is the manipulation
  working.

  IMPORTANT — this uses n-back PERFORMANCE, which is manipulation-check data
  ONLY. Per the leakage rules, n-back RT/accuracy/d-prime must NEVER be model
  features. This script lives entirely on the validation side.

WHAT IT REPORTS
  1. Per-participant low-vs-high performance.
  2. A group-level paired test (each participant is their own control).
  3. Effect size (how big the difference is, not just whether it's significant).
  4. A per-participant flag for anyone whose conditions did NOT differ, since
     those participants' labels are suspect.

  With a small pilot (n = 15-20) the group test may be underpowered; the
  per-participant view and the effect size matter more than a single p-value.
  This is stated in the output so the result isn't over-read.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ParticipantResult:
    pid: str
    low_score: float      # performance in the low (0-back) condition
    high_score: float     # performance in the high (n*-back) condition
    delta: float          # low - high (positive = performance dropped under load)
    manipulation_ok: bool # did this participant show the expected drop?


def check_manipulation(per_participant, metric_name="dprime",
                       min_drop=0.0):
    """
    per_participant : list of (pid, low_score, high_score)
        one row per participant, giving their mean performance in the low
        (0-back) and high (n*-back) conditions on the chosen metric.
    metric_name : just a label for the printout.
    min_drop : the smallest low-minus-high difference counted as "the
        manipulation worked" for that participant. 0.0 means any drop counts;
        set higher to require a meaningful drop.

    Returns a dict with per-participant results, the group-level paired test,
    and an effect size.
    """
    results = []
    lows, highs = [], []
    for pid, low, high in per_participant:
        delta = low - high                 # positive = performance fell under load
        ok = delta > min_drop
        results.append(ParticipantResult(pid, low, high, delta, ok))
        lows.append(low)
        highs.append(high)

    lows = np.asarray(lows, float)
    highs = np.asarray(highs, float)
    deltas = lows - highs

    n = len(deltas)
    mean_delta = float(np.mean(deltas)) if n else float("nan")
    sd_delta = float(np.std(deltas, ddof=1)) if n > 1 else float("nan")

    # paired t-test (each participant is their own control)
    t_stat = p_value = float("nan")
    if n > 1 and sd_delta > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(lows, highs)

    # effect size: Cohen's dz for a paired design
    cohens_dz = mean_delta / sd_delta if (n > 1 and sd_delta > 0) else float("nan")

    n_ok = sum(r.manipulation_ok for r in results)

    return {
        "metric": metric_name,
        "results": results,
        "n": n,
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_dz": float(cohens_dz),
        "n_participants_ok": n_ok,
    }


def print_report(res):
    print("=" * 68)
    print(f"  WORKLOAD MANIPULATION CHECK  (metric: {res['metric']})")
    print("=" * 68)
    print(f"  Question: was performance WORSE under high load than low load?")
    print(f"  (a drop from low -> high means the manipulation worked)\n")

    print(f"  {'participant':12s} {'low':>8s} {'high':>8s} {'drop':>8s}   manipulation")
    print("  " + "-" * 60)
    for r in res["results"]:
        mark = "OK" if r.manipulation_ok else "!! no drop"
        print(f"  {r.pid:12s} {r.low_score:8.2f} {r.high_score:8.2f} "
              f"{r.delta:8.2f}   {mark}")

    print("\n  " + "-" * 60)
    print(f"  participants showing the expected drop: "
          f"{res['n_participants_ok']} / {res['n']}")
    print(f"  mean drop (low - high): {res['mean_delta']:.3f}  "
          f"(SD {res['sd_delta']:.3f})")

    if not np.isnan(res["p_value"]):
        print(f"  paired t-test: t = {res['t_stat']:.2f}, p = {res['p_value']:.4f}")
        print(f"  effect size (Cohen's dz): {res['cohens_dz']:.2f}", end="")
        dz = abs(res["cohens_dz"])
        size = ("negligible" if dz < 0.2 else "small" if dz < 0.5
                else "medium" if dz < 0.8 else "large")
        print(f"  ({size})")

    print("\n  Interpretation:")
    if res["n"] < 10:
        print("  ⚠️  Small n — the group p-value is underpowered. Weight the")
        print("     per-participant column and the effect size more than the")
        print("     p-value. A consistent drop across most participants is the")
        print("     real signal even if p is not significant.")
    if res["mean_delta"] > 0 and res["n_participants_ok"] >= 0.7 * res["n"]:
        print("  ✅ Manipulation appears to have worked: most participants")
        print("     performed worse under high load. Condition labels are")
        print("     meaningful and safe to use for classification.")
    else:
        print("  ❌ Manipulation is NOT clearly working: performance did not")
        print("     consistently drop under high load. Condition labels are")
        print("     suspect — investigate before trusting ANY model accuracy.")
        print("     (Check: were participants actually doing the n-back? Were")
        print("     responses registering? Was n* set correctly?)")
    print("=" * 68)


# ----------------------------------------------------------------------------
# Demo with synthetic per-participant scores
# ----------------------------------------------------------------------------
def _demo():
    print("Manipulation check — demo on synthetic per-participant d-prime\n")

    # Case 1: manipulation worked — most people drop under high load
    rng = np.random.default_rng(0)
    worked = []
    for i in range(1, 13):
        low = rng.normal(2.8, 0.4)          # good performance at 0-back
        high = low - rng.normal(1.2, 0.5)   # worse under n*-back load
        worked.append((f"P{i:02d}", round(low, 2), round(high, 2)))
    res1 = check_manipulation(worked)
    print_report(res1)

    print("\n")

    # Case 2: manipulation FAILED — no consistent difference
    failed = []
    for i in range(1, 13):
        low = rng.normal(2.5, 0.4)
        high = rng.normal(2.5, 0.4)         # same as low — no load effect
        failed.append((f"P{i:02d}", round(low, 2), round(high, 2)))
    res2 = check_manipulation(failed)
    print_report(res2)


if __name__ == "__main__":
    _demo()
