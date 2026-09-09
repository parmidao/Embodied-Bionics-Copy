"""
leakage_guard.py
================
Step 3 of the workload pipeline: the leakage guard.

Its ONE job: crash the run loudly if any forbidden column is about to enter the
model (X). Per the brief, a mistake must crash — not quietly produce a
good-looking number.

This guard does NOT hard-code column names. It loads Hamid's schema config
(leakage_guard_config.json, the C123 Row-13 candidate). When Hamid freezes the
final schema, drop in the new JSON and the guard updates itself — no code edit.

Categories enforced (from the config):
  hard_block_exact  — exact names that can never be in X (labels, condition,
                      current n-back level, current-trial outcomes).
  hard_block_globs  — patterns that can never be in X (questionnaire_*,
                      sim_tlx*, raw_nasa_tlx*, completion_time*).
  group_only        — identifiers allowed for LOPO grouping ONLY, never in X.
  conditional       — selected_N / selected_nback_level: allowed as a covariate
                      ONLY if the value is invariant within each participant;
                      if it varies (e.g. 0 in low blocks, n* in high blocks) it
                      leaks the condition and is blocked. Decided from the data.
  tau               — excluded from primary X (sensitivity analysis only).

⚠️ Config status is "candidate". Names must be diffed against the live logger
before Row 13 is frozen. Until then this blocks the candidate names plus obvious
aliases; it can miss a real column named something nobody anticipated. That is
why the guard also FAILS on any column it cannot classify when strict mode is on.
"""

import json
import fnmatch
from pathlib import Path
import pandas as pd

CONFIG_PATH = Path(__file__).parent / "leakage_guard_config.json"


class LeakageError(AssertionError):
    """Raised when a forbidden column is about to enter the model."""
    pass


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def _matches_any_glob(name, globs):
    return any(fnmatch.fnmatch(name, g) for g in globs)


def check_conditional_invariance(df, col, group_col="participant"):
    """
    For a conditional field like selected_N: return True if it is INVARIANT
    within every participant (safe to use as covariate), False if it varies
    within any participant (leaks condition -> must block).
    """
    if col not in df.columns:
        return None  # not present, nothing to decide
    varies = df.groupby(group_col)[col].nunique().max() > 1
    return not varies


def assert_no_leakage(X_columns, df=None, config=None,
                      group_col="participant", strict=True):
    """
    The guard. Call this with the EXACT list of columns about to become X.

    Parameters
    ----------
    X_columns : list[str]   columns you intend to feed the model
    df        : DataFrame    the full table (needed to test conditional fields)
    config    : dict         loaded guard config (defaults to the JSON on disk)
    strict    : bool         if True, any column not recognised as a known-safe
                             feature also fails (catches unanticipated leaks).

    Raises
    ------
    LeakageError with a full report of every violation. Does not return a
    partial pass — one violation fails the whole run.
    """
    if config is None:
        config = load_config()

    hard_exact = set(config.get("hard_block_exact", []))
    hard_globs = config.get("hard_block_globs", [])
    group_only = set(config.get("group_only", []))
    conditional = config.get("conditional", {})
    restricted = config.get("restricted_design_variables", {})
    allowed_cov = set(config.get("allowed_covariates", []))
    primary_globs = config.get("primary_feature_families", [])
    # primary families are given as globs like "physiology_feature_*"; also
    # accept the plain physiological feature names this pipeline actually makes:
    known_feature_names = {
        "hr_mean", "sdnn", "rmssd", "pnn50", "n_beats",
        "scl_mean", "scl_std", "scl_slope", "emg_rms", "emg_mav",
    }

    violations = []

    for col in X_columns:
        # 1. exact hard blocks
        if col in hard_exact:
            violations.append((col, "HARD_BLOCK exact — label/condition/"
                                    "current-N/current-outcome"))
            continue
        # 2. glob hard blocks
        if _matches_any_glob(col, hard_globs):
            violations.append((col, "HARD_BLOCK pattern — questionnaire/"
                                    "tlx/completion-time"))
            continue
        # 3. identifiers must not be in X
        if col in group_only:
            violations.append((col, "GROUP_ONLY — identifier, use for LOPO "
                                    "grouping only, never as a feature"))
            continue
        # 4. tau / restricted design variables
        if col in restricted:
            violations.append((col, "RESTRICTED — design variable (tau); "
                                    "sensitivity analysis only, not primary X"))
            continue
        # 5. conditional fields (selected_N etc.)
        if col in conditional:
            if df is None:
                violations.append((col, "CONDITIONAL — cannot verify "
                                        "invariance without df; supply df"))
                continue
            invariant = check_conditional_invariance(df, col, group_col)
            if invariant is False:
                violations.append((col, "CONDITIONAL FAILED — value VARIES "
                                        "within participant, so it reveals the "
                                        "condition -> blocked"))
                continue
            # invariant True -> allowed, fall through
            continue
        # 6. explicitly allowed covariate
        if col in allowed_cov:
            continue
        # 7. known physiological feature this pipeline makes
        if col in known_feature_names:
            continue
        # 8. primary feature-family globs
        if _matches_any_glob(col, primary_globs):
            continue
        # 9. anything else: unclassified
        if strict:
            violations.append((col, "UNCLASSIFIED — not a known-safe feature. "
                                    "In strict mode this fails: verify against "
                                    "the live logger before allowing it in X."))

    if violations:
        lines = ["", "=" * 68,
                 "LEAKAGE GUARD FAILED — the following columns must not enter X:",
                 "=" * 68]
        for col, reason in violations:
            lines.append(f"  ✗ {col!r}\n      {reason}")
        lines.append("=" * 68)
        lines.append(f"{len(violations)} violation(s). Run halted. "
                     f"Nothing was trained.")
        lines.append("=" * 68)
        raise LeakageError("\n".join(lines))

    return True   # only reached if every column passed


# ----------------------------------------------------------------------------
# Self-test. Run this file directly to SEE the guard catch a planted leak.
# ----------------------------------------------------------------------------
def _self_test():
    print("Leakage guard self-test")
    print("-" * 60)
    cfg = load_config()
    print(f"Loaded config: {cfg['guard_version']}")
    print(f"Status: {cfg['status']}\n")

    # A clean, legitimate feature list (should PASS).
    good_X = ["hr_mean", "sdnn", "rmssd", "pnn50",
              "scl_mean", "scl_std", "scl_slope", "emg_rms", "emg_mav"]
    try:
        assert_no_leakage(good_X, df=None, strict=True)
        print("✅ TEST 1 PASSED: clean feature list accepted.\n")
    except LeakageError as e:
        print("❌ TEST 1 unexpectedly failed:", e)

    # Now deliberately sneak in a forbidden column (should CRASH).
    print("Planting a forbidden column ('workload_condition') in X...")
    bad_X = good_X + ["workload_condition"]
    try:
        assert_no_leakage(bad_X, df=None, strict=True)
        print("❌ TEST 2 FAILED: guard did NOT catch the leak (bad!)")
    except LeakageError as e:
        print("✅ TEST 2 PASSED: guard caught the planted leak and halted:")
        print(e)

    # Test the conditional selected_N logic with fake data.
    print("\nTesting conditional selected_N logic...")
    import numpy as np
    # invariant version (same value per participant) -> should be allowed
    df_ok = pd.DataFrame({
        "participant": ["P01"] * 4 + ["P02"] * 4,
        "selected_N":  [2, 2, 2, 2] + [3, 3, 3, 3],
    })
    try:
        assert_no_leakage(["selected_N"], df=df_ok, strict=False)
        print("✅ TEST 3 PASSED: invariant selected_N allowed as covariate.")
    except LeakageError as e:
        print("❌ TEST 3 failed:", e)

    # varying version (changes within participant) -> should be blocked
    df_leak = pd.DataFrame({
        "participant": ["P01"] * 4 + ["P02"] * 4,
        "selected_N":  [0, 2, 0, 2] + [0, 3, 0, 3],   # 0 in low, n* in high
    })
    try:
        assert_no_leakage(["selected_N"], df=df_leak, strict=False)
        print("❌ TEST 4 FAILED: varying selected_N was NOT blocked (bad!)")
    except LeakageError as e:
        print("✅ TEST 4 PASSED: varying selected_N blocked (it leaks condition).")

    print("\n" + "=" * 60)
    print("Self-test complete. The guard passes clean lists and crashes on leaks.")


if __name__ == "__main__":
    _self_test()
