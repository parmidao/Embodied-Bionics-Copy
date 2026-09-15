"""
train_model.py
==============
Step 4: train and evaluate the workload classifier.

WHAT IT DOES
  - loads the feature table (features.parquet)
  - runs the LEAKAGE GUARD on the exact feature columns before any training;
    a forbidden column crashes the run (nothing gets trained on leaked data)
  - trains XGBoost with LEAVE-ONE-PARTICIPANT-OUT cross-validation
  - runs the WHOLE thing TWICE: once on nominal labels, once on calibrated
    labels, identical features / folds / model — only the label column differs.
    This is the H1 contrast.
  - macro-averages over the classes PRESENT for each held-out participant
    (Dr. Park's rule: n*=1 has no 'below', n*=3 has no 'above', so class
    structure isn't identical across folds)
  - prints a per-participant class-presence table
  - times one full LOPO pass so the permutation-null cost can be projected

⚠️ On synthetic data every number here is meaningless — it only shows the
pipeline runs, returns chance at signal=0, and recovers calibrated labels when
signal is present. No number from synthetic data is a result.

When real .xdf arrives: only the loader (build_features.py) changes. This file
does not.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from leakage_guard import assert_no_leakage, load_config, LeakageError

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

FEATURES_PATH = Path(__file__).parent / "features.parquet"

# The physiological feature columns this pipeline produces. These are the ONLY
# columns allowed into X. Everything else in the table is metadata or labels.
FEATURE_COLUMNS = ["hr_mean", "sdnn", "rmssd", "pnn50",
                   "scl_mean", "scl_std", "scl_slope", "emg_rms", "emg_mav"]

GROUP_COL = "participant"


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    # We evaluate H1 on the staircase windows, where BOTH label schemes exist.
    df = df[df["segment"] == "staircase"].copy()
    return df


def make_model():
    # shallow, strongly regularised — right posture for small-n physiological data
    return XGBClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=0, n_jobs=2,
    )


def encode_labels(series):
    """Map label strings/ints to 0..K-1 and return (y, classes)."""
    classes = sorted(series.dropna().unique(), key=lambda x: str(x))
    mapping = {c: i for i, c in enumerate(classes)}
    y = series.map(mapping).values
    return y, classes, mapping


def run_lopo(df, label_col):
    """
    Leave-one-participant-out for one label scheme. Returns per-fold results,
    macro-averaged over classes present in each fold's TEST participant.
    """
    # rows with a valid label under this scheme
    d = df[df[label_col].notna()].copy()
    X = d[FEATURE_COLUMNS]
    y, classes, mapping = encode_labels(d[label_col])
    groups = d[GROUP_COL].values

    # ---- LEAKAGE GUARD: hard-fail if any forbidden column is in X ----
    assert_no_leakage(list(X.columns), df=d, strict=True)

    logo = LeaveOneGroupOut()
    fold_bacc = []
    class_presence = []   # (participant, sorted list of classes present in test)

    y_true_all, y_pred_all = [], []

    for tr, te in logo.split(X, y, groups):
        model = make_model()
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])

        # classes present in THIS held-out participant
        present = sorted(np.unique(y[te]))
        present_names = [classes[i] for i in present]
        pid = groups[te][0]
        class_presence.append((pid, present_names))

        # balanced accuracy over the classes present in the test fold
        bacc = balanced_accuracy_score(y[te], pred)
        fold_bacc.append((pid, bacc, len(present)))

        y_true_all.extend(y[te]); y_pred_all.extend(pred)

    macro = np.mean([b for _, b, _ in fold_bacc])
    return {
        "classes": classes,
        "fold_bacc": fold_bacc,
        "class_presence": class_presence,
        "macro_balanced_accuracy": macro,
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
    }


def chance_level(n_classes):
    return 1.0 / n_classes


def print_scheme_report(name, res):
    print(f"\n{'='*64}\n  {name.upper()} LABELS\n{'='*64}")
    print(f"  classes: {res['classes']}")
    print(f"  chance (balanced acc): {chance_level(len(res['classes'])):.3f}")
    print(f"\n  per-participant balanced accuracy (over classes present):")
    for pid, bacc, ncls in res["fold_bacc"]:
        print(f"    {pid:6s}  bacc={bacc:.3f}  ({ncls} classes present)")
    print(f"\n  MACRO balanced accuracy (mean over folds): "
          f"{res['macro_balanced_accuracy']:.3f}")


def print_class_presence_table(nominal_res, calibrated_res):
    print(f"\n{'='*64}\n  PER-PARTICIPANT CLASS-PRESENCE TABLE\n{'='*64}")
    print("  (Dr. Park's check: calibrated class structure is NOT identical")
    print("   across participants — n*=1 lacks 'below', n*=3 lacks 'above')\n")
    pres = dict(calibrated_res["class_presence"])
    print(f"    {'participant':12s}  calibrated classes present")
    for pid in sorted(pres):
        print(f"    {pid:12s}  {pres[pid]}")


def main():
    print("XGBoost training + LOPO evaluation")
    print("⚠️  If this is synthetic data, NO number below is a result.\n")

    cfg = load_config()
    print(f"Leakage guard config: {cfg['guard_version']} (status: {cfg['status']})")

    df = load_features()
    print(f"Staircase windows: {len(df)} from "
          f"{df[GROUP_COL].nunique()} participants\n")

    # time one full LOPO pass (for projecting the permutation-null cost)
    t0 = time.perf_counter()
    try:
        nominal_res = run_lopo(df, "nominal")
    except LeakageError as e:
        print(e); sys.exit(1)
    one_pass_s = time.perf_counter() - t0

    try:
        calibrated_res = run_lopo(df, "calibrated")
    except LeakageError as e:
        print(e); sys.exit(1)

    print_scheme_report("nominal", nominal_res)
    print_scheme_report("calibrated", calibrated_res)
    print_class_presence_table(nominal_res, calibrated_res)

    # ---- H1 summary ----
    print(f"\n{'='*64}\n  H1 CONTRAST\n{'='*64}")
    n_macro = nominal_res["macro_balanced_accuracy"]
    c_macro = calibrated_res["macro_balanced_accuracy"]
    print(f"  nominal    macro balanced acc: {n_macro:.3f}")
    print(f"  calibrated macro balanced acc: {c_macro:.3f}")
    print(f"  (H1 predicts calibrated > nominal on real data)")

    # ---- timing projection for permutation null ----
    print(f"\n{'='*64}\n  TIMING (for permutation-null projection)\n{'='*64}")
    print(f"  one full LOPO pass: {one_pass_s:.2f} s")
    n_perm = 1000
    proj = one_pass_s * n_perm
    print(f"  projected {n_perm} permutations (primary model only): "
          f"{proj:.0f} s  (~{proj/60:.1f} min)")
    print(f"  (Dr. Park: run the permutation null for the pre-registered "
          f"primary model ONLY)")

    print(f"\n{'='*64}")
    print("  Pipeline ran end to end. Leakage guard passed (no forbidden cols).")
    if df[GROUP_COL].nunique() <= 20:
        print("  ⚠️  Synthetic / small-n: numbers above are plumbing checks, "
              "not results.")


if __name__ == "__main__":
    main()
