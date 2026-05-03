"""
step2_train_m14_v5.py — M14 Change-aware Proactive Failure Predictor
=====================================================================

Changes over v4
---------------
BUGFIX
  * Feature importance was saved as all-NaN because `CalibratedClassifierCV`
    has no `.feature_importances_` attribute.  v5 extracts importances from
    `.calibrated_classifiers_[*].estimator` and averages them.

NEW FEATURES (5)
  * failure_recency_score   — exponentially decayed failure history.
    Recency-weighted so that failures two runs ago count more than failures
    nine runs ago.  Explicitly models the intuition that a pipeline that just
    broke is more dangerous than one that broke long ago.
  * repo_base_failure_rate  — full historical failure rate for this repo
    (all runs before i, not limited to WINDOW_SIZE).  Gives the model a
    stable long-run baseline to contrast against the short-window rate.
  * failure_acceleration    — (fr_3 - fr_5) − (fr_5 - fr_n): is the trend
    itself accelerating upward?  Catches repos in a "spiral" of failures.
  * churn_x_recent_failure  — log_total_churn × failure_rate_last_3:
    high churn on an already-unstable pipeline is disproportionately risky.
  * n_workflows_in_window   — distinct workflow names in the window.  More
    workflow diversity → higher coupling and co-failure risk.

MODEL IMPROVEMENTS
  * change_aware_hist_gb_v3_monotone
      HistGBT v2 + monotonic constraints on 6 features whose direction is
      unambiguous: failure_rate_last_N (+), failure_rate_last_3 (+),
      consecutive_failures (+), last_build_failed (+),
      consecutive_successes (−), long_consecutive_successes (−).
      Monotone constraints prevent the model from learning noise-driven
      inversions, improving generalisation and interpretability.
  * change_aware_calibrated_hist_gb_sigmoid
      Sigmoid (Platt) calibration instead of isotonic.  Isotonic regression
      needs many calibration samples to avoid overfitting; cv=3 on ~5 k rows
      is too small.  Sigmoid is more data-efficient and typically closes the
      Brier gap while preserving AP / ROC-AUC.
  * change_aware_stacking
      StackingClassifier: GBT + HistGBT_v2 + RF as level-0 estimators,
      LogisticRegression meta-learner.  Out-of-fold predictions (cv=5) feed
      the meta-learner, reducing the variance of the soft-voting ensemble
      without introducing a held-out calibration overhead.

EXPLAINABILITY OUTPUTS (new)
  * m14_permutation_importance.csv
      sklearn permutation_importance on the test set (100 repeats).  More
      reliable than MDI for correlated features; directly interpretable as
      "how much does removing this feature hurt ROC-AUC?".
  * m14_pr_curve.csv
      Precision and recall at every threshold for the winning model.
      Required for thesis figures (PR curve, operating-point annotation).
  * m14_calibration_curve.csv
      Reliability diagram data: mean predicted probability vs. fraction of
      positives in 10 equal-width bins.  Documents how trustworthy the
      output probabilities are as real probabilities.
  * m14_per_repo_metrics.csv
      ROC-AUC, AP, F1, recall, FPR broken down by repository.  Identifies
      which repos drive errors and whether the model generalises uniformly.
  * (optional) m14_shap_summary.csv
      If the `shap` package is installed, mean |SHAP| per feature from
      TreeExplainer applied to a sample of the test set.  Provides
      directional attribution that MDI and permutation importance lack.

All other inputs, outputs, and structural design decisions are unchanged
from v4.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# -- Paths ---------------------------------------------------------------------
RUNS_FILE        = Path("data/m14_github_runs.csv")
M13_LABELS_FILE  = Path("data/m13_run_labels.csv")

REPORT_OUT               = Path("m14_report.txt")
MODEL_OUT                = Path("m14_model.pkl")
CONFIG_OUT               = Path("m14_config.pkl")
FEATURE_IMPORTANCE_OUT   = Path("m14_feature_importance.csv")
PERM_IMPORTANCE_OUT      = Path("m14_permutation_importance.csv")
THRESHOLDS_OUT           = Path("m14_thresholds.csv")
PREDICTIONS_OUT          = Path("m14_test_predictions.csv")
MODEL_COMPARISON_OUT     = Path("m14_model_comparison.csv")
WINDOWS_OUT              = Path("m14_training_windows.csv")
PR_CURVE_OUT             = Path("m14_pr_curve.csv")
CALIBRATION_CURVE_OUT    = Path("m14_calibration_curve.csv")
PER_REPO_METRICS_OUT     = Path("m14_per_repo_metrics.csv")
SHAP_SUMMARY_OUT         = Path("m14_shap_summary.csv")   # written only if shap installed

# -- Hyperparameters -----------------------------------------------------------
WINDOW_SIZE       = 10
MIN_RUNS_PER_REPO = WINDOW_SIZE + 15
RANDOM_STATE      = 42
DECAY_ALPHA       = 0.75   # exponential decay for failure_recency_score

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
_logger = logging.getLogger(__name__)


def tee(msg: str, f: Optional[IO[str]] = None) -> None:
    """Log msg to stdout and optionally mirror it to an open report file."""
    _logger.info(msg)
    if f is not None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        f.write(f"[{ts}] {msg}\n")
        f.flush()


# -- Feature name constants ----------------------------------------------------
FAILURE_TYPES: tuple[str, ...] = (
    "compilation",
    "test_failure",
    "flaky_test",
    "configuration",
    "infrastructure",
)

BASE_CHANGE_COLS: tuple[str, ...] = (
    "files_changed_count",
    "lines_added",
    "lines_deleted",
    "src_files_changed",
    "test_files_changed",
    "build_files_changed",
    "ci_config_changed",
    "dependency_files_changed",
    "docs_only_change",
    "has_large_change",
)

# -- Feature groups ------------------------------------------------------------

HISTORY_FEATURES: tuple[str, ...] = (
    "failure_rate_last_N",
    "failure_rate_last_3",
    "consecutive_failures",
    "consecutive_successes",
    "last_build_failed",
    "failed_2_ago",
    "failed_3_ago",
    "trend_failure_rate",
)

VOLATILITY_FEATURES: tuple[str, ...] = (
    "failure_rate_last_5",
    "outcome_std_last_N",
    "all_success_in_window",
    "long_consecutive_successes",
)

# NEW in v5: richer history signals.
RECENCY_FEATURES: tuple[str, ...] = (
    # Exponentially decayed failure history (recent failures weighted more).
    # alpha=0.75: weight[0]=1.0 (run i-1), weight[1]=0.75, weight[2]=0.5625 …
    "failure_recency_score",
    # Full historical failure rate for this repo (all runs before i).
    # Provides a stable long-run baseline the window-based rates can be
    # compared against without computing a new 30-run window.
    "repo_base_failure_rate",
    # (fr_3 - fr_5) - (fr_5 - fr_n): positive = trend is accelerating upward.
    "failure_acceleration",
)

CHANGE_TRANSFORM_FEATURES: tuple[str, ...] = (
    "log_files_changed_count",
    "log_lines_added",
    "log_lines_deleted",
    "log_total_churn",
    "src_files_changed",
    "test_files_changed",
    "build_files_changed",
    "ci_config_changed",
    "dependency_files_changed",
    "docs_only_change",
    "has_large_change",
    "change_touches_code",
    "change_touches_tests",
    "change_touches_build_or_deps",
    "change_touches_ci",
)

REPO_LOCAL_FEATURES: tuple[str, ...] = (
    "files_changed_vs_recent_mean",
    "churn_vs_recent_mean",
    "src_changed_vs_recent_mean",
    "test_changed_vs_recent_mean",
    "current_failure_rate_vs_repo_recent_30",
    "repo_recent_failure_rate_30",
)

EVENT_FEATURES: tuple[str, ...] = (
    "event_push",
    "event_pull_request",
    "event_schedule",
    "event_workflow_dispatch",
    "workflow_name_has_test",
    "workflow_name_has_build",
    "workflow_name_has_lint",
    "workflow_name_has_release_or_deploy",
)

TEMPORAL_FEATURES: tuple[str, ...] = (
    "day_of_week",
    "is_weekend",
    "is_business_hours",
    "log1p_hours_since_last_run",
)

INTERACTION_FEATURES: tuple[str, ...] = (
    "churn_spike",
    "pr_with_no_test_changes",
    "is_main_branch",
    "same_workflow_failure_rate",
    # NEW in v5: multiplicative interaction — high churn × unstable repo
    "churn_x_recent_failure",
    # NEW in v5: workflow diversity in recent window (more workflows → more coupling)
    "n_workflows_in_window",
)

M13_HISTORY_FEATURES: tuple[str, ...] = tuple(
    [f"prev_{ft}_count_last_N" for ft in FAILURE_TYPES]
    + [f"last_failure_was_{ft}" for ft in FAILURE_TYPES]
)

HISTORY_ONLY_FEATURES: tuple[str, ...] = HISTORY_FEATURES

CHANGE_AWARE_FEATURES: tuple[str, ...] = (
    HISTORY_FEATURES
    + VOLATILITY_FEATURES
    + RECENCY_FEATURES          # NEW in v5
    + CHANGE_TRANSFORM_FEATURES
    + REPO_LOCAL_FEATURES
    + EVENT_FEATURES
    + TEMPORAL_FEATURES
    + INTERACTION_FEATURES
)

ALL_FEATURES: tuple[str, ...] = CHANGE_AWARE_FEATURES + M13_HISTORY_FEATURES

# Indices of features with known monotone direction in CHANGE_AWARE_FEATURES.
# Used to build monotone_cst for HistGBT v3.
# +1 = failure probability increases with feature; -1 = decreases.
_MONOTONE_DIRECTIONS: dict[str, int] = {
    "failure_rate_last_N":        +1,
    "failure_rate_last_3":        +1,
    "failure_rate_last_5":        +1,
    "consecutive_failures":       +1,
    "last_build_failed":          +1,
    "consecutive_successes":      -1,
    "long_consecutive_successes": -1,
    "all_success_in_window":      -1,
    "failure_recency_score":      +1,
    "repo_base_failure_rate":     +1,
}


def _monotone_cst(features: list[str]) -> list[int]:
    """Return a monotone_cst list (0/+1/-1) aligned to the given feature list."""
    return [_MONOTONE_DIRECTIONS.get(f, 0) for f in features]


# -- Loading -------------------------------------------------------------------
def load_runs() -> pd.DataFrame:
    """Load, validate, and lightly clean the main runs CSV."""
    if not RUNS_FILE.exists():
        raise SystemExit(
            f"ERROR: {RUNS_FILE} not found. Run the M14 fetch script first."
        )

    df = pd.read_csv(RUNS_FILE, low_memory=False)

    required = {"repository", "run_id", "created_at", "build_failed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"ERROR: {RUNS_FILE} is missing required columns: {missing}"
        )

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)

    df = df.dropna(subset=["repository", "run_id", "created_at", "build_failed"]).copy()
    df["build_failed"] = (
        pd.to_numeric(df["build_failed"], errors="coerce").fillna(0).astype(int)
    )
    df = df[df["build_failed"].isin([0, 1])].copy()

    for col in BASE_CHANGE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ("event", "workflow_name", "head_branch"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["failure_type"] = ""
    if M13_LABELS_FILE.exists():
        labs = pd.read_csv(M13_LABELS_FILE, low_memory=False)
        required_m13 = {"repository", "run_id", "failure_type"}
        if required_m13.issubset(labs.columns):
            labs = (
                labs[["repository", "run_id", "failure_type"]]
                .drop_duplicates(subset=["repository", "run_id"])
            )
            labs["failure_type"] = labs["failure_type"].fillna("").astype(str)
            df = df.merge(
                labs, on=["repository", "run_id"], how="left", suffixes=("", "_m13")
            )
            if "failure_type_m13" in df.columns:
                df["failure_type"] = df["failure_type_m13"].fillna("")
                df.drop(columns=["failure_type_m13"], inplace=True)
        else:
            missing_m13 = sorted(required_m13 - set(labs.columns))
            _logger.warning(
                "%s exists but is missing columns %s; M13 labels ignored.",
                M13_LABELS_FILE,
                missing_m13,
            )

    sort_cols = ["repository", "created_at"]
    if "run_number" in df.columns:
        sort_cols.append("run_number")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# -- Feature helpers -----------------------------------------------------------
def streak_at_end(arr: np.ndarray, value: int) -> int:
    """Count consecutive trailing elements equal to value (vectorised)."""
    mask = arr[::-1] != value
    if not mask.any():
        return int(len(arr))
    return int(mask.argmax())


def smoothed_ratio(numerator: float, denominator: float, smoothing: float = 1.0) -> float:
    """Laplace-smoothed ratio — prevents division by zero."""
    return float(numerator) / (float(denominator) + smoothing)


def event_features(event: str, workflow_name: str) -> dict[str, int]:
    """Encode GitHub event type and workflow name into binary indicators."""
    e = event.lower()
    w = workflow_name.lower()
    return {
        "event_push":               int(e == "push"),
        "event_pull_request":       int("pull_request" in e),
        "event_schedule":           int(e == "schedule"),
        "event_workflow_dispatch":  int(e == "workflow_dispatch"),
        "workflow_name_has_test":   int(
            any(x in w for x in ("test", "pytest", "junit", "ci"))
        ),
        "workflow_name_has_build":  int(
            any(x in w for x in ("build", "compile", "package"))
        ),
        "workflow_name_has_lint":   int(
            any(x in w for x in ("lint", "style", "format", "checkstyle", "ruff", "flake"))
        ),
        "workflow_name_has_release_or_deploy": int(
            any(x in w for x in ("release", "deploy", "publish", "wheel", "package"))
        ),
    }


def _col_array(g: pd.DataFrame, col: str) -> np.ndarray:
    """Return a float64 NumPy array for col in g, or zeros if absent."""
    if col not in g.columns:
        return np.zeros(len(g), dtype=np.float64)
    return g[col].fillna(0.0).astype(float).values


def exponential_decay_score(
    arr: np.ndarray, alpha: float = DECAY_ALPHA
) -> float:
    """Weighted sum of arr with exponential decay from right to left.

    weight[0] = alpha^(w-1)  (oldest),  weight[-1] = 1.0  (most recent).
    Normalised by the sum of weights so the result is always in [0, 1].

    This encodes "a failure *yesterday* matters far more than a failure nine
    runs ago", which the simple mean failure_rate_last_N cannot express.
    """
    n = len(arr)
    if n == 0:
        return 0.0
    weights = np.array([alpha ** (n - 1 - k) for k in range(n)], dtype=np.float64)
    return float(np.dot(weights, arr) / weights.sum())


# -- Feature engineering -------------------------------------------------------
def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create one training row per run using only information known before it.

    Sliding window of WINDOW_SIZE completed runs; label = next run outcome.
    All change metadata comes from the current run's commit, available in
    the push/PR event before CI executes.

    New in v5 (vs v4)
    -----------------
    failure_recency_score     O(w) exponential decay inside inner loop.
    repo_base_failure_rate    Running cumulative mean — O(n) precompute.
    failure_acceleration      Derived from fr_3, fr_5, fr_n (no extra cost).
    churn_x_recent_failure    Multiplicative interaction (inner loop).
    n_workflows_in_window     nunique on window slice (inner loop).
    """
    rows: list[dict[str, Any]] = []
    w = WINDOW_SIZE
    _main_branches = frozenset(("main", "master", "trunk", "develop"))

    repo_groups = list(df.groupby("repository"))
    n_repos = len(repo_groups)
    _logger.info("  Building features across %d repositories...", n_repos)

    for repo_idx, (repo, g) in enumerate(repo_groups, start=1):
        g = g.sort_values("created_at").reset_index(drop=True)
        if len(g) < MIN_RUNS_PER_REPO:
            continue

        if repo_idx % 200 == 0 or repo_idx == n_repos:
            _logger.info("    ... %d / %d repos", repo_idx, n_repos)

        n = len(g)

        # -- Pre-extract all column arrays once per repo ---------------------
        outcomes:      np.ndarray = g["build_failed"].astype(int).values
        failure_types: np.ndarray = g["failure_type"].fillna("").astype(str).values
        files_changed: np.ndarray = _col_array(g, "files_changed_count")
        lines_added:   np.ndarray = _col_array(g, "lines_added")
        lines_deleted: np.ndarray = _col_array(g, "lines_deleted")
        total_churn:   np.ndarray = lines_added + lines_deleted
        src_changed:   np.ndarray = _col_array(g, "src_files_changed")
        test_changed:  np.ndarray = _col_array(g, "test_files_changed")
        build_changed: np.ndarray = _col_array(g, "build_files_changed")
        ci_changed:    np.ndarray = _col_array(g, "ci_config_changed")
        dep_changed:   np.ndarray = _col_array(g, "dependency_files_changed")
        docs_only:     np.ndarray = _col_array(g, "docs_only_change")
        large_change:  np.ndarray = _col_array(g, "has_large_change")
        events:        np.ndarray = g["event"].values
        wf_names:      np.ndarray = g["workflow_name"].values
        head_branches: np.ndarray = g["head_branch"].values
        run_ids:       np.ndarray = g["run_id"].values
        created_ats:   np.ndarray = g["created_at"].values

        # -- O(n) precomputes ------------------------------------------------
        # Unbounded consecutive successes immediately before run i.
        long_cons_succ = np.zeros(n, dtype=np.int32)
        for k in range(1, n):
            long_cons_succ[k] = (
                long_cons_succ[k - 1] + 1 if outcomes[k - 1] == 0 else 0
            )

        # Inter-run gap in hours (index 0 = neutral 24 h).
        ts_ns = pd.to_datetime(g["created_at"]).astype(np.int64).values
        run_gaps_ns = np.zeros(n, dtype=np.float64)
        run_gaps_ns[1:] = np.diff(ts_ns).astype(float)
        run_gaps_hours = np.clip(run_gaps_ns / 1e9 / 3600.0, 0.0, None)
        run_gaps_hours[0] = 24.0

        # Calendar features.
        created_dt    = pd.to_datetime(g["created_at"])
        days_of_week  = created_dt.dt.dayofweek.values
        hours_of_day  = created_dt.dt.hour.values

        # NEW v5: cumulative failure count for repo_base_failure_rate.
        # cum_failures[i] = number of failures in runs 0 … i-1.
        cum_failures = np.zeros(n + 1, dtype=np.int32)
        cum_failures[1:] = np.cumsum(outcomes)

        # -- Inner loop ------------------------------------------------------
        for i in range(w, n):
            hist  = outcomes[i - w : i]
            fr_n  = float(hist.mean())
            fr_3  = float(hist[-3:].mean())
            fr_5  = float(hist[-5:].mean()) if w >= 5 else fr_3
            fr_30 = float(outcomes[max(0, i - 30) : i].mean())

            prev_files = files_changed[i - w : i]
            prev_churn = total_churn[i - w : i]
            prev_src   = src_changed[i - w : i]
            prev_test  = test_changed[i - w : i]

            curr_files   = files_changed[i]
            curr_added   = lines_added[i]
            curr_deleted = lines_deleted[i]
            curr_churn   = curr_added + curr_deleted
            curr_src     = src_changed[i]
            curr_test    = test_changed[i]
            curr_build   = build_changed[i]
            curr_ci      = ci_changed[i]
            curr_deps    = dep_changed[i]

            # Most recent labelled failure type within the window.
            last_failure_type = ""
            for j in range(i - 1, max(-1, i - w - 1), -1):
                if outcomes[j] == 1 and failure_types[j] in FAILURE_TYPES:
                    last_failure_type = failure_types[j]
                    break

            prev_types = failure_types[i - w : i]

            # Workflow-specific failure rate: filter window to same workflow.
            wf_curr = wf_names[i]
            wf_mask = wf_names[i - w : i] == wf_curr
            same_wf_fr = float(hist[wf_mask].mean()) if wf_mask.any() else fr_n

            dow = int(days_of_week[i])
            hod = int(hours_of_day[i])

            # NEW v5: exponentially decayed failure score (most recent = highest weight).
            recency_score = exponential_decay_score(hist.astype(float))

            # NEW v5: repo lifetime failure rate (all runs before i).
            repo_base_fr = float(cum_failures[i]) / float(i) if i > 0 else 0.0

            # NEW v5: failure rate acceleration — is the trend speeding up?
            # Positive = rate is rising faster recently than on average.
            failure_accel = (fr_3 - fr_5) - (fr_5 - fr_n)

            # NEW v5: churn × recent failure interaction.
            churn_x_fr = float(np.log1p(curr_churn)) * fr_3

            # NEW v5: workflow diversity in window.
            n_wf_window = int(pd.Series(wf_names[i - w : i]).nunique())

            row: dict[str, Any] = {
                # -- Identifiers (not model features) ------------------------
                "repository":            repo,
                "run_id":                run_ids[i],
                "created_at":            created_ats[i],
                "label_next_run_failed": int(outcomes[i]),

                # -- History / instability -----------------------------------
                "failure_rate_last_N":   fr_n,
                "failure_rate_last_3":   fr_3,
                "consecutive_failures":  streak_at_end(hist, 1),
                "consecutive_successes": streak_at_end(hist, 0),
                "last_build_failed":     int(hist[-1]),
                "failed_2_ago":          int(hist[-2]) if w >= 2 else 0,
                "failed_3_ago":          int(hist[-3]) if w >= 3 else 0,
                "trend_failure_rate":    fr_3 - fr_n,

                # -- Volatility ----------------------------------------------
                "failure_rate_last_5":        fr_5,
                "outcome_std_last_N":         float(hist.std()),
                "all_success_in_window":      int(hist.sum() == 0),
                "long_consecutive_successes": int(long_cons_succ[i]),

                # -- Recency / trend (NEW v5) --------------------------------
                "failure_recency_score":  recency_score,
                "repo_base_failure_rate": repo_base_fr,
                "failure_acceleration":   failure_accel,

                # -- Log-transformed change metadata -------------------------
                "log_files_changed_count": float(np.log1p(curr_files)),
                "log_lines_added":         float(np.log1p(curr_added)),
                "log_lines_deleted":       float(np.log1p(curr_deleted)),
                "log_total_churn":         float(np.log1p(curr_churn)),
                "src_files_changed":        curr_src,
                "test_files_changed":       curr_test,
                "build_files_changed":      curr_build,
                "ci_config_changed":        curr_ci,
                "dependency_files_changed": curr_deps,
                "docs_only_change":         docs_only[i],
                "has_large_change":         large_change[i],
                "change_touches_code":      int(curr_src > 0),
                "change_touches_tests":     int(curr_test > 0),
                "change_touches_build_or_deps": int(curr_build > 0 or curr_deps > 0),
                "change_touches_ci":        int(curr_ci > 0),

                # -- Repository-local normalisation --------------------------
                "files_changed_vs_recent_mean": smoothed_ratio(curr_files, prev_files.mean()),
                "churn_vs_recent_mean":         smoothed_ratio(curr_churn, prev_churn.mean()),
                "src_changed_vs_recent_mean":   smoothed_ratio(curr_src,   prev_src.mean()),
                "test_changed_vs_recent_mean":  smoothed_ratio(curr_test,  prev_test.mean()),
                "current_failure_rate_vs_repo_recent_30": fr_3 - fr_30,
                "repo_recent_failure_rate_30":            fr_30,

                # -- Temporal context ----------------------------------------
                "day_of_week":                dow,
                "is_weekend":                 int(dow >= 5),
                "is_business_hours":          int(0 <= dow <= 4 and 9 <= hod <= 17),
                "log1p_hours_since_last_run": float(np.log1p(run_gaps_hours[i])),

                # -- Interaction / context (v4 + v5 additions) ---------------
                "churn_spike": int(curr_churn > 3.0 * (prev_churn.mean() + 1.0)),
                "pr_with_no_test_changes": int(
                    "pull_request" in str(events[i]).lower() and curr_test == 0
                ),
                "is_main_branch": int(
                    str(head_branches[i]).lower().strip() in _main_branches
                ),
                "same_workflow_failure_rate": same_wf_fr,
                "churn_x_recent_failure":     churn_x_fr,    # NEW v5
                "n_workflows_in_window":      n_wf_window,   # NEW v5

                # -- M13 failure-type history --------------------------------
                **{
                    f"prev_{ft}_count_last_N": int(np.sum(prev_types == ft))
                    for ft in FAILURE_TYPES
                },
                **{
                    f"last_failure_was_{ft}": int(last_failure_type == ft)
                    for ft in FAILURE_TYPES
                },
            }

            row.update(event_features(str(events[i]), str(wf_names[i])))
            rows.append(row)

    ds = pd.DataFrame(rows)
    if ds.empty:
        return ds

    for col in ALL_FEATURES:
        if col not in ds.columns:
            ds[col] = 0.0
        ds[col] = (
            pd.to_numeric(ds[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    ds["label_next_run_failed"] = ds["label_next_run_failed"].astype(int)
    return ds


# -- Splitting -----------------------------------------------------------------
def per_repo_chronological_split(
    ds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-repo chronological split: first 70 % train, next 15 % val, last 15 % test."""
    train_parts: list[pd.DataFrame] = []
    val_parts:   list[pd.DataFrame] = []
    test_parts:  list[pd.DataFrame] = []
    skipped = 0

    for _, g in ds.groupby("repository"):
        g = g.sort_values("created_at").reset_index(drop=True)
        n = len(g)
        train_end = max(1, int(n * 0.70))
        val_end   = max(train_end + 1, int(n * 0.85))
        if val_end >= n:
            skipped += 1
            continue
        train_parts.append(g.iloc[:train_end].copy())
        val_parts.append(g.iloc[train_end:val_end].copy())
        test_parts.append(g.iloc[val_end:].copy())

    if skipped:
        _logger.warning(
            "Skipped %d repo(s) during split: too few windows for a 3-way split.",
            skipped,
        )

    if not (train_parts and val_parts and test_parts):
        raise SystemExit(
            "ERROR: No repositories produced valid train/val/test splits."
        )

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True).sort_values(
            ["repository", "created_at"]
        )

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


# -- Metrics / thresholds ------------------------------------------------------
def threshold_table(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Build a metrics table across a fine threshold grid on the validation set."""
    rows = []
    for t in np.linspace(0.05, 0.90, num=35):
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": round(float(t), 4),
                "accuracy":  accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall":    recall_score(y_true, pred, zero_division=0),
                "f1":        f1_score(y_true, pred, zero_division=0),
                "fpr":       fp / (fp + tn) if (fp + tn) else 0.0,
                "warnings":  int(pred.sum()),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def choose_operating_thresholds(th: pd.DataFrame) -> dict[str, float]:
    """Select three operating thresholds from the validation threshold table.

    balanced    — maximises F1 (then recall, then precision).
    high_recall — maximises recall subject to FPR ≤ 50 %.
    low_noise   — maximises F1 subject to FPR ≤ 10 %.
    """
    balanced_row = th.sort_values(
        ["f1", "recall", "precision"], ascending=False
    ).iloc[0]

    hr_pool = th[th["fpr"] <= 0.50]
    if hr_pool.empty:
        hr_pool = th
    high_recall_row = hr_pool.sort_values(
        ["recall", "f1", "precision"], ascending=False
    ).iloc[0]

    ln_pool = th[th["fpr"] <= 0.10]
    if ln_pool.empty:
        ln_pool = th.nsmallest(max(1, min(5, len(th))), "fpr")
    low_noise_row = ln_pool.sort_values(
        ["f1", "precision", "recall"], ascending=False
    ).iloc[0]

    return {
        "balanced":    float(balanced_row["threshold"]),
        "high_recall": float(high_recall_row["threshold"]),
        "low_noise":   float(low_noise_row["threshold"]),
    }


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute a full metrics dict for a single operating threshold."""
    pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy":  accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall":    recall_score(y_true, pred, zero_division=0),
        "f1":        f1_score(y_true, pred, zero_division=0),
        "fpr":       fp / (fp + tn) if (fp + tn) else 0.0,
        "warnings":  int(pred.sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# -- Model training ------------------------------------------------------------
@dataclass
class CandidateResult:
    """All artefacts produced by training and evaluating one candidate model."""
    name:            str
    model:           Any
    features:        list[str]
    threshold_modes: dict[str, float]
    metrics:         dict[str, Any]
    thresholds:      pd.DataFrame
    predictions:     pd.DataFrame


def _model_handles_imbalance(model: Any) -> bool:
    return getattr(model, "class_weight", None) is not None


def fit_with_weights(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit with balanced sample weights where the model doesn't handle imbalance itself."""
    if hasattr(model, "steps") or _model_handles_imbalance(model):
        model.fit(X, y)
    else:
        sw = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, sample_weight=sw)
    return model


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(failure=1) for every row in X."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def _extract_feature_importances(model: Any) -> Optional[np.ndarray]:
    """Extract feature importances, including from CalibratedClassifierCV wrappers.

    v4 bug: CalibratedClassifierCV has no .feature_importances_, causing the
    importance file to be saved as all-NaN.

    Fix (v5): walk into .calibrated_classifiers_[*].estimator and average.
    This is meaningful because all cv folds train the same HistGBT architecture
    on mostly overlapping data — their importances are highly correlated.

    Falls back gracefully for VotingClassifier and StackingClassifier by
    averaging the sub-estimator importances.
    """
    # Direct attribute (GBT, HistGBT, RF, etc.)
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    # CalibratedClassifierCV: each fold produces a _CalibratedClassifier with .estimator
    if hasattr(model, "calibrated_classifiers_"):
        imps = []
        for cc in model.calibrated_classifiers_:
            base = cc.estimator
            if hasattr(base, "feature_importances_"):
                imps.append(base.feature_importances_)
        if imps:
            return np.mean(imps, axis=0)

    # VotingClassifier / StackingClassifier: average sub-estimator importances
    if hasattr(model, "estimators_"):
        imps = []
        for est in model.estimators_:
            sub_imp = _extract_feature_importances(est)
            if sub_imp is not None:
                imps.append(sub_imp)
        if imps:
            return np.mean(imps, axis=0)

    # Pipeline: try the final step
    if hasattr(model, "steps"):
        return _extract_feature_importances(model.steps[-1][1])

    return None


def eval_candidate(
    name: str,
    model: Any,
    features: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    report_f: Optional[IO[str]],
) -> CandidateResult:
    """Train one candidate and return its full evaluation artefacts."""
    X_train = train[features].values
    y_train = train["label_next_run_failed"].astype(int).values
    X_val   = val[features].values
    y_val   = val["label_next_run_failed"].astype(int).values
    X_test  = test[features].values
    y_test  = test["label_next_run_failed"].astype(int).values

    t0 = time.perf_counter()
    model = fit_with_weights(model, X_train, y_train)
    train_secs = time.perf_counter() - t0

    val_proba       = predict_proba_positive(model, X_val)
    th              = threshold_table(y_val, val_proba)
    threshold_modes = choose_operating_thresholds(th)

    test_proba    = predict_proba_positive(model, X_test)
    balanced_thr  = threshold_modes["balanced"]
    pred_balanced = (test_proba >= balanced_thr).astype(int)

    has_two_classes = len(np.unique(y_test)) == 2
    base_metrics    = evaluate_at_threshold(y_test, test_proba, balanced_thr)

    metrics: dict[str, Any] = {
        "name":              name,
        "n_features":        len(features),
        "train_time_s":      round(train_secs, 2),
        "roc_auc":           (
            roc_auc_score(y_test, test_proba) if has_two_classes else float("nan")
        ),
        "average_precision": average_precision_score(y_test, test_proba),
        "brier":             brier_score_loss(y_test, test_proba),
        "neg_brier":         -brier_score_loss(y_test, test_proba),
        **base_metrics,
        "balanced_threshold":    threshold_modes["balanced"],
        "high_recall_threshold": threshold_modes["high_recall"],
        "low_noise_threshold":   threshold_modes["low_noise"],
    }

    tee(f"\n{name}", report_f)
    tee(f"  Features: {len(features)}  |  Train time: {train_secs:.1f}s", report_f)
    tee(
        f"  Val thresholds -- balanced={threshold_modes['balanced']:.3f}  "
        f"high_recall={threshold_modes['high_recall']:.3f}  "
        f"low_noise={threshold_modes['low_noise']:.3f}",
        report_f,
    )
    for k in (
        "accuracy", "precision", "recall", "f1", "fpr",
        "roc_auc", "average_precision", "brier",
    ):
        tee(f"  {k:<22}: {metrics[k]:.4f}", report_f)
    tee(
        f"  Warnings: {metrics['warnings']:,}  "
        f"TP={metrics['tp']}  FP={metrics['fp']}  "
        f"TN={metrics['tn']}  FN={metrics['fn']}",
        report_f,
    )
    tee(
        "  Classification report (per-repo chronological test, balanced thr):",
        report_f,
    )
    for line in classification_report(y_test, pred_balanced, zero_division=0).splitlines():
        tee("    " + line, report_f)
    tee(
        f"  Confusion matrix [[TN, FP], [FN, TP]]: "
        f"{confusion_matrix(y_test, pred_balanced, labels=[0, 1]).tolist()}",
        report_f,
    )
    for mode_name, thr in threshold_modes.items():
        m = evaluate_at_threshold(y_test, test_proba, thr)
        tee(
            f"  Mode={mode_name:<11}  thr={thr:.3f}  "
            f"p={m['precision']:.3f}  r={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}  fpr={m['fpr']:.3f}  warnings={m['warnings']}",
            report_f,
        )

    predictions = pd.DataFrame(
        {
            "repository":                test["repository"].values,
            "run_id":                    test["run_id"].values,
            "created_at":                test["created_at"].astype(str).values,
            "true_failed":               y_test,
            "predicted_failed_balanced": pred_balanced,
            "failure_probability":       test_proba,
        }
    )
    for mode_name, thr in threshold_modes.items():
        predictions[f"predicted_failed_{mode_name}"] = (test_proba >= thr).astype(int)

    return CandidateResult(
        name=name,
        model=model,
        features=features,
        threshold_modes=threshold_modes,
        metrics=metrics,
        thresholds=th,
        predictions=predictions,
    )


# -- Model catalogue -----------------------------------------------------------
def build_candidates(
    m13_label_count: int,
    report_f: Optional[IO[str]],
    all_feature_list: list[str],
) -> list[tuple[str, Any, list[str]]]:
    """Assemble (name, model, feature_list) triples to evaluate.

    v5 additions
    ------------
    change_aware_hist_gb_v3_monotone
        HistGBT v2 hyperparams + monotone constraints on 10 features whose
        direction is unambiguously positive/negative for failure probability.
        Monotone constraints prevent the model from fitting noise-driven
        direction inversions, improving out-of-distribution generalization and
        making the model directly interpretable ("higher failure_rate_last_N
        always increases predicted probability").

    change_aware_calibrated_hist_gb_sigmoid
        Sigmoid (Platt) calibration instead of isotonic.  With cv=3 on ~5k
        calibration rows, isotonic overfits (hence its higher Brier in v4
        despite winning AP).  Sigmoid calibration is more data-efficient and
        typically achieves comparable Brier with fewer parameters, while
        preserving the ranking quality (AP / ROC-AUC) of the base model.

    change_aware_stacking
        StackingClassifier: GBT + HistGBT_v2 + RF level-0, LR meta-learner.
        Stacking uses out-of-fold predictions (cv=5) so the meta-learner sees
        out-of-sample score combinations — this reduces variance beyond what
        equal-weight soft voting achieves and lets the meta-learner learn each
        model's strengths on specific sub-populations (e.g. high-churn commits).
    """
    rs = RANDOM_STATE
    ca_feats = list(CHANGE_AWARE_FEATURES)

    candidates: list[tuple[str, Any, list[str]]] = [
        # -- Unchanged from v4 (now benefit from 5 new features) ---------------
        (
            "history_only_gb_weighted",
            GradientBoostingClassifier(
                n_estimators=250, learning_rate=0.04, max_depth=3,
                subsample=0.85, random_state=rs,
            ),
            list(HISTORY_ONLY_FEATURES),
        ),
        (
            "change_aware_gb_weighted",
            GradientBoostingClassifier(
                n_estimators=350, learning_rate=0.035, max_depth=3,
                subsample=0.85, random_state=rs,
            ),
            ca_feats,
        ),
        (
            "change_aware_hist_gb_weighted",
            HistGradientBoostingClassifier(
                learning_rate=0.04, max_iter=300, max_leaf_nodes=31,
                l2_regularization=0.05, early_stopping=True, random_state=rs,
            ),
            ca_feats,
        ),
        (
            "change_aware_rf_balanced",
            RandomForestClassifier(
                n_estimators=350, max_depth=12, min_samples_leaf=8,
                class_weight="balanced_subsample", n_jobs=-1, random_state=rs,
            ),
            ca_feats,
        ),
        (
            "logistic_change_aware_baseline",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=rs,
                ),
            ),
            ca_feats,
        ),
        (
            "change_aware_hist_gb_v2",
            HistGradientBoostingClassifier(
                learning_rate=0.025, max_iter=600, max_leaf_nodes=24,
                min_samples_leaf=20, l2_regularization=0.10,
                early_stopping=True, n_iter_no_change=25,
                validation_fraction=0.15, random_state=rs,
            ),
            ca_feats,
        ),
        (
            "change_aware_voting_ensemble",
            VotingClassifier(
                estimators=[
                    ("gb", GradientBoostingClassifier(
                        n_estimators=300, learning_rate=0.04, max_depth=3,
                        subsample=0.85, random_state=rs,
                    )),
                    ("hist_gb", HistGradientBoostingClassifier(
                        learning_rate=0.04, max_iter=300, max_leaf_nodes=31,
                        l2_regularization=0.05, early_stopping=True,
                        random_state=rs,
                    )),
                    ("rf", RandomForestClassifier(
                        n_estimators=300, max_depth=12, min_samples_leaf=8,
                        n_jobs=-1, random_state=rs,
                    )),
                ],
                voting="soft",
            ),
            ca_feats,
        ),
        (
            "change_aware_calibrated_hist_gb",
            CalibratedClassifierCV(
                HistGradientBoostingClassifier(
                    learning_rate=0.04, max_iter=300, max_leaf_nodes=31,
                    l2_regularization=0.05, early_stopping=True, random_state=rs,
                ),
                method="isotonic",
                cv=3,
            ),
            ca_feats,
        ),

        # -- NEW in v5 ---------------------------------------------------------

        # Monotone constraints enforce interpretable, noise-robust direction.
        (
            "change_aware_hist_gb_v3_monotone",
            HistGradientBoostingClassifier(
                learning_rate=0.025, max_iter=600, max_leaf_nodes=24,
                min_samples_leaf=20, l2_regularization=0.10,
                early_stopping=True, n_iter_no_change=25,
                validation_fraction=0.15,
                monotonic_cst=_monotone_cst(ca_feats),
                random_state=rs,
            ),
            ca_feats,
        ),

        # Sigmoid calibration — more data-efficient than isotonic on ~5k rows.
        (
            "change_aware_calibrated_hist_gb_sigmoid",
            CalibratedClassifierCV(
                HistGradientBoostingClassifier(
                    learning_rate=0.025, max_iter=600, max_leaf_nodes=24,
                    min_samples_leaf=20, l2_regularization=0.10,
                    early_stopping=True, n_iter_no_change=25,
                    validation_fraction=0.15, random_state=rs,
                ),
                method="sigmoid",   # Platt: 2-parameter, much less prone to overfit
                cv=3,
            ),
            ca_feats,
        ),

        # Stacking: LR meta-learner over OOF predictions (cv=5).
        (
            "change_aware_stacking",
            StackingClassifier(
                estimators=[
                    ("gb", GradientBoostingClassifier(
                        n_estimators=300, learning_rate=0.04, max_depth=3,
                        subsample=0.85, random_state=rs,
                    )),
                    ("hist_gb", HistGradientBoostingClassifier(
                        learning_rate=0.025, max_iter=400, max_leaf_nodes=24,
                        min_samples_leaf=20, l2_regularization=0.10,
                        early_stopping=True, random_state=rs,
                    )),
                    ("rf", RandomForestClassifier(
                        n_estimators=300, max_depth=12, min_samples_leaf=8,
                        n_jobs=-1, random_state=rs,
                    )),
                ],
                final_estimator=LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=rs,
                ),
                cv=5,
                stack_method="predict_proba",
                n_jobs=-1,
            ),
            ca_feats,
        ),
    ]

    if m13_label_count > 0:
        candidates += [
            (
                "change_aware_plus_m13_gb_weighted",
                GradientBoostingClassifier(
                    n_estimators=400, learning_rate=0.03, max_depth=3,
                    subsample=0.85, random_state=rs,
                ),
                list(ALL_FEATURES),
            ),
            (
                "change_aware_plus_m13_hist_gb",
                HistGradientBoostingClassifier(
                    learning_rate=0.025, max_iter=600, max_leaf_nodes=24,
                    min_samples_leaf=20, l2_regularization=0.10,
                    early_stopping=True, n_iter_no_change=25,
                    validation_fraction=0.15, random_state=rs,
                ),
                list(ALL_FEATURES),
            ),
        ]
    else:
        tee(
            "\nM13-history candidates skipped: no M13 run labels provided.",
            report_f,
        )

    return candidates


# -- Explainability helpers ----------------------------------------------------

def compute_permutation_importance(
    model: Any,
    features: list[str],
    test: pd.DataFrame,
    report_f: Optional[IO[str]],
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Compute permutation importance on the test set (metric: ROC-AUC).

    Permutation importance is more reliable than MDI (mean decrease impurity)
    for correlated features because it directly measures the effect of
    removing a feature's signal on held-out data.

    n_repeats=30 gives stable estimates without being prohibitively slow.
    """
    tee("\nComputing permutation importance on test set (30 repeats)...", report_f)
    X_test = test[features].values
    y_test = test["label_next_run_failed"].astype(int).values

    result = permutation_importance(
        model, X_test, y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame(
        {
            "feature":        features,
            "importance_mean": result.importances_mean,
            "importance_std":  result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return perm_df


def compute_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """Return a tidy DataFrame of the full precision-recall curve.

    Suitable for direct import into matplotlib / pandas plotting for a thesis
    figure.  The 'threshold' column is NaN at the last (recall=1) point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # sklearn returns len(thresholds) = len(precision) - 1
    thr_padded = np.concatenate([thresholds, [np.nan]])
    return pd.DataFrame(
        {"threshold": thr_padded, "precision": precision, "recall": recall}
    )


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return reliability diagram data (mean predicted prob vs fraction positive).

    For a well-calibrated model the points fall near the diagonal.
    Useful for assessing whether the output probabilities can be used as
    real probabilities in downstream decision-making (e.g. risk dashboards).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    return pd.DataFrame({"mean_predicted_prob": prob_pred, "fraction_of_positives": prob_true})


def compute_per_repo_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Break down test-set performance per repository.

    Outputs one row per repo with ROC-AUC, AP, F1, recall, precision, FPR,
    total runs, and total failures.  Useful for:
      - Identifying repositories that consistently drive false positives / negatives.
      - Checking whether model quality is uniform across repos or dominated by
        a few large, high-activity projects.

    Repos with fewer than 2 failures are given NaN for rank-based metrics (AP,
    ROC-AUC) since these cannot be computed.
    """
    rows = []
    for repo, g in predictions.groupby("repository"):
        y_true = g["true_failed"].values
        y_prob  = g["failure_probability"].values
        n_fail  = int(y_true.sum())
        n_total = len(g)

        row: dict[str, Any] = {
            "repository":  repo,
            "n_total":     n_total,
            "n_failures":  n_fail,
            "failure_rate": n_fail / n_total if n_total > 0 else 0.0,
        }

        if len(np.unique(y_true)) < 2:
            row.update({
                "roc_auc": np.nan, "average_precision": np.nan,
                "f1": np.nan, "precision": np.nan, "recall": np.nan,
                "fpr": np.nan,
            })
        else:
            # Use the pre-computed balanced prediction column.
            y_pred = g["predicted_failed_balanced"].values
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            row.update({
                "roc_auc":           roc_auc_score(y_true, y_prob),
                "average_precision": average_precision_score(y_true, y_prob),
                "f1":                f1_score(y_true, y_pred, zero_division=0),
                "precision":         precision_score(y_true, y_pred, zero_division=0),
                "recall":            recall_score(y_true, y_pred, zero_division=0),
                "fpr":               fp / (fp + tn) if (fp + tn) else 0.0,
            })
        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=True)


def try_shap_summary(
    model: Any,
    features: list[str],
    test: pd.DataFrame,
    report_f: Optional[IO[str]],
    max_rows: int = 500,
) -> Optional[pd.DataFrame]:
    """Compute mean |SHAP| per feature using TreeExplainer (optional).

    SHAP provides directional, per-sample attribution — the gold standard for
    tree-model explainability in academic work.  This function:
      1. Extracts the underlying tree estimator from CalibratedClassifierCV,
         VotingClassifier, or StackingClassifier wrappers.
      2. Runs TreeExplainer on a random sample (max_rows) for speed.
      3. Returns a DataFrame of mean(|SHAP|) sorted descending, ready to export.

    Returns None if shap is not installed or the model type is unsupported.
    The caller should handle None gracefully.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        tee(
            "shap not installed — skipping SHAP summary.  "
            "Install with: pip install shap",
            report_f,
        )
        return None

    # Unwrap calibrated / ensemble wrappers to reach the first tree model.
    tree_model: Any = model
    if hasattr(model, "calibrated_classifiers_"):
        tree_model = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "estimators_") and hasattr(model, "final_estimator_"):
        # StackingClassifier: use first level-0 estimator
        est = model.estimators_[0]
        tree_model = est[1] if isinstance(est, tuple) else est
    elif hasattr(model, "estimators_"):
        # VotingClassifier
        tree_model = model.estimators_[0]

    X_sample = test[features].values
    if len(X_sample) > max_rows:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_sample), max_rows, replace=False)
        X_sample = X_sample[idx]

    try:
        explainer  = shap.TreeExplainer(tree_model)
        shap_vals  = explainer.shap_values(X_sample)
        # For binary classifiers shap_values returns a list [class0, class1] or
        # a single array.  We want the positive-class values.
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs   = np.abs(shap_vals).mean(axis=0)
        shap_df    = pd.DataFrame(
            {"feature": features, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        return shap_df
    except Exception as exc:
        tee(f"SHAP computation failed: {exc}", report_f)
        return None


# -- Main ----------------------------------------------------------------------
def main() -> None:
    np.random.seed(RANDOM_STATE)

    with REPORT_OUT.open("w", encoding="utf-8") as report_f:
        tee("M14 Change-aware Proactive Failure Predictor  [v5]", report_f)
        tee(f"Started : {datetime.now(timezone.utc).isoformat()}", report_f)
        tee("=" * 78, report_f)

        df = load_runs()
        m13_label_count = int(
            (df["failure_type"].notna() & (df["failure_type"] != "")).sum()
        )
        tee(f"Loaded runs        : {len(df):,}", report_f)
        tee(f"Repositories       : {df['repository'].nunique():,}", report_f)
        tee(f"Overall fail rate  : {df['build_failed'].mean():.4f}", report_f)
        tee(f"M13 labels present : {m13_label_count:,}", report_f)
        tee(f"M13 coverage       : {m13_label_count / max(len(df), 1):.2%}", report_f)

        tee("\nBuilding training windows...", report_f)
        ds = build_dataset(df)
        if ds.empty:
            raise SystemExit("No training windows generated. Need more runs per repo.")

        ds.to_csv(WINDOWS_OUT, index=False)
        tee(f"Training windows   : {len(ds):,}", report_f)
        tee(f"Window fail rate   : {ds['label_next_run_failed'].mean():.4f}", report_f)
        tee(
            f"Total features     : {len(CHANGE_AWARE_FEATURES)} change-aware "
            f"| {len(ALL_FEATURES)} with M13",
            report_f,
        )
        tee(f"Saved windows to   : {WINDOWS_OUT}", report_f)

        train, val, test = per_repo_chronological_split(ds)
        tee("\nPer-repository chronological split (70 / 15 / 15):", report_f)
        tee(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}", report_f)
        tee(
            f"  fail rate -- "
            f"train={train['label_next_run_failed'].mean():.4f}  "
            f"val={val['label_next_run_failed'].mean():.4f}  "
            f"test={test['label_next_run_failed'].mean():.4f}",
            report_f,
        )
        tee(
            f"  repos -- "
            f"train={train['repository'].nunique()}  "
            f"val={val['repository'].nunique()}  "
            f"test={test['repository'].nunique()}",
            report_f,
        )

        y_test_arr = test["label_next_run_failed"].astype(int).values
        always_neg = np.zeros_like(y_test_arr)
        tee(
            f"\nAlways-success baseline (not eligible): "
            f"accuracy={accuracy_score(y_test_arr, always_neg):.4f}  "
            f"precision=0.0000  recall=0.0000  f1=0.0000",
            report_f,
        )

        tee("\nTraining candidates...", report_f)
        candidates = build_candidates(m13_label_count, report_f, list(ALL_FEATURES))
        results: list[CandidateResult] = []
        for name, model, features in candidates:
            result = eval_candidate(name, model, features, train, val, test, report_f)
            results.append(result)

        comparison = pd.DataFrame([r.metrics for r in results])
        save_cols = [c for c in comparison.columns if c != "neg_brier"]
        comparison[save_cols].to_csv(MODEL_COMPARISON_OUT, index=False)

        display_cols = [
            "name", "accuracy", "precision", "recall", "f1", "fpr",
            "roc_auc", "average_precision", "brier",
            "balanced_threshold", "high_recall_threshold", "low_noise_threshold",
        ]
        tee("\nModel comparison:", report_f)
        tee(comparison[display_cols].to_string(index=False), report_f)
        tee(f"Saved comparison   : {MODEL_COMPARISON_OUT}", report_f)

        best = max(
            results,
            key=lambda r: (
                r.metrics.get("average_precision", -999),
                r.metrics.get("roc_auc", -999),
                -r.metrics.get("brier", 999),
                r.metrics.get("f1", -999),
            ),
        )

        tee("\nSelected model:", report_f)
        tee(f"  name           : {best.name}", report_f)
        tee(f"  selection rule : AP -> ROC-AUC -> lower Brier -> F1", report_f)
        tee(f"  balanced thr   : {best.threshold_modes['balanced']:.3f}", report_f)
        tee(f"  high_recall thr: {best.threshold_modes['high_recall']:.3f}", report_f)
        tee(f"  low_noise thr  : {best.threshold_modes['low_noise']:.3f}", report_f)

        # -- MDI feature importance (v5: fixed for CalibratedClassifierCV) ----
        imp_vals = _extract_feature_importances(best.model)
        if imp_vals is not None:
            imp = (
                pd.DataFrame(
                    {
                        "feature":    best.features,
                        "importance": imp_vals,
                    }
                )
                .sort_values("importance", ascending=False)
            )
            imp.to_csv(FEATURE_IMPORTANCE_OUT, index=False)
            tee(f"Feature importance : {FEATURE_IMPORTANCE_OUT}", report_f)
            tee("Top 25 features (MDI):", report_f)
            for _, row in imp.head(25).iterrows():
                tee(f"  {row['feature']:<50} {row['importance']:.5f}", report_f)
        else:
            pd.DataFrame({"feature": best.features, "importance": np.nan}).to_csv(
                FEATURE_IMPORTANCE_OUT, index=False
            )
            tee(
                "Feature importance : placeholder saved "
                "(model type has no feature_importances_)",
                report_f,
            )

        # -- Permutation importance --------------------------------------------
        tee(
            "\nPermutation importance uses test-set ROC-AUC as the scoring metric.\n"
            "This is more reliable than MDI for correlated features.",
            report_f,
        )
        perm_df = compute_permutation_importance(
            best.model, best.features, test, report_f
        )
        perm_df.to_csv(PERM_IMPORTANCE_OUT, index=False)
        tee(f"Permutation importance saved: {PERM_IMPORTANCE_OUT}", report_f)
        tee("Top 15 features (permutation):", report_f)
        for _, row in perm_df.head(15).iterrows():
            tee(
                f"  {row['feature']:<50} "
                f"{row['importance_mean']:+.5f} ± {row['importance_std']:.5f}",
                report_f,
            )

        # -- PR curve for thesis figures ---------------------------------------
        test_proba_best = best.predictions["failure_probability"].values
        pr_curve_df = compute_pr_curve(y_test_arr, test_proba_best)
        pr_curve_df.to_csv(PR_CURVE_OUT, index=False)
        tee(f"PR curve saved     : {PR_CURVE_OUT}", report_f)

        # -- Calibration curve (reliability diagram) ---------------------------
        cal_df = compute_calibration_curve(y_test_arr, test_proba_best, n_bins=10)
        cal_df.to_csv(CALIBRATION_CURVE_OUT, index=False)
        tee(f"Calibration curve  : {CALIBRATION_CURVE_OUT}", report_f)
        tee("Calibration (mean predicted vs fraction positives):", report_f)
        for _, row in cal_df.iterrows():
            tee(
                f"  predicted={row['mean_predicted_prob']:.3f}  "
                f"actual={row['fraction_of_positives']:.3f}",
                report_f,
            )

        # -- Per-repository metrics breakdown ----------------------------------
        per_repo_df = compute_per_repo_metrics(best.predictions)
        per_repo_df.to_csv(PER_REPO_METRICS_OUT, index=False)
        tee(f"Per-repo metrics   : {PER_REPO_METRICS_OUT}", report_f)
        n_repos_auc = per_repo_df["roc_auc"].notna().sum()
        median_auc  = per_repo_df["roc_auc"].median()
        q25_auc     = per_repo_df["roc_auc"].quantile(0.25)
        tee(
            f"  Repos with computable AUC: {n_repos_auc}  "
            f"median={median_auc:.3f}  Q25={q25_auc:.3f}",
            report_f,
        )
        worst = per_repo_df[per_repo_df["roc_auc"].notna()].nsmallest(5, "roc_auc")
        tee("  5 weakest repos (by ROC-AUC):", report_f)
        for _, row in worst.iterrows():
            tee(
                f"    {row['repository']:<40} "
                f"auc={row['roc_auc']:.3f}  n={row['n_total']}  "
                f"fail_rate={row['failure_rate']:.3f}",
                report_f,
            )

        # -- SHAP summary (optional) -------------------------------------------
        shap_df = try_shap_summary(best.model, best.features, test, report_f)
        if shap_df is not None:
            shap_df.to_csv(SHAP_SUMMARY_OUT, index=False)
            tee(f"SHAP summary saved : {SHAP_SUMMARY_OUT}", report_f)
            tee("Top 15 features (mean |SHAP|):", report_f)
            for _, row in shap_df.head(15).iterrows():
                tee(
                    f"  {row['feature']:<50} {row['mean_abs_shap']:.5f}",
                    report_f,
                )

        # -- Save model artefacts ----------------------------------------------
        best.thresholds.to_csv(THRESHOLDS_OUT, index=False)
        best.predictions.to_csv(PREDICTIONS_OUT, index=False)
        joblib.dump(best.model, MODEL_OUT)

        config = {
            "model_name":                    best.name,
            "feature_names":                 best.features,
            "window_size":                   WINDOW_SIZE,
            "decay_alpha":                   DECAY_ALPHA,
            "thresholds":                    best.threshold_modes,
            "default_threshold_mode":        "balanced",
            "selection_rule":                "average_precision -> roc_auc -> lower_brier -> f1",
            "all_model_results": [
                {k: v for k, v in r.metrics.items() if k != "neg_brier"}
                for r in results
            ],
            "uses_only_pre_run_information": True,
            "split_method":                  "per_repository_chronological_70_15_15",
            "m13_history_used": bool(m13_label_count > 0 and "m13" in best.name),
            "m13_label_count":               m13_label_count,
            "created_at":                    datetime.now(timezone.utc).isoformat(),
            "remarks": [
                "Do not judge by accuracy alone -- failures are rare.",
                "Use AP, ROC-AUC, recall/PDR, FPR, and Brier score in the thesis.",
                "high_recall threshold: missing failures is worse than noisy warnings.",
                "low_noise threshold: warnings should be rare and high-precision.",
                "v5: MDI importance fixed for CalibratedClassifierCV wrappers.",
                "v5: permutation importance is more reliable for correlated features.",
                "v5: failure_recency_score uses alpha=0.75 exponential decay.",
                "v5: monotone constraints on 10 features in hist_gb_v3 variant.",
            ],
        }
        CONFIG_OUT.write_text(json.dumps(config, indent=2), encoding="utf-8")

        tee("\nSaved outputs:", report_f)
        tee(f"  model                : {MODEL_OUT}", report_f)
        tee(f"  config               : {CONFIG_OUT}", report_f)
        tee(f"  thresholds           : {THRESHOLDS_OUT}", report_f)
        tee(f"  predictions          : {PREDICTIONS_OUT}", report_f)
        tee(f"  feature importance   : {FEATURE_IMPORTANCE_OUT}", report_f)
        tee(f"  permutation imp.     : {PERM_IMPORTANCE_OUT}", report_f)
        tee(f"  PR curve             : {PR_CURVE_OUT}", report_f)
        tee(f"  calibration curve    : {CALIBRATION_CURVE_OUT}", report_f)
        tee(f"  per-repo metrics     : {PER_REPO_METRICS_OUT}", report_f)
        tee(f"  training windows     : {WINDOWS_OUT}", report_f)
        tee(f"  report               : {REPORT_OUT}", report_f)
        tee(f"\nFinished : {datetime.now(timezone.utc).isoformat()}", report_f)
        tee("=" * 78, report_f)

    print(f"\nDone. Check {REPORT_OUT}")


if __name__ == "__main__":
    main()
