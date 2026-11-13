from pathlib import Path
import argparse
import hashlib
import math

import numpy as np
import pandas as pd

from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tag",
    required=True,
    help="Execution tag used only for output filenames, e.g. A or B.",
)

args = parser.parse_args()

if not args.tag.replace("_", "").replace("-", "").isalnum():
    raise ValueError(
        "Tag may contain only letters, numbers, underscores, or hyphens."
    )

tag = args.tag


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = RUN_ROOT / "data" / "compas_farayola_numeric.csv"

PARENT_RESULTS_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_008_parent_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)

data = pd.read_csv(DATA_PATH)

parent_results = pd.read_csv(PARENT_RESULTS_PATH)

assert data.shape == (5278, 12), (
    f"Unexpected dataset shape: {data.shape}"
)

assert data["two_year_recid"].value_counts().to_dict() == {
    0: 2795,
    1: 2483,
}, "Unexpected target counts"

assert data["race"].value_counts().to_dict() == {
    0: 3175,
    1: 2103,
}, "Unexpected race counts"

sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

assert sha256 == EXPECTED_SHA256, (
    f"Unexpected input SHA256: {sha256}"
)


# ---------------------------------------------------------------------
# Predictors and target
# ---------------------------------------------------------------------

X = data.drop(columns=["two_year_recid"])
y = data["two_year_recid"]

assert "race" in X.columns


# ---------------------------------------------------------------------
# Experimental protocol
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

BASE_LOGREG_MAX_ITER = 1000

favorable_label = 1
unfavorable_label = 0

protected_attribute_names = ["race"]

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]

rows = []


for seed in SEEDS:

    # -------------------------------------------------------------
    # Split — identical protocol to Run 008
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        shuffle=True,
        stratify=X["race"],
    )

    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    X_train_snapshot = X_train_reset.copy(deep=True)
    y_train_snapshot = y_train_reset.copy(deep=True)

    X_test_snapshot = X_test_reset.copy(deep=True)
    y_test_snapshot = y_test_reset.copy(deep=True)

    # -------------------------------------------------------------
    # Baseline classifier — same as Run 008
    # -------------------------------------------------------------

    classifier = LogisticRegression(
        max_iter=BASE_LOGREG_MAX_ITER
    )

    classifier.fit(
        X_train_reset,
        y_train_reset,
    )

    y_pred_base = classifier.predict(
        X_test_reset
    ).astype(int)

    y_pred_base_snapshot = y_pred_base.copy()

    # -------------------------------------------------------------
    # Construct true and baseline-prediction AIF360 datasets
    # -------------------------------------------------------------

    test_df = pd.concat(
        [
            X_test_reset,
            y_test_reset,
        ],
        axis=1,
    )

    pred_df = X_test_reset.copy()
    pred_df["two_year_recid"] = y_pred_base

    dataset_test_bld = BinaryLabelDataset(
        df=test_df,
        label_names=["two_year_recid"],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        protected_attribute_names=protected_attribute_names,
        privileged_protected_attributes=privileged_group,
    )

    predictions_base_bld = BinaryLabelDataset(
        df=pred_df,
        label_names=["two_year_recid"],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        protected_attribute_names=protected_attribute_names,
        privileged_protected_attributes=privileged_group,
    )

    # -------------------------------------------------------------
    # Pre-EO metric — must reproduce Run 008
    # -------------------------------------------------------------

    metric_base = ClassificationMetric(
        dataset_test_bld,
        predictions_base_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    base_accuracy = accuracy_score(
        y_test_reset,
        y_pred_base,
    )

    base_cm = confusion_matrix(
        y_test_reset,
        y_pred_base,
        labels=[0, 1],
    )

    base_tn, base_fp, base_fn, base_tp = base_cm.ravel()

    # -------------------------------------------------------------
    # Verify this really reproduces the frozen Run 008 baseline
    # -------------------------------------------------------------

    parent_row = parent_results.loc[
        parent_results["seed"] == seed
    ]

    assert len(parent_row) == 1, (
        f"Expected exactly one Run 008 row for seed {seed}"
    )

    parent_row = parent_row.iloc[0]

    assert int(parent_row["tn"]) == int(base_tn)
    assert int(parent_row["fp"]) == int(base_fp)
    assert int(parent_row["fn"]) == int(base_fn)
    assert int(parent_row["tp"]) == int(base_tp)

    parent_metric_checks = {
        "accuracy": base_accuracy,
        "disparate_impact": metric_base.disparate_impact(),
        "statistical_parity_difference":
            metric_base.statistical_parity_difference(),
        "equal_opportunity_difference":
            metric_base.equal_opportunity_difference(),
        "predictive_equality_difference":
            metric_base.false_positive_rate_difference(),
    }

    for column, current_value in parent_metric_checks.items():

        parent_value = float(parent_row[column])

        assert np.isclose(
            current_value,
            parent_value,
            rtol=0.0,
            atol=1e-12,
        ), (
            f"Run 008 baseline mismatch for seed {seed}, "
            f"metric {column}: "
            f"{current_value} vs {parent_value}"
        )

    # -------------------------------------------------------------
    # Equalized Odds post-processing
    #
    # Repository-faithful structure:
    # fit using test true labels + test baseline predictions,
    # then evaluate on that same test set.
    #
    # EO seed = experimental seed for deterministic reconstruction.
    # -------------------------------------------------------------

    eo = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        seed=seed,
    )

    predictions_eo_bld = eo.fit_predict(
        dataset_test_bld,
        predictions_base_bld,
    )

    y_pred_eo = (
        predictions_eo_bld.labels
        .ravel()
        .astype(int)
    )

    # -------------------------------------------------------------
    # Integrity checks
    # -------------------------------------------------------------

    pd.testing.assert_frame_equal(
        X_train_reset,
        X_train_snapshot,
    )

    pd.testing.assert_series_equal(
        y_train_reset,
        y_train_snapshot,
    )

    pd.testing.assert_frame_equal(
        X_test_reset,
        X_test_snapshot,
    )

    pd.testing.assert_series_equal(
        y_test_reset,
        y_test_snapshot,
    )

    assert np.array_equal(
        y_pred_base,
        y_pred_base_snapshot,
    ), "EO unexpectedly modified stored baseline predictions"

    assert len(y_pred_eo) == len(y_test_reset), (
        "Unexpected number of EO predictions"
    )

    unique_eo_predictions = set(
        np.unique(y_pred_eo).tolist()
    )

    assert unique_eo_predictions.issubset({0, 1}), (
        f"Unexpected EO labels: {unique_eo_predictions}"
    )

    changed_predictions = int(
        np.sum(y_pred_eo != y_pred_base)
    )

    # -------------------------------------------------------------
    # Post-EO evaluation
    # -------------------------------------------------------------

    metric_eo = ClassificationMetric(
        dataset_test_bld,
        predictions_eo_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    eo_accuracy = accuracy_score(
        y_test_reset,
        y_pred_eo,
    )

    eo_cm = confusion_matrix(
        y_test_reset,
        y_pred_eo,
        labels=[0, 1],
    )

    tn, fp, fn, tp = eo_cm.ravel()

    # -------------------------------------------------------------
    # Split composition
    # -------------------------------------------------------------

    raw_test_df = X_test_reset.copy()
    raw_test_df["two_year_recid"] = y_test_reset

    test_cross = pd.crosstab(
        raw_test_df["race"],
        raw_test_df["two_year_recid"],
    )

    row = {
        "seed": seed,

        "train_rows": len(X_train_reset),
        "test_rows": len(X_test_reset),

        "train_race_0": int(
            (X_train_reset["race"] == 0).sum()
        ),

        "train_race_1": int(
            (X_train_reset["race"] == 1).sum()
        ),

        "test_race_0": int(
            (X_test_reset["race"] == 0).sum()
        ),

        "test_race_1": int(
            (X_test_reset["race"] == 1).sum()
        ),

        "train_target_0": int(
            (y_train_reset == 0).sum()
        ),

        "train_target_1": int(
            (y_train_reset == 1).sum()
        ),

        "test_target_0": int(
            (y_test_reset == 0).sum()
        ),

        "test_target_1": int(
            (y_test_reset == 1).sum()
        ),

        "test_race0_target0": int(
            test_cross.loc[0, 0]
        ),

        "test_race0_target1": int(
            test_cross.loc[0, 1]
        ),

        "test_race1_target0": int(
            test_cross.loc[1, 0]
        ),

        "test_race1_target1": int(
            test_cross.loc[1, 1]
        ),

        # Baseline / pre-EO information
        "base_tn": int(base_tn),
        "base_fp": int(base_fp),
        "base_fn": int(base_fn),
        "base_tp": int(base_tp),

        "base_accuracy": float(base_accuracy),

        "base_disparate_impact": float(
            metric_base.disparate_impact()
        ),

        "base_statistical_parity_difference": float(
            metric_base.statistical_parity_difference()
        ),

        "base_equal_opportunity_difference": float(
            metric_base.equal_opportunity_difference()
        ),

        "base_predictive_equality_difference": float(
            metric_base.false_positive_rate_difference()
        ),

        "base_matches_run008": True,

        # EO information
        "eo_seed": seed,
        "eo_test_calibration": True,
        "eo_changed_predictions": changed_predictions,

        "eo_predicted_0": int(
            (y_pred_eo == 0).sum()
        ),

        "eo_predicted_1": int(
            (y_pred_eo == 1).sum()
        ),

        # Post-EO confusion matrix
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        # Integrity
        "eo_train_untouched": True,
        "eo_test_features_untouched": True,
        "eo_base_predictions_preserved": True,

        # Primary Run 012 outcome metrics
        "accuracy": float(eo_accuracy),

        "disparate_impact": float(
            metric_eo.disparate_impact()
        ),

        "statistical_parity_difference": float(
            metric_eo.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            metric_eo.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            metric_eo.false_positive_rate_difference()
        ),
    }

    rows.append(row)


results = pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Aggregate summary of POST-EO outcome
# ---------------------------------------------------------------------

metric_columns = [
    "accuracy",
    "disparate_impact",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "predictive_equality_difference",
]

summary_rows = []

n = len(results)
t_critical = t.ppf(0.975, df=n - 1)

for metric_name in metric_columns:

    values = results[metric_name].astype(float)

    mean = values.mean()
    std = values.std(ddof=1)
    minimum = values.min()
    maximum = values.max()

    standard_error = std / math.sqrt(n)
    margin = t_critical * standard_error

    summary_rows.append(
        {
            "metric": metric_name,
            "n": n,
            "mean": mean,
            "sample_std": std,
            "min": minimum,
            "max": maximum,
            "ci95_lower": mean - margin,
            "ci95_upper": mean + margin,
        }
    )

summary = pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------
# Preserve machine-readable outputs
# ---------------------------------------------------------------------

results_path = (
    METADATA_DIR
    / f"per_seed_results_{tag}.csv"
)

summary_path = (
    METADATA_DIR
    / f"aggregate_summary_{tag}.csv"
)

results.to_csv(
    results_path,
    index=False,
    float_format="%.15g",
)

summary.to_csv(
    summary_path,
    index=False,
    float_format="%.15g",
)


# ---------------------------------------------------------------------
# Human-readable stdout
# ---------------------------------------------------------------------

print("=" * 78)
print("RUN 012 — INDIVIDUAL EQUALIZED ODDS — COMPAS")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")

print(
    "EO is the primary scientific change from Run 008."
)

print(
    "Run 012 reproduces the repository structure in which EO is "
    "fit using test-set true labels and baseline test predictions."
)

print(
    "This is test-set calibration/leakage under a conventional "
    "evaluation design and is preserved explicitly for reproduction."
)

print(
    "EO receives the current experimental seed for deterministic "
    "A/B reproduction. This is not claimed as an author setting."
)

print()

print("INPUT")
print("-----")
print("File:", DATA_PATH)
print("SHA256:", sha256)
print("Shape:", data.shape)
print()

print("PROTOCOL")
print("--------")
print("Base classifier: LogisticRegression(max_iter=1000)")
print("Fairness intervention: EqOddsPostprocessing")
print("EO fitted on test labels/predictions: True")
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("Seeds:", SEEDS)
print("Number of trials:", len(SEEDS))
print()

print("EO INTEGRITY")
print("------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "base_matches_run008={base_matches_run008} | "
        "eo_seed={eo_seed:.0f} | "
        "changed_predictions={eo_changed_predictions:.0f} | "
        "predicted_0={eo_predicted_0:.0f} | "
        "predicted_1={eo_predicted_1:.0f}"
        .format(**row)
    )

print()
print("PER-SEED POST-EO RESULTS")
print("------------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "Accuracy={accuracy:.12f} | "
        "DI={disparate_impact:.12f} | "
        "SPD={statistical_parity_difference:.12f} | "
        "EOD={equal_opportunity_difference:.12f} | "
        "PED={predictive_equality_difference:.12f}"
        .format(**row)
    )

print()
print("AGGREGATE POST-EO SUMMARY")
print("-------------------------")

for _, row in summary.iterrows():

    print(
        "{metric}: "
        "mean={mean:.12f}, "
        "std={sample_std:.12f}, "
        "min={min:.12f}, "
        "max={max:.12f}, "
        "95% CI=[{ci95_lower:.12f}, "
        "{ci95_upper:.12f}]"
        .format(**row)
    )
