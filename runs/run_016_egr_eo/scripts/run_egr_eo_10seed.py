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

from aif360.algorithms.inprocessing import (
    ExponentiatedGradientReduction,
)
from aif360.algorithms.postprocessing import (
    EqOddsPostprocessing,
)
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

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

RUN011_RESULTS_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_011_egr_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)

data = pd.read_csv(DATA_PATH)
run011_results = pd.read_csv(RUN011_RESULTS_PATH)

assert data.shape == (5278, 12), (
    f"Unexpected dataset shape: {data.shape}"
)

assert data["two_year_recid"].value_counts().to_dict() == {
    0: 2795,
    1: 2483,
}

assert data["race"].value_counts().to_dict() == {
    0: 3175,
    1: 2103,
}

sha256 = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

assert sha256 == EXPECTED_SHA256, (
    f"Unexpected input SHA256: {sha256}"
)


# ---------------------------------------------------------------------
# Predictors and target
# ---------------------------------------------------------------------

X = data.drop(
    columns=["two_year_recid"]
)

y = data[
    "two_year_recid"
]

assert "race" in X.columns


# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------

SEEDS = list(
    range(101, 111)
)

EGR_CONSTRAINT = "DemographicParity"
EGR_EPS = 0.02
EGR_MAX_ITER = 20
EGR_DROP_PROT_ATTR = False

BASE_LOGREG_MAX_ITER = 1000

favorable_label = 1
unfavorable_label = 0

protected_attribute_names = [
    "race"
]

privileged_group = [
    {"race": 1}
]

unprivileged_group = [
    {"race": 0}
]

rows = []


for seed in SEEDS:

    # -------------------------------------------------------------
    # Standardized split
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        shuffle=True,
        stratify=X["race"],
    )

    X_train_reset = (
        X_train
        .reset_index(drop=True)
    )

    y_train_reset = (
        y_train
        .reset_index(drop=True)
    )

    X_test_reset = (
        X_test
        .reset_index(drop=True)
    )

    y_test_reset = (
        y_test
        .reset_index(drop=True)
    )

    X_train_snapshot = (
        X_train_reset
        .copy(deep=True)
    )

    y_train_snapshot = (
        y_train_reset
        .copy(deep=True)
    )

    X_test_snapshot = (
        X_test_reset
        .copy(deep=True)
    )

    y_test_snapshot = (
        y_test_reset
        .copy(deep=True)
    )

    # -------------------------------------------------------------
    # AIF360 datasets
    # -------------------------------------------------------------

    train_df = pd.concat(
        [
            X_train_reset,
            y_train_reset,
        ],
        axis=1,
    )

    test_df = pd.concat(
        [
            X_test_reset,
            y_test_reset,
        ],
        axis=1,
    )

    dataset_train_bld = BinaryLabelDataset(
        df=train_df,
        label_names=["two_year_recid"],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        protected_attribute_names=protected_attribute_names,
        privileged_protected_attributes=privileged_group,
    )

    dataset_test_bld = BinaryLabelDataset(
        df=test_df,
        label_names=["two_year_recid"],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        protected_attribute_names=protected_attribute_names,
        privileged_protected_attributes=privileged_group,
    )

    # -------------------------------------------------------------
    # Stage 1: EGR
    # -------------------------------------------------------------

    base_classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=BASE_LOGREG_MAX_ITER,
    )

    egr = ExponentiatedGradientReduction(
        base_classifier,
        constraints=EGR_CONSTRAINT,
        eps=EGR_EPS,
        max_iter=EGR_MAX_ITER,
        drop_prot_attr=EGR_DROP_PROT_ATTR,
    )

    np.random.seed(seed)

    egr.fit(
        dataset_train_bld
    )

    np.random.seed(seed)

    predictions_egr_bld = egr.predict(
        dataset_test_bld
    )

    y_pred_egr = (
        predictions_egr_bld
        .labels
        .ravel()
        .astype(int)
    )

    egr_labels_snapshot = (
        predictions_egr_bld
        .labels
        .copy()
    )

    # -------------------------------------------------------------
    # Evaluate pre-EO EGR
    # -------------------------------------------------------------

    egr_metric = ClassificationMetric(
        dataset_test_bld,
        predictions_egr_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    egr_accuracy = accuracy_score(
        y_test_reset,
        y_pred_egr,
    )

    egr_cm = confusion_matrix(
        y_test_reset,
        y_pred_egr,
        labels=[0, 1],
    )

    (
        egr_tn,
        egr_fp,
        egr_fn,
        egr_tp,
    ) = egr_cm.ravel()

    egr_metrics = {
        "accuracy": float(
            egr_accuracy
        ),

        "disparate_impact": float(
            egr_metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            egr_metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            egr_metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            egr_metric.false_positive_rate_difference()
        ),
    }

    # -------------------------------------------------------------
    # Critical Run 011 reproduction check
    # -------------------------------------------------------------

    run011_row = run011_results.loc[
        run011_results["seed"] == seed
    ]

    assert len(run011_row) == 1, (
        f"Expected one Run 011 row for seed {seed}"
    )

    run011_row = (
        run011_row.iloc[0]
    )

    assert (
        int(run011_row["tn"]) == int(egr_tn)
    )

    assert (
        int(run011_row["fp"]) == int(egr_fp)
    )

    assert (
        int(run011_row["fn"]) == int(egr_fn)
    )

    assert (
        int(run011_row["tp"]) == int(egr_tp)
    )

    for metric_name, current_value in egr_metrics.items():

        reference_value = float(
            run011_row[metric_name]
        )

        assert np.isclose(
            current_value,
            reference_value,
            rtol=0.0,
            atol=1e-12,
        ), (
            f"Run 011 mismatch for seed {seed}, "
            f"{metric_name}: "
            f"{current_value} vs {reference_value}"
        )

    # -------------------------------------------------------------
    # Stage 2: Equalized Odds
    #
    # Repository-supported structure:
    #
    # eo.fit_predict(
    #     dataset_test_bld,
    #     predictions_egr_bld
    # )
    #
    # This calibrates EO using evaluation test labels.
    # -------------------------------------------------------------

    eo = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        seed=seed,
    )

    predictions_eo_bld = eo.fit_predict(
        dataset_test_bld,
        predictions_egr_bld,
    )

    y_pred_eo = (
        predictions_eo_bld
        .labels
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
        predictions_egr_bld.labels,
        egr_labels_snapshot,
    ), (
        "EO unexpectedly modified stored EGR predictions"
    )

    assert len(y_pred_eo) == len(
        y_test_reset
    )

    assert set(
        np.unique(
            y_pred_eo
        ).tolist()
    ).issubset({0, 1})

    changed_predictions = int(
        np.sum(
            y_pred_eo
            != y_pred_egr
        )
    )

    # -------------------------------------------------------------
    # Post-EO metrics
    # -------------------------------------------------------------

    eo_metric = ClassificationMetric(
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

    (
        tn,
        fp,
        fn,
        tp,
    ) = eo_cm.ravel()

    # -------------------------------------------------------------
    # Output row
    # -------------------------------------------------------------

    row = {
        "seed": seed,

        "train_rows": len(
            X_train_reset
        ),

        "test_rows": len(
            X_test_reset
        ),

        # EGR configuration
        "egr_constraint":
            EGR_CONSTRAINT,

        "egr_eps":
            EGR_EPS,

        "egr_max_iter":
            EGR_MAX_ITER,

        "egr_drop_prot_attr":
            EGR_DROP_PROT_ATTR,

        "egr_prediction_seed":
            seed,

        "base_logreg_max_iter":
            BASE_LOGREG_MAX_ITER,

        # Pre-EO EGR diagnostics
        "egr_matches_run011":
            True,

        "egr_tn":
            int(egr_tn),

        "egr_fp":
            int(egr_fp),

        "egr_fn":
            int(egr_fn),

        "egr_tp":
            int(egr_tp),

        "egr_accuracy":
            egr_metrics["accuracy"],

        "egr_disparate_impact":
            egr_metrics["disparate_impact"],

        "egr_statistical_parity_difference":
            egr_metrics[
                "statistical_parity_difference"
            ],

        "egr_equal_opportunity_difference":
            egr_metrics[
                "equal_opportunity_difference"
            ],

        "egr_predictive_equality_difference":
            egr_metrics[
                "predictive_equality_difference"
            ],

        # EO diagnostics
        "eo_seed":
            seed,

        "eo_test_calibration":
            True,

        "eo_changed_predictions":
            changed_predictions,

        "eo_predicted_0":
            int(
                (y_pred_eo == 0).sum()
            ),

        "eo_predicted_1":
            int(
                (y_pred_eo == 1).sum()
            ),

        "egr_predictions_preserved":
            True,

        "train_untouched":
            True,

        "test_untouched":
            True,

        # Post-EO confusion matrix
        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        # Primary Run 016 outcomes
        "accuracy":
            float(eo_accuracy),

        "disparate_impact":
            float(
                eo_metric.disparate_impact()
            ),

        "statistical_parity_difference":
            float(
                eo_metric.statistical_parity_difference()
            ),

        "equal_opportunity_difference":
            float(
                eo_metric.equal_opportunity_difference()
            ),

        "predictive_equality_difference":
            float(
                eo_metric.false_positive_rate_difference()
            ),
    }

    rows.append(row)


results = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Aggregate post-EO summary
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

t_critical = t.ppf(
    0.975,
    df=n - 1,
)

for metric_name in metric_columns:

    values = results[
        metric_name
    ].astype(float)

    mean = values.mean()

    std = values.std(
        ddof=1
    )

    minimum = values.min()

    maximum = values.max()

    standard_error = (
        std
        / math.sqrt(n)
    )

    margin = (
        t_critical
        * standard_error
    )

    summary_rows.append(
        {
            "metric":
                metric_name,

            "n":
                n,

            "mean":
                mean,

            "sample_std":
                std,

            "min":
                minimum,

            "max":
                maximum,

            "ci95_lower":
                mean - margin,

            "ci95_upper":
                mean + margin,
        }
    )

summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

results.to_csv(
    METADATA_DIR
    / f"per_seed_results_{tag}.csv",
    index=False,
    float_format="%.15g",
)

summary.to_csv(
    METADATA_DIR
    / f"aggregate_summary_{tag}.csv",
    index=False,
    float_format="%.15g",
)


# ---------------------------------------------------------------------
# Human-readable stdout
# ---------------------------------------------------------------------

print("=" * 78)
print("RUN 016 — EGR + EO (IP) — COMPAS")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")

print(
    "Stage 1 EGR must reproduce frozen Run 011 exactly."
)

print(
    "Stage 2 EO is fit using test-set true labels and "
    "EGR test predictions, matching the pinned repository."
)

print(
    "This is test-set calibration/leakage under "
    "conventional evaluation practice."
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
print("Stage 1: ExponentiatedGradientReduction")
print("Constraint:", EGR_CONSTRAINT)
print("eps:", EGR_EPS)
print("EGR max_iter:", EGR_MAX_ITER)
print("drop_prot_attr:", EGR_DROP_PROT_ATTR)
print("Stage 2: EqOddsPostprocessing")
print("EO seed: current experimental seed")
print("EO calibrated on test set: True")
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("Seeds:", SEEDS)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "EGR_matches_Run011={egr_matches_run011} | "
        "EO_seed={eo_seed:.0f} | "
        "EO_changed_predictions={eo_changed_predictions:.0f} | "
        "EGR_predictions_preserved={egr_predictions_preserved}"
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
