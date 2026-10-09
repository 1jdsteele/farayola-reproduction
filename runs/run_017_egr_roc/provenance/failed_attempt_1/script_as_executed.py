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
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
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

assert data.shape == (5278, 12)

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

assert sha256 == EXPECTED_SHA256


# ---------------------------------------------------------------------
# Predictors / target
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

SEEDS = list(range(101, 111))

BASE_LOGREG_MAX_ITER = 1000

EGR_CONSTRAINT = "DemographicParity"
EGR_EPS = 0.02
EGR_MAX_ITER = 20
EGR_DROP_PROT_ATTR = False

ROC_METRIC_NAME = "Equal opportunity difference"
ROC_LOW_CLASS_THRESH = 0.3
ROC_HIGH_CLASS_THRESH = 0.8
ROC_NUM_CLASS_THRESH = 100
ROC_NUM_MARGIN = 50
ROC_METRIC_UB = 0.05
ROC_METRIC_LB = -0.05

favorable_label = 1
unfavorable_label = 0

protected_attribute_names = ["race"]

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]

rows = []


for seed in SEEDS:

    # -------------------------------------------------------------
    # Split
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
    # AIF360 datasets
    # -------------------------------------------------------------

    train_df = pd.concat(
        [X_train_reset, y_train_reset],
        axis=1,
    )

    test_df = pd.concat(
        [X_test_reset, y_test_reset],
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
        predictions_egr_bld.labels
        .ravel()
        .astype(int)
    )

    egr_labels_snapshot = (
        predictions_egr_bld.labels.copy()
    )

    egr_scores_snapshot = (
        predictions_egr_bld.scores.copy()
    )

    egr_scores = (
        predictions_egr_bld.scores
        .ravel()
        .astype(float)
    )

    # -------------------------------------------------------------
    # EGR score diagnostics
    # -------------------------------------------------------------

    assert np.all(
        np.isfinite(egr_scores)
    )

    assert np.all(
        egr_scores >= 0.0
    )

    assert np.all(
        egr_scores <= 1.0
    )

    egr_unique_score_count = int(
        np.unique(egr_scores).size
    )

    egr_scores_are_hard_labels = bool(
        np.all(
            np.isin(
                egr_scores,
                [0.0, 1.0],
            )
        )
    )

    assert not egr_scores_are_hard_labels, (
        "EGR scores unexpectedly contain only hard 0/1 labels"
    )

    assert egr_unique_score_count > 2

    # -------------------------------------------------------------
    # Pre-ROC EGR evaluation
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
        "accuracy": float(egr_accuracy),

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

    assert len(run011_row) == 1

    run011_row = run011_row.iloc[0]

    assert int(run011_row["tn"]) == int(egr_tn)
    assert int(run011_row["fp"]) == int(egr_fp)
    assert int(run011_row["fn"]) == int(egr_fn)
    assert int(run011_row["tp"]) == int(egr_tp)

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
    # Stage 2: Reject Option Classification
    #
    # This mirrors the pinned repository:
    #
    # ROC.fit_predict(
    #     dataset_test_bld,
    #     predictions_egr_bld
    # )
    #
    # Unlike Run 013, predictions_egr_bld.scores contains
    # continuous EGR probability-like scores.
    # -------------------------------------------------------------

    roc = RejectOptionClassification(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        metric_name=ROC_METRIC_NAME,
        low_class_thresh=ROC_LOW_CLASS_THRESH,
        high_class_thresh=ROC_HIGH_CLASS_THRESH,
        num_class_thresh=ROC_NUM_CLASS_THRESH,
        num_ROC_margin=ROC_NUM_MARGIN,
        metric_ub=ROC_METRIC_UB,
        metric_lb=ROC_METRIC_LB,
    )

    predictions_roc_bld = roc.fit_predict(
        dataset_test_bld,
        predictions_egr_bld,
    )

    y_pred_roc = (
        predictions_roc_bld.labels
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
    ), "ROC unexpectedly modified stored EGR labels"

    assert np.array_equal(
        predictions_egr_bld.scores,
        egr_scores_snapshot,
    ), "ROC unexpectedly modified stored EGR scores"

    assert len(y_pred_roc) == len(y_test_reset)

    assert set(
        np.unique(y_pred_roc).tolist()
    ).issubset({0, 1})

    assert roc.classification_threshold is not None
    assert roc.ROC_margin is not None

    changed_predictions = int(
        np.sum(
            y_pred_roc != y_pred_egr
        )
    )

    # -------------------------------------------------------------
    # Post-ROC evaluation
    # -------------------------------------------------------------

    roc_metric = ClassificationMetric(
        dataset_test_bld,
        predictions_roc_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    roc_accuracy = accuracy_score(
        y_test_reset,
        y_pred_roc,
    )

    roc_cm = confusion_matrix(
        y_test_reset,
        y_pred_roc,
        labels=[0, 1],
    )

    (
        tn,
        fp,
        fn,
        tp,
    ) = roc_cm.ravel()

    # -------------------------------------------------------------
    # Output row
    # -------------------------------------------------------------

    row = {
        "seed": seed,

        "train_rows": len(X_train_reset),
        "test_rows": len(X_test_reset),

        # EGR config
        "egr_constraint": EGR_CONSTRAINT,
        "egr_eps": EGR_EPS,
        "egr_max_iter": EGR_MAX_ITER,
        "egr_drop_prot_attr": EGR_DROP_PROT_ATTR,
        "egr_prediction_seed": seed,
        "base_logreg_max_iter": BASE_LOGREG_MAX_ITER,

        # EGR integrity / score diagnostics
        "egr_matches_run011": True,

        "egr_score_min": float(
            egr_scores.min()
        ),

        "egr_score_max": float(
            egr_scores.max()
        ),

        "egr_unique_score_count":
            egr_unique_score_count,

        "egr_scores_are_hard_labels":
            egr_scores_are_hard_labels,

        "egr_tn": int(egr_tn),
        "egr_fp": int(egr_fp),
        "egr_fn": int(egr_fn),
        "egr_tp": int(egr_tp),

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

        # ROC configuration
        "roc_metric_name": ROC_METRIC_NAME,
        "roc_low_class_thresh": ROC_LOW_CLASS_THRESH,
        "roc_high_class_thresh": ROC_HIGH_CLASS_THRESH,
        "roc_num_class_thresh": ROC_NUM_CLASS_THRESH,
        "roc_num_margin": ROC_NUM_MARGIN,
        "roc_metric_ub": ROC_METRIC_UB,
        "roc_metric_lb": ROC_METRIC_LB,

        "roc_test_calibration": True,

        "roc_selected_classification_threshold": float(
            roc.classification_threshold
        ),

        "roc_selected_margin": float(
            roc.ROC_margin
        ),

        "roc_changed_predictions":
            changed_predictions,

        "roc_predicted_0": int(
            (y_pred_roc == 0).sum()
        ),

        "roc_predicted_1": int(
            (y_pred_roc == 1).sum()
        ),

        # Integrity
        "egr_labels_preserved": True,
        "egr_scores_preserved": True,
        "train_untouched": True,
        "test_untouched": True,

        # Post-ROC confusion matrix
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        # Primary Run 017 outcomes
        "accuracy": float(
            roc_accuracy
        ),

        "disparate_impact": float(
            roc_metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            roc_metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            roc_metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            roc_metric.false_positive_rate_difference()
        ),
    }

    rows.append(row)


results = pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Aggregate summary
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
    std = values.std(ddof=1)
    minimum = values.min()
    maximum = values.max()

    standard_error = (
        std / math.sqrt(n)
    )

    margin = (
        t_critical
        * standard_error
    )

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

summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Outputs
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
print("RUN 017 — EGR + ROC (IP) — COMPAS")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")

print(
    "Stage 1 EGR must reproduce frozen Run 011 exactly."
)

print(
    "Unlike Run 013, ROC receives continuous EGR scores "
    "rather than hard Logistic Regression labels."
)

print(
    "ROC is calibrated and evaluated on the same test set, "
    "matching the pinned repository pathway."
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
print("Stage 2: RejectOptionClassification")
print("ROC metric:", ROC_METRIC_NAME)
print("ROC threshold range:", ROC_LOW_CLASS_THRESH, ROC_HIGH_CLASS_THRESH)
print("ROC num thresholds:", ROC_NUM_CLASS_THRESH)
print("ROC num margins:", ROC_NUM_MARGIN)
print("ROC fairness bounds:", ROC_METRIC_LB, ROC_METRIC_UB)
print("ROC calibrated on test set: True")
print("Seeds:", SEEDS)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "EGR_matches_Run011={egr_matches_run011} | "
        "scores_hard={egr_scores_are_hard_labels} | "
        "unique_scores={egr_unique_score_count:.0f} | "
        "score_range=[{egr_score_min:.6f},{egr_score_max:.6f}] | "
        "threshold={roc_selected_classification_threshold:.6f} | "
        "margin={roc_selected_margin:.6f} | "
        "changed={roc_changed_predictions:.0f}"
        .format(**row)
    )

print()
print("PER-SEED POST-ROC RESULTS")
print("-------------------------")

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
print("AGGREGATE POST-ROC SUMMARY")
print("--------------------------")

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
