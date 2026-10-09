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

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
args = parser.parse_args()

tag = args.tag


RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = RUN_ROOT / "data" / "compas_farayola_numeric.csv"

RUN008_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_008_none_per_seed_results_A.csv"
)

RUN013_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_013_roc_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)


data = pd.read_csv(DATA_PATH)
run008 = pd.read_csv(RUN008_PATH)
run013 = pd.read_csv(RUN013_PATH)

assert data.shape == (5278, 12)

sha256 = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

assert sha256 == EXPECTED_SHA256


X = data.drop(columns=["two_year_recid"])
y = data["two_year_recid"]

SEEDS = list(range(101, 111))

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]

ROC_METRIC_NAME = "Equal opportunity difference"
ROC_LOW = 0.3
ROC_HIGH = 0.8
ROC_NUM_THRESH = 100
ROC_NUM_MARGIN = 50
ROC_UB = 0.05
ROC_LB = -0.05

rows = []


for seed in SEEDS:

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        shuffle=True,
        stratify=X["race"],
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_snapshot = X_train.copy(deep=True)
    X_test_snapshot = X_test.copy(deep=True)
    y_train_snapshot = y_train.copy(deep=True)
    y_test_snapshot = y_test.copy(deep=True)


    # ---------------------------------------------------------
    # AIF360 datasets
    # ---------------------------------------------------------

    train_df = pd.concat(
        [X_train, y_train],
        axis=1,
    )

    test_df = pd.concat(
        [X_test, y_test],
        axis=1,
    )

    train_bld = BinaryLabelDataset(
        df=train_df,
        label_names=["two_year_recid"],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"],
        privileged_protected_attributes=privileged_group,
    )

    test_bld = BinaryLabelDataset(
        df=test_df,
        label_names=["two_year_recid"],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"],
        privileged_protected_attributes=privileged_group,
    )


    # ---------------------------------------------------------
    # Stage 1: Reweighing
    # ---------------------------------------------------------

    rw = Reweighing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    re_dataset = rw.fit_transform(
        train_bld
    )

    re_weights = (
        re_dataset.instance_weights
        .ravel()
        .astype(float)
    )

    original_weights = (
        train_bld.instance_weights
        .ravel()
        .astype(float)
    )

    weights_changed = not np.allclose(
        original_weights,
        re_weights,
        rtol=0.0,
        atol=1e-12,
    )

    assert weights_changed

    assert np.array_equal(
        train_bld.features,
        re_dataset.features,
    )

    assert np.array_equal(
        train_bld.labels,
        re_dataset.labels,
    )


    # ---------------------------------------------------------
    # Released integration behavior:
    # ignore Reweighing instance_weights
    # ---------------------------------------------------------

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        re_dataset.features,
        re_dataset.labels.ravel(),
    )

    y_pred_lr = (
        model.predict(test_bld.features)
        .astype(int)
    )


    # ---------------------------------------------------------
    # Build hard-score prediction BinaryLabelDataset
    # ---------------------------------------------------------

    prediction_df = X_test.copy()

    prediction_df[
        "two_year_recid"
    ] = y_pred_lr

    pred_lr_bld = BinaryLabelDataset(
        df=prediction_df,
        label_names=["two_year_recid"],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"],
        privileged_protected_attributes=privileged_group,
    )

    lr_scores = (
        pred_lr_bld.scores
        .ravel()
        .astype(float)
    )

    scores_are_hard_labels = bool(
        np.array_equal(
            lr_scores,
            y_pred_lr.astype(float),
        )
    )

    assert scores_are_hard_labels

    assert set(
        np.unique(lr_scores).tolist()
    ).issubset({0.0, 1.0})


    # ---------------------------------------------------------
    # Pre-ROC metrics
    # ---------------------------------------------------------

    lr_metric = ClassificationMetric(
        test_bld,
        pred_lr_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    lr_accuracy = accuracy_score(
        y_test,
        y_pred_lr,
    )

    lr_cm = confusion_matrix(
        y_test,
        y_pred_lr,
        labels=[0, 1],
    )

    lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()

    lr_metrics = {
        "accuracy": float(lr_accuracy),

        "disparate_impact": float(
            lr_metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            lr_metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            lr_metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            lr_metric.false_positive_rate_difference()
        ),
    }


    # ---------------------------------------------------------
    # Compare pre-ROC LR with Run 008
    # ---------------------------------------------------------

    r8 = run008.loc[
        run008["seed"] == seed
    ].iloc[0]

    baseline_cm_match = (
        int(r8["tn"]) == int(lr_tn)
        and int(r8["fp"]) == int(lr_fp)
        and int(r8["fn"]) == int(lr_fn)
        and int(r8["tp"]) == int(lr_tp)
    )

    baseline_metric_match = True

    for name, value in lr_metrics.items():

        if not np.isclose(
            value,
            float(r8[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            baseline_metric_match = False

    assert baseline_cm_match
    assert baseline_metric_match


    # ---------------------------------------------------------
    # Stage 2: ROC
    # ---------------------------------------------------------

    lr_labels_snapshot = (
        pred_lr_bld.labels.copy()
    )

    lr_scores_snapshot = (
        pred_lr_bld.scores.copy()
    )

    roc = RejectOptionClassification(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        metric_name=ROC_METRIC_NAME,
        low_class_thresh=ROC_LOW,
        high_class_thresh=ROC_HIGH,
        num_class_thresh=ROC_NUM_THRESH,
        num_ROC_margin=ROC_NUM_MARGIN,
        metric_ub=ROC_UB,
        metric_lb=ROC_LB,
    )

    pred_roc_bld = roc.fit_predict(
        test_bld,
        pred_lr_bld,
    )

    y_pred_roc = (
        pred_roc_bld.labels
        .ravel()
        .astype(int)
    )

    assert np.array_equal(
        pred_lr_bld.labels,
        lr_labels_snapshot,
    )

    assert np.array_equal(
        pred_lr_bld.scores,
        lr_scores_snapshot,
    )

    changed_predictions = int(
        np.sum(
            y_pred_roc != y_pred_lr
        )
    )


    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

    roc_metric = ClassificationMetric(
        test_bld,
        pred_roc_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred_roc,
    )

    cm = confusion_matrix(
        y_test,
        y_pred_roc,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy),

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


    # ---------------------------------------------------------
    # Compare final result with Run 013
    # ---------------------------------------------------------

    r13 = run013.loc[
        run013["seed"] == seed
    ].iloc[0]

    run013_cm_match = (
        int(r13["tn"]) == int(tn)
        and int(r13["fp"]) == int(fp)
        and int(r13["fn"]) == int(fn)
        and int(r13["tp"]) == int(tp)
    )

    run013_metric_match = True

    for name, value in metrics.items():

        if not np.isclose(
            value,
            float(r13[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run013_metric_match = False


    # ---------------------------------------------------------
    # Integrity
    # ---------------------------------------------------------

    pd.testing.assert_frame_equal(
        X_train,
        X_train_snapshot,
    )

    pd.testing.assert_series_equal(
        y_train,
        y_train_snapshot,
    )

    pd.testing.assert_frame_equal(
        X_test,
        X_test_snapshot,
    )

    pd.testing.assert_series_equal(
        y_test,
        y_test_snapshot,
    )


    rows.append(
        {
            "seed": seed,

            "re_weights_changed":
                weights_changed,

            "re_weight_min":
                float(re_weights.min()),

            "re_weight_max":
                float(re_weights.max()),

            "re_weight_sum":
                float(re_weights.sum()),

            "re_features_unchanged":
                True,

            "re_labels_unchanged":
                True,

            "lr_used_instance_weights":
                False,

            "baseline_run008_cm_match":
                baseline_cm_match,

            "baseline_run008_metric_match":
                baseline_metric_match,

            "roc_scores_are_hard_labels":
                scores_are_hard_labels,

            "roc_unique_score_count":
                int(np.unique(lr_scores).size),

            "roc_selected_classification_threshold":
                float(roc.classification_threshold),

            "roc_selected_margin":
                float(roc.ROC_margin),

            "roc_changed_predictions":
                changed_predictions,

            "run013_cm_match":
                run013_cm_match,

            "run013_metric_match":
                run013_metric_match,

            "train_untouched":
                True,

            "test_untouched":
                True,

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),

            **metrics,
        }
    )


results = pd.DataFrame(rows)


# ---------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------

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

    margin = (
        t_critical
        * std
        / math.sqrt(n)
    )

    summary_rows.append(
        {
            "metric": metric_name,
            "n": n,
            "mean": mean,
            "sample_std": std,
            "min": values.min(),
            "max": values.max(),
            "ci95_lower": mean - margin,
            "ci95_upper": mean + margin,
        }
    )

summary = pd.DataFrame(summary_rows)


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


print("=" * 78)
print("RUN 019 — REWEIGHING + ROC (PP) — COMPAS")
print("=" * 78)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "weights_changed={re_weights_changed} | "
        "LR_used_weights={lr_used_instance_weights} | "
        "Run008_match={baseline_run008_metric_match} | "
        "hard_scores={roc_scores_are_hard_labels} | "
        "threshold={roc_selected_classification_threshold:.6f} | "
        "margin={roc_selected_margin:.6f} | "
        "changed={roc_changed_predictions:.0f} | "
        "Run013_match={run013_metric_match}"
        .format(**row)
    )

print()
print("AGGREGATE SUMMARY")
print("-----------------")

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
