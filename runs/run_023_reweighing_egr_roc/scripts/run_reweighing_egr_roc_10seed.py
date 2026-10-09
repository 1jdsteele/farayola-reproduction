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
parser.add_argument("--tag", required=True)

args = parser.parse_args()
tag = args.tag

if not tag.replace("_", "").replace("-", "").isalnum():
    raise ValueError("Invalid tag")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

RUN011_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_011_egr_per_seed_results_A.csv"
)

RUN017_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_017_egr_roc_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

data = pd.read_csv(DATA_PATH)
run011 = pd.read_csv(RUN011_PATH)
run017 = pd.read_csv(RUN017_PATH)

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


X = data.drop(
    columns=["two_year_recid"]
)

y = data[
    "two_year_recid"
]


# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

BASE_LOGREG_MAX_ITER = 1000

EGR_CONSTRAINT = "DemographicParity"
EGR_EPS = 0.02
EGR_MAX_ITER = 20
EGR_DROP_PROT_ATTR = False

SCORE_BOUND_TOLERANCE = 1e-12

ROC_METRIC_NAME = "Equal opportunity difference"
ROC_LOW = 0.3
ROC_HIGH = 0.8
ROC_NUM_THRESH = 100
ROC_NUM_MARGIN = 50
ROC_UB = 0.05
ROC_LB = -0.05

privileged_group = [
    {"race": 1}
]

unprivileged_group = [
    {"race": 0}
]

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

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_snapshot = X_train.copy(deep=True)
    X_test_snapshot = X_test.copy(deep=True)

    y_train_snapshot = y_train.copy(deep=True)
    y_test_snapshot = y_test.copy(deep=True)


    # -------------------------------------------------------------
    # AIF360 datasets
    # -------------------------------------------------------------

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


    # -------------------------------------------------------------
    # Stage 1: Reweighing
    # -------------------------------------------------------------

    rw = Reweighing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    re_dataset = rw.fit_transform(
        train_bld
    )

    original_weights = (
        train_bld.instance_weights
        .ravel()
        .astype(float)
    )

    re_weights = (
        re_dataset.instance_weights
        .ravel()
        .astype(float)
    )

    re_weights_changed = not np.allclose(
        original_weights,
        re_weights,
        rtol=0.0,
        atol=1e-12,
    )

    assert re_weights_changed

    assert np.all(
        np.isfinite(re_weights)
    )

    assert np.all(
        re_weights > 0
    )

    re_features_unchanged = np.array_equal(
        train_bld.features,
        re_dataset.features,
    )

    re_labels_unchanged = np.array_equal(
        train_bld.labels,
        re_dataset.labels,
    )

    re_race_unchanged = np.array_equal(
        train_bld.protected_attributes,
        re_dataset.protected_attributes,
    )

    assert re_features_unchanged
    assert re_labels_unchanged
    assert re_race_unchanged


    # -------------------------------------------------------------
    # Stage 2: EGR
    #
    # Reweighted dataset supplied directly.
    # No manual sample_weight supplied.
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
        re_dataset
    )

    np.random.seed(seed)

    pred_egr_bld = egr.predict(
        test_bld
    )

    y_pred_egr = (
        pred_egr_bld.labels
        .ravel()
        .astype(int)
    )

    egr_scores = (
        pred_egr_bld.scores
        .ravel()
        .astype(float)
    )


    # -------------------------------------------------------------
    # EGR score integrity
    #
    # Preserve raw values. Allow only machine-precision range
    # tolerance learned from Run 017.
    # -------------------------------------------------------------

    assert np.all(
        np.isfinite(egr_scores)
    )

    assert np.all(
        egr_scores >= -SCORE_BOUND_TOLERANCE
    ), (
        f"EGR score materially below 0: {egr_scores.min()}"
    )

    assert np.all(
        egr_scores <= 1.0 + SCORE_BOUND_TOLERANCE
    ), (
        f"EGR score materially above 1: {egr_scores.max()}"
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

    assert not egr_scores_are_hard_labels
    assert egr_unique_score_count > 2


    # -------------------------------------------------------------
    # Pre-ROC EGR evaluation
    # -------------------------------------------------------------

    egr_metric = ClassificationMetric(
        test_bld,
        pred_egr_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    egr_accuracy = accuracy_score(
        y_test,
        y_pred_egr,
    )

    egr_cm = confusion_matrix(
        y_test,
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
    # Critical Run 011 control
    # -------------------------------------------------------------

    r11 = run011.loc[
        run011["seed"] == seed
    ]

    assert len(r11) == 1

    r11 = r11.iloc[0]

    run011_cm_match = (
        int(r11["tn"]) == int(egr_tn)
        and int(r11["fp"]) == int(egr_fp)
        and int(r11["fn"]) == int(egr_fn)
        and int(r11["tp"]) == int(egr_tp)
    )

    run011_metric_match = True

    for name, value in egr_metrics.items():

        if not np.isclose(
            value,
            float(r11[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run011_metric_match = False

    assert run011_cm_match
    assert run011_metric_match


    # -------------------------------------------------------------
    # Stage 3: ROC
    # -------------------------------------------------------------

    egr_labels_snapshot = (
        pred_egr_bld.labels.copy()
    )

    egr_scores_snapshot = (
        pred_egr_bld.scores.copy()
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
        pred_egr_bld,
    )

    y_pred_roc = (
        pred_roc_bld.labels
        .ravel()
        .astype(int)
    )

    assert np.array_equal(
        pred_egr_bld.labels,
        egr_labels_snapshot,
    )

    assert np.array_equal(
        pred_egr_bld.scores,
        egr_scores_snapshot,
    )

    changed_predictions = int(
        np.sum(
            y_pred_roc != y_pred_egr
        )
    )


    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

    final_metric = ClassificationMetric(
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
        "accuracy": float(
            accuracy
        ),

        "disparate_impact": float(
            final_metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            final_metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            final_metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            final_metric.false_positive_rate_difference()
        ),
    }


    # -------------------------------------------------------------
    # Compare final result with Run 017 EGR+ROC
    # -------------------------------------------------------------

    r17 = run017.loc[
        run017["seed"] == seed
    ]

    assert len(r17) == 1

    r17 = r17.iloc[0]

    run017_cm_match = (
        int(r17["tn"]) == int(tn)
        and int(r17["fp"]) == int(fp)
        and int(r17["fn"]) == int(fn)
        and int(r17["tp"]) == int(tp)
    )

    run017_metric_match = True

    for name, value in metrics.items():

        if not np.isclose(
            value,
            float(r17[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run017_metric_match = False

    run017_threshold_match = bool(
        np.isclose(
            float(roc.classification_threshold),
            float(
                r17[
                    "roc_selected_classification_threshold"
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        )
    )

    run017_margin_match = bool(
        np.isclose(
            float(roc.ROC_margin),
            float(
                r17[
                    "roc_selected_margin"
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        )
    )

    run017_changed_count_match = (
        int(changed_predictions)
        == int(
            r17[
                "roc_changed_predictions"
            ]
        )
    )


    # -------------------------------------------------------------
    # Source integrity
    # -------------------------------------------------------------

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

    assert len(y_pred_roc) == len(y_test)

    assert set(
        np.unique(y_pred_roc).tolist()
    ).issubset({0, 1})


    # -------------------------------------------------------------
    # Store
    # -------------------------------------------------------------

    rows.append(
        {
            "seed": seed,

            # Re
            "re_weights_changed":
                re_weights_changed,

            "re_weight_min":
                float(re_weights.min()),

            "re_weight_max":
                float(re_weights.max()),

            "re_weight_sum":
                float(re_weights.sum()),

            "re_features_unchanged":
                re_features_unchanged,

            "re_labels_unchanged":
                re_labels_unchanged,

            "re_race_unchanged":
                re_race_unchanged,

            "manual_sample_weight_passed_to_egr":
                False,

            # Pre-ROC EGR
            "run011_pre_roc_cm_match":
                run011_cm_match,

            "run011_pre_roc_metric_match":
                run011_metric_match,

            "egr_score_min":
                float(egr_scores.min()),

            "egr_score_max":
                float(egr_scores.max()),

            "egr_unique_score_count":
                egr_unique_score_count,

            "egr_scores_are_hard_labels":
                egr_scores_are_hard_labels,

            # ROC
            "roc_selected_classification_threshold":
                float(
                    roc.classification_threshold
                ),

            "roc_selected_margin":
                float(
                    roc.ROC_margin
                ),

            "roc_changed_predictions":
                changed_predictions,

            "roc_test_calibration":
                True,

            # Run 017 comparison
            "run017_final_cm_match":
                run017_cm_match,

            "run017_final_metric_match":
                run017_metric_match,

            "run017_threshold_match":
                run017_threshold_match,

            "run017_margin_match":
                run017_margin_match,

            "run017_changed_count_match":
                run017_changed_count_match,

            # Integrity
            "egr_labels_preserved":
                True,

            "egr_scores_preserved":
                True,

            "train_source_untouched":
                True,

            "test_untouched":
                True,

            # Final CM
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),

            # Final metrics
            **metrics,
        }
    )


results = pd.DataFrame(
    rows
)


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
# Human-readable output
# ---------------------------------------------------------------------

print("=" * 78)
print("RUN 023 — REWEIGHING + EGR + ROC (PIP) — COMPAS")
print("=" * 78)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "weights_changed={re_weights_changed} | "
        "manual_weight_to_EGR={manual_sample_weight_passed_to_egr} | "
        "Run011_preROC_match={run011_pre_roc_metric_match} | "
        "hard_scores={egr_scores_are_hard_labels} | "
        "unique_scores={egr_unique_score_count:.0f} | "
        "threshold={roc_selected_classification_threshold:.6f} | "
        "margin={roc_selected_margin:.6f} | "
        "changed={roc_changed_predictions:.0f} | "
        "Run017_final_match={run017_final_metric_match}"
        .format(**row)
    )

print()
print("PER-SEED RESULTS")
print("----------------")

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
