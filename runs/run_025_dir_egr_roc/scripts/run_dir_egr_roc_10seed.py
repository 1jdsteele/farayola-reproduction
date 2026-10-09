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

from aif360.algorithms.preprocessing import DisparateImpactRemover
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

RUN015_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_015_dir_egr_per_seed_results_A.csv"
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
run015 = pd.read_csv(RUN015_PATH)
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

NONPROTECTED = [
    column
    for column in X.columns
    if column != "race"
]


# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

DIR_REPAIR_LEVEL = 1.0

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
    # Stage 1: DIR
    # -------------------------------------------------------------

    dir_transformer = DisparateImpactRemover(
        repair_level=DIR_REPAIR_LEVEL,
        sensitive_attribute="race",
    )

    repaired_bld = dir_transformer.fit_transform(
        train_bld
    )

    repaired_df, _ = (
        repaired_bld.convert_to_dataframe()
    )

    X_train_repaired = repaired_df[
        X.columns
    ].copy()

    repaired_labels = (
        repaired_df["two_year_recid"]
        .astype(int)
        .to_numpy()
    )


    # -------------------------------------------------------------
    # DIR integrity
    # -------------------------------------------------------------

    assert np.array_equal(
        repaired_labels,
        y_train.astype(int).to_numpy(),
    )

    assert np.array_equal(
        X_train_repaired["race"].to_numpy(),
        X_train["race"].to_numpy(),
    )

    feature_change_counts = {}

    for column in NONPROTECTED:

        before = X_train[
            column
        ].to_numpy(dtype=float)

        after = X_train_repaired[
            column
        ].to_numpy(dtype=float)

        changed = ~np.isclose(
            before,
            after,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

        feature_change_counts[column] = int(
            changed.sum()
        )

    changed_features = [
        column
        for column, count
        in feature_change_counts.items()
        if count > 0
    ]

    changed_cells = int(
        sum(feature_change_counts.values())
    )

    assert changed_cells > 0


    # -------------------------------------------------------------
    # Stage 2: EGR
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
        repaired_bld
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

    unique_score_count = int(
        np.unique(egr_scores).size
    )

    scores_are_hard_labels = bool(
        np.all(
            np.isin(
                egr_scores,
                [0.0, 1.0],
            )
        )
    )

    assert not scores_are_hard_labels
    assert unique_score_count > 2


    # -------------------------------------------------------------
    # Pre-ROC DIR+EGR metrics
    # -------------------------------------------------------------

    pre_metric = ClassificationMetric(
        test_bld,
        pred_egr_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    pre_accuracy = accuracy_score(
        y_test,
        y_pred_egr,
    )

    pre_cm = confusion_matrix(
        y_test,
        y_pred_egr,
        labels=[0, 1],
    )

    (
        pre_tn,
        pre_fp,
        pre_fn,
        pre_tp,
    ) = pre_cm.ravel()

    pre_metrics = {
        "accuracy": float(
            pre_accuracy
        ),

        "disparate_impact": float(
            pre_metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            pre_metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            pre_metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            pre_metric.false_positive_rate_difference()
        ),
    }


    # -------------------------------------------------------------
    # Critical Run 015 check
    # -------------------------------------------------------------

    r15 = run015.loc[
        run015["seed"] == seed
    ]

    assert len(r15) == 1

    r15 = r15.iloc[0]

    run015_pre_cm_match = (
        int(r15["tn"]) == int(pre_tn)
        and int(r15["fp"]) == int(pre_fp)
        and int(r15["fn"]) == int(pre_fn)
        and int(r15["tp"]) == int(pre_tp)
    )

    run015_pre_metric_match = True

    for name, value in pre_metrics.items():

        if not np.isclose(
            value,
            float(r15[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run015_pre_metric_match = False

    assert run015_pre_cm_match
    assert run015_pre_metric_match


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

    roc_changed_predictions = int(
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
    # Compare final output with Run 015 DIR+EGR.
    #
    # Paper reports the same rounded row for DIR+EGR and
    # DIR+EGR+ROC. Equality here is NOT asserted.
    # -------------------------------------------------------------

    run015_final_cm_match = (
        int(r15["tn"]) == int(tn)
        and int(r15["fp"]) == int(fp)
        and int(r15["fn"]) == int(fn)
        and int(r15["tp"]) == int(tp)
    )

    run015_final_metric_match = True

    for name, value in metrics.items():

        if not np.isclose(
            value,
            float(r15[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run015_final_metric_match = False


    # -------------------------------------------------------------
    # Compare final output with Run 017 EGR+ROC.
    #
    # Diagnostic only. Equality is not expected because DIR is active.
    # -------------------------------------------------------------

    r17 = run017.loc[
        run017["seed"] == seed
    ]

    assert len(r17) == 1

    r17 = r17.iloc[0]

    run017_final_cm_match = (
        int(r17["tn"]) == int(tn)
        and int(r17["fp"]) == int(fp)
        and int(r17["fn"]) == int(fn)
        and int(r17["tp"]) == int(tp)
    )

    run017_final_metric_match = True

    for name, value in metrics.items():

        if not np.isclose(
            value,
            float(r17[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run017_final_metric_match = False


    # -------------------------------------------------------------
    # Integrity
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

            # DIR
            "dir_repair_level":
                DIR_REPAIR_LEVEL,

            "dir_labels_unchanged":
                True,

            "dir_race_unchanged":
                True,

            "dir_changed_nonprotected_cells":
                changed_cells,

            "dir_changed_nonprotected_features":
                len(changed_features),

            "dir_changed_feature_names":
                ";".join(changed_features),

            # Pre-ROC Run 015 control
            "run015_pre_roc_cm_match":
                run015_pre_cm_match,

            "run015_pre_roc_metric_match":
                run015_pre_metric_match,

            # EGR scores
            "egr_score_min":
                float(egr_scores.min()),

            "egr_score_max":
                float(egr_scores.max()),

            "egr_unique_score_count":
                unique_score_count,

            "egr_scores_are_hard_labels":
                scores_are_hard_labels,

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
                roc_changed_predictions,

            "roc_test_calibration":
                True,

            # Compare to Run 015
            "run015_final_cm_match":
                run015_final_cm_match,

            "run015_final_metric_match":
                run015_final_metric_match,

            # Compare to Run 017
            "run017_final_cm_match":
                run017_final_cm_match,

            "run017_final_metric_match":
                run017_final_metric_match,

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
print("RUN 025 — DIR + EGR + ROC (PIP) — COMPAS")
print("=" * 78)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "DIR_changed_cells={dir_changed_nonprotected_cells:.0f} | "
        "Run015_preROC_match={run015_pre_roc_metric_match} | "
        "hard_scores={egr_scores_are_hard_labels} | "
        "unique_scores={egr_unique_score_count:.0f} | "
        "threshold={roc_selected_classification_threshold:.6f} | "
        "margin={roc_selected_margin:.6f} | "
        "changed={roc_changed_predictions:.0f} | "
        "Run015_final_match={run015_final_metric_match} | "
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
