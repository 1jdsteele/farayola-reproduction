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
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
args = parser.parse_args()

tag = args.tag

if not tag.replace("_", "").replace("-", "").isalnum():
    raise ValueError("Invalid tag")


RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

RUN009_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_009_dir_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)


data = pd.read_csv(DATA_PATH)
run009 = pd.read_csv(RUN009_PATH)

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

SEEDS = list(
    range(101, 111)
)

DIR_REPAIR_LEVEL = 1.0

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
    # Stage 1: DIR
    # ---------------------------------------------------------

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

    y_train_repaired = (
        repaired_df["two_year_recid"]
        .astype(int)
        .to_numpy()
    )


    # ---------------------------------------------------------
    # DIR integrity
    # ---------------------------------------------------------

    assert len(X_train_repaired) == len(X_train)

    assert np.array_equal(
        y_train_repaired,
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

        feature_change_counts[
            column
        ] = int(
            changed.sum()
        )

    changed_features = [
        column
        for column, count
        in feature_change_counts.items()
        if count > 0
    ]

    changed_cells = int(
        sum(
            feature_change_counts.values()
        )
    )

    assert changed_cells > 0


    # ---------------------------------------------------------
    # Logistic Regression on repaired training data
    # ---------------------------------------------------------

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        X_train_repaired,
        y_train_repaired,
    )

    y_pred_dir = (
        model.predict(X_test)
        .astype(int)
    )


    # ---------------------------------------------------------
    # Build prediction BinaryLabelDataset from HARD labels.
    #
    # Like Run 013, this means scores become 0/1 labels rather
    # than probability scores.
    # ---------------------------------------------------------

    prediction_df = X_test.copy()

    prediction_df[
        "two_year_recid"
    ] = y_pred_dir

    pred_dir_bld = BinaryLabelDataset(
        df=prediction_df,
        label_names=["two_year_recid"],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"],
        privileged_protected_attributes=privileged_group,
    )

    dir_scores = (
        pred_dir_bld.scores
        .ravel()
        .astype(float)
    )

    scores_are_hard_labels = bool(
        np.array_equal(
            dir_scores,
            y_pred_dir.astype(float),
        )
    )

    assert scores_are_hard_labels

    unique_score_count = int(
        np.unique(dir_scores).size
    )

    assert set(
        np.unique(dir_scores).tolist()
    ).issubset({0.0, 1.0})


    # ---------------------------------------------------------
    # Pre-ROC metrics
    # ---------------------------------------------------------

    dir_metric = ClassificationMetric(
        test_bld,
        pred_dir_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    dir_accuracy = accuracy_score(
        y_test,
        y_pred_dir,
    )

    dir_cm = confusion_matrix(
        y_test,
        y_pred_dir,
        labels=[0, 1],
    )

    (
        dir_tn,
        dir_fp,
        dir_fn,
        dir_tp,
    ) = dir_cm.ravel()

    dir_metrics = {
        "accuracy": float(
            dir_accuracy
        ),

        "disparate_impact": float(
            dir_metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            dir_metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            dir_metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            dir_metric.false_positive_rate_difference()
        ),
    }


    # ---------------------------------------------------------
    # Critical Run 009 reproduction check
    # ---------------------------------------------------------

    r9 = run009.loc[
        run009["seed"] == seed
    ]

    assert len(r9) == 1

    r9 = r9.iloc[0]

    run009_pre_cm_match = (
        int(r9["tn"]) == int(dir_tn)
        and int(r9["fp"]) == int(dir_fp)
        and int(r9["fn"]) == int(dir_fn)
        and int(r9["tp"]) == int(dir_tp)
    )

    run009_pre_metric_match = True

    for name, value in dir_metrics.items():

        reference = float(
            r9[name]
        )

        if not np.isclose(
            value,
            reference,
            rtol=0.0,
            atol=1e-12,
        ):
            run009_pre_metric_match = False

    assert run009_pre_cm_match
    assert run009_pre_metric_match


    # ---------------------------------------------------------
    # Stage 2: ROC
    # ---------------------------------------------------------

    pre_roc_labels_snapshot = (
        pred_dir_bld.labels.copy()
    )

    pre_roc_scores_snapshot = (
        pred_dir_bld.scores.copy()
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
        pred_dir_bld,
    )

    y_pred_roc = (
        pred_roc_bld.labels
        .ravel()
        .astype(int)
    )

    assert np.array_equal(
        pred_dir_bld.labels,
        pre_roc_labels_snapshot,
    )

    assert np.array_equal(
        pred_dir_bld.scores,
        pre_roc_scores_snapshot,
    )

    changed_predictions = int(
        np.sum(
            y_pred_roc
            != y_pred_dir
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
        "accuracy": float(
            accuracy
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


    # ---------------------------------------------------------
    # Compare final result against Run 009.
    #
    # Equality is recorded, not asserted. If ROC is a no-op,
    # these should all match.
    # ---------------------------------------------------------

    run009_final_cm_match = (
        int(r9["tn"]) == int(tn)
        and int(r9["fp"]) == int(fp)
        and int(r9["fn"]) == int(fn)
        and int(r9["tp"]) == int(tp)
    )

    run009_final_metric_match = True

    for name, value in metrics.items():

        reference = float(
            r9[name]
        )

        if not np.isclose(
            value,
            reference,
            rtol=0.0,
            atol=1e-12,
        ):
            run009_final_metric_match = False


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

    assert len(y_pred_roc) == len(y_test)

    assert set(
        np.unique(y_pred_roc).tolist()
    ).issubset({0, 1})


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

            # Pre-ROC control
            "run009_pre_roc_cm_match":
                run009_pre_cm_match,

            "run009_pre_roc_metric_match":
                run009_pre_metric_match,

            # ROC input
            "roc_scores_are_hard_labels":
                scores_are_hard_labels,

            "roc_unique_score_count":
                unique_score_count,

            # ROC configuration/result
            "roc_metric_name":
                ROC_METRIC_NAME,

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

            # Final comparison against DIR alone
            "run009_final_cm_match":
                run009_final_cm_match,

            "run009_final_metric_match":
                run009_final_metric_match,

            # Integrity
            "train_source_untouched":
                True,

            "test_untouched":
                True,

            # Confusion matrix
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),

            # Outcomes
            **metrics,
        }
    )


results = pd.DataFrame(
    rows
)


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
            "metric":
                metric_name,

            "n":
                n,

            "mean":
                mean,

            "sample_std":
                std,

            "min":
                values.min(),

            "max":
                values.max(),

            "ci95_lower":
                mean - margin,

            "ci95_upper":
                mean + margin,
        }
    )

summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------
# Outputs
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------

print("=" * 78)
print("RUN 021 — DIR + ROC (PP) — COMPAS")
print("=" * 78)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "DIR_changed_cells={dir_changed_nonprotected_cells:.0f} | "
        "Run009_preROC_match={run009_pre_roc_metric_match} | "
        "hard_scores={roc_scores_are_hard_labels} | "
        "unique_scores={roc_unique_score_count:.0f} | "
        "threshold={roc_selected_classification_threshold:.6f} | "
        "margin={roc_selected_margin:.6f} | "
        "changed={roc_changed_predictions:.0f} | "
        "Run009_final_match={run009_final_metric_match}"
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
