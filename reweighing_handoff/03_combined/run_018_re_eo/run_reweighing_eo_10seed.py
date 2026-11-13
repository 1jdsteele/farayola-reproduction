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
from aif360.algorithms.postprocessing import EqOddsPostprocessing
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

RUN008_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_008_none_per_seed_results_A.csv"
)

RUN012_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_012_eo_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)


data = pd.read_csv(DATA_PATH)
run008 = pd.read_csv(RUN008_PATH)
run012 = pd.read_csv(RUN012_PATH)

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


X = data.drop(columns=["two_year_recid"])
y = data["two_year_recid"]

SEEDS = list(range(101, 111))

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]

rows = []


for seed in SEEDS:

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

    assert np.array_equal(
        train_bld.features,
        re_dataset.features,
    ), "Reweighing unexpectedly changed features"

    assert np.array_equal(
        train_bld.labels,
        re_dataset.labels,
    ), "Reweighing unexpectedly changed labels"

    assert np.array_equal(
        train_bld.protected_attributes,
        re_dataset.protected_attributes,
    ), "Reweighing unexpectedly changed race"


    # -------------------------------------------------------------
    # Released PP integration behavior:
    #
    # Extract features and labels from re_dataset but DO NOT
    # consume re_dataset.instance_weights.
    # -------------------------------------------------------------

    X_train_re = re_dataset.features
    y_train_re = re_dataset.labels.ravel()

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        X_train_re,
        y_train_re,
    )

    y_pred_lr = (
        model.predict(test_bld.features)
        .astype(int)
    )


    # -------------------------------------------------------------
    # Pre-EO baseline metrics
    # -------------------------------------------------------------

    pred_lr_bld = test_bld.copy()

    pred_lr_bld.labels = (
        y_pred_lr
        .reshape(-1, 1)
        .astype(float)
    )

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


    # -------------------------------------------------------------
    # Verify pre-EO classifier against Run 008
    # -------------------------------------------------------------

    run008_row = run008.loc[
        run008["seed"] == seed
    ]

    assert len(run008_row) == 1

    run008_row = run008_row.iloc[0]

    baseline_cm_match = (
        int(run008_row["tn"]) == int(lr_tn)
        and int(run008_row["fp"]) == int(lr_fp)
        and int(run008_row["fn"]) == int(lr_fn)
        and int(run008_row["tp"]) == int(lr_tp)
    )

    baseline_metric_match = True

    for metric_name, value in lr_metrics.items():

        reference = float(
            run008_row[metric_name]
        )

        if not np.isclose(
            value,
            reference,
            rtol=0.0,
            atol=1e-12,
        ):
            baseline_metric_match = False

    assert baseline_cm_match
    assert baseline_metric_match


    # -------------------------------------------------------------
    # Stage 2: Equalized Odds
    #
    # Repository-style test-set calibration.
    # -------------------------------------------------------------

    lr_labels_snapshot = (
        pred_lr_bld.labels.copy()
    )

    eo = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        seed=seed,
    )

    pred_eo_bld = eo.fit_predict(
        test_bld,
        pred_lr_bld,
    )

    y_pred_eo = (
        pred_eo_bld.labels
        .ravel()
        .astype(int)
    )

    assert np.array_equal(
        pred_lr_bld.labels,
        lr_labels_snapshot,
    )

    changed_predictions = int(
        np.sum(
            y_pred_eo != y_pred_lr
        )
    )


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

    assert len(y_pred_eo) == len(y_test)

    assert set(
        np.unique(y_pred_eo).tolist()
    ).issubset({0, 1})


    # -------------------------------------------------------------
    # Final Re + EO metrics
    # -------------------------------------------------------------

    eo_metric = ClassificationMetric(
        test_bld,
        pred_eo_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred_eo,
    )

    cm = confusion_matrix(
        y_test,
        y_pred_eo,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy),

        "disparate_impact": float(
            eo_metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            eo_metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            eo_metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            eo_metric.false_positive_rate_difference()
        ),
    }


    # -------------------------------------------------------------
    # Compare final result against frozen Run 012 EO
    # -------------------------------------------------------------

    run012_row = run012.loc[
        run012["seed"] == seed
    ]

    assert len(run012_row) == 1

    run012_row = run012_row.iloc[0]

    run012_cm_match = (
        int(run012_row["tn"]) == int(tn)
        and int(run012_row["fp"]) == int(fp)
        and int(run012_row["fn"]) == int(fn)
        and int(run012_row["tp"]) == int(tp)
    )

    run012_metric_match = True

    for metric_name, value in metrics.items():

        reference = float(
            run012_row[metric_name]
        )

        if not np.isclose(
            value,
            reference,
            rtol=0.0,
            atol=1e-12,
        ):
            run012_metric_match = False


    rows.append(
        {
            "seed": seed,

            "train_rows": len(X_train),
            "test_rows": len(X_test),

            "re_weights_changed":
                re_weights_changed,

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

            "re_race_unchanged":
                True,

            "lr_used_instance_weights":
                False,

            "baseline_run008_cm_match":
                baseline_cm_match,

            "baseline_run008_metric_match":
                baseline_metric_match,

            "eo_seed":
                seed,

            "eo_test_calibration":
                True,

            "eo_changed_predictions":
                changed_predictions,

            "run012_cm_match":
                run012_cm_match,

            "run012_metric_match":
                run012_metric_match,

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


# -------------------------------------------------------------
# Aggregate summary
# -------------------------------------------------------------

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

    se = std / math.sqrt(n)

    margin = (
        t_critical * se
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
print("RUN 018 — REWEIGHING + EO (PP) — COMPAS")
print("=" * 78)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "weights_changed={re_weights_changed} | "
        "weight_range=[{re_weight_min:.6f},{re_weight_max:.6f}] | "
        "weight_sum={re_weight_sum:.6f} | "
        "LR_used_weights={lr_used_instance_weights} | "
        "Run008_preEO_match={baseline_run008_metric_match} | "
        "EO_changed={eo_changed_predictions:.0f} | "
        "Run012_final_match={run012_metric_match}"
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
