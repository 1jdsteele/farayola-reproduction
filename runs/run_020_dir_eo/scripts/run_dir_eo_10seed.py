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

RUN009_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_009_dir_per_seed_results_A.csv"
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
run009 = pd.read_csv(RUN009_PATH)
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

NONPROTECTED = [
    column
    for column in X.columns
    if column != "race"
]

SEEDS = list(range(101, 111))

DIR_REPAIR_LEVEL = 1.0

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]

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


    # ---------------------------------------------------------
    # Logistic Regression on repaired training features
    # Test features remain unrepaired.
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
    # Pre-EO prediction dataset
    # ---------------------------------------------------------

    pred_dir_bld = test_bld.copy()

    pred_dir_bld.labels = (
        y_pred_dir
        .reshape(-1, 1)
        .astype(float)
    )


    # ---------------------------------------------------------
    # Pre-EO metrics
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

    dir_tn, dir_fp, dir_fn, dir_tp = (
        dir_cm.ravel()
    )

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

    run009_cm_match = (
        int(r9["tn"]) == int(dir_tn)
        and int(r9["fp"]) == int(dir_fp)
        and int(r9["fn"]) == int(dir_fn)
        and int(r9["tp"]) == int(dir_tp)
    )

    run009_metric_match = True

    for name, value in dir_metrics.items():

        if not np.isclose(
            value,
            float(r9[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run009_metric_match = False

    assert run009_cm_match
    assert run009_metric_match


    # ---------------------------------------------------------
    # Stage 2: Equalized Odds
    # ---------------------------------------------------------

    pre_eo_labels_snapshot = (
        pred_dir_bld.labels.copy()
    )

    eo = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        seed=seed,
    )

    pred_eo_bld = eo.fit_predict(
        test_bld,
        pred_dir_bld,
    )

    y_pred_eo = (
        pred_eo_bld.labels
        .ravel()
        .astype(int)
    )

    assert np.array_equal(
        pred_dir_bld.labels,
        pre_eo_labels_snapshot,
    )

    changed_predictions = int(
        np.sum(
            y_pred_eo != y_pred_dir
        )
    )


    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

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
        "accuracy": float(
            accuracy
        ),

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


    # ---------------------------------------------------------
    # Compare to ordinary EO Run 012 as a diagnostic only.
    # No equality is required.
    # ---------------------------------------------------------

    r12 = run012.loc[
        run012["seed"] == seed
    ]

    assert len(r12) == 1

    r12 = r12.iloc[0]

    run012_cm_match = (
        int(r12["tn"]) == int(tn)
        and int(r12["fp"]) == int(fp)
        and int(r12["fn"]) == int(fn)
        and int(r12["tp"]) == int(tp)
    )

    run012_metric_match = True

    for name, value in metrics.items():

        if not np.isclose(
            value,
            float(r12[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run012_metric_match = False


    # ---------------------------------------------------------
    # Source-data integrity
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

    assert len(y_pred_eo) == len(y_test)

    assert set(
        np.unique(y_pred_eo).tolist()
    ).issubset({0, 1})


    rows.append(
        {
            "seed": seed,

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

            "run009_pre_eo_cm_match":
                run009_cm_match,

            "run009_pre_eo_metric_match":
                run009_metric_match,

            "eo_seed":
                seed,

            "eo_test_calibration":
                True,

            "eo_changed_predictions":
                changed_predictions,

            "run012_final_cm_match":
                run012_cm_match,

            "run012_final_metric_match":
                run012_metric_match,

            "train_source_untouched":
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
print("RUN 020 — DIR + EO (PP) — COMPAS")
print("=" * 78)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "DIR_changed_cells={dir_changed_nonprotected_cells:.0f} | "
        "DIR_changed_features={dir_changed_nonprotected_features:.0f} | "
        "Run009_preEO_match={run009_pre_eo_metric_match} | "
        "EO_changed={eo_changed_predictions:.0f} | "
        "Run012_final_match={run012_final_metric_match}"
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
