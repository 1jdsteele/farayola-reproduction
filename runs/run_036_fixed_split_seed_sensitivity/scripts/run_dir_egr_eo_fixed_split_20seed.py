from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import math

import numpy as np
import pandas as pd

from scipy.stats import t

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing import EqOddsPostprocessing
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
# Paths / fixed protocol
# ---------------------------------------------------------------------

RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

MANIFEST_PATH = (
    RUN_ROOT
    / "metadata"
    / "fixed_split_manifest_seed101.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_DATA_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)

EXPECTED_TRAIN_ID_SHA256 = (
    "db883dde9864689948334065e30b9aaf628d4e406c7d565df492b5cfa96021cb"
)

EXPECTED_TEST_ID_SHA256 = (
    "a9d9f03486e3577a0eac1345527af731b9f473450cb33a53a0330c71d7a42de5"
)

SPLIT_SEED = 101
ALGORITHM_SEEDS = list(range(101, 121))

DIR_REPAIR_LEVEL = 1.0
DIR_SENSITIVE_ATTRIBUTE = "race"

BASE_LOGREG_MAX_ITER = 1000

EGR_CONSTRAINT = "DemographicParity"
EGR_EPS = 0.02
EGR_MAX_ITER = 20
EGR_DROP_PROT_ATTR = False

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def array_sha256(array, dtype):
    normalized = np.ascontiguousarray(
        np.asarray(array, dtype=dtype)
    )

    return hashlib.sha256(
        normalized.tobytes()
    ).hexdigest()


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

data = pd.read_csv(DATA_PATH)
manifest = pd.read_csv(MANIFEST_PATH)

assert data.shape == (5278, 12)

assert data["two_year_recid"].value_counts().to_dict() == {
    0: 2795,
    1: 2483,
}

assert data["race"].value_counts().to_dict() == {
    0: 3175,
    1: 2103,
}

data_sha256 = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

assert data_sha256 == EXPECTED_DATA_SHA256


# ---------------------------------------------------------------------
# Recover canonical fixed split
# ---------------------------------------------------------------------

train_manifest = (
    manifest.loc[manifest["split"] == "train"]
    .sort_values("split_order")
)

test_manifest = (
    manifest.loc[manifest["split"] == "test"]
    .sort_values("split_order")
)

train_ids = train_manifest[
    "row_id"
].to_numpy(dtype=np.int64)

test_ids = test_manifest[
    "row_id"
].to_numpy(dtype=np.int64)

assert len(train_ids) == 3694
assert len(test_ids) == 1584

assert len(
    set(train_ids).intersection(set(test_ids))
) == 0

assert len(
    set(train_ids).union(set(test_ids))
) == len(data)

train_id_sha256 = array_sha256(
    train_ids,
    np.int64,
)

test_id_sha256 = array_sha256(
    test_ids,
    np.int64,
)

assert train_id_sha256 == EXPECTED_TRAIN_ID_SHA256
assert test_id_sha256 == EXPECTED_TEST_ID_SHA256


# ---------------------------------------------------------------------
# Fixed ordered datasets
# ---------------------------------------------------------------------

train_df = (
    data.iloc[train_ids]
    .reset_index(drop=True)
)

test_df = (
    data.iloc[test_ids]
    .reset_index(drop=True)
)

X_train = train_df.drop(
    columns=["two_year_recid"]
)

y_train = train_df[
    "two_year_recid"
]

X_test = test_df.drop(
    columns=["two_year_recid"]
)

y_test = test_df[
    "two_year_recid"
]

NONPROTECTED = [
    column
    for column in X_train.columns
    if column != "race"
]


# ---------------------------------------------------------------------
# Fixed-split integrity
# ---------------------------------------------------------------------

assert int((X_train["race"] == 0).sum()) == 2222
assert int((X_train["race"] == 1).sum()) == 1472

assert int((X_test["race"] == 0).sum()) == 953
assert int((X_test["race"] == 1).sum()) == 631

assert int((y_train == 0).sum()) == 1914
assert int((y_train == 1).sum()) == 1780

assert int((y_test == 0).sum()) == 881
assert int((y_test == 1).sum()) == 703


# ---------------------------------------------------------------------
# Run 20 algorithm seeds
# ---------------------------------------------------------------------

rows = []

for algorithm_seed in ALGORITHM_SEEDS:

    # -------------------------------------------------------------
    # Fresh AIF360 datasets each iteration
    # -------------------------------------------------------------

    train_bld = BinaryLabelDataset(
        df=train_df.copy(deep=True),
        label_names=["two_year_recid"],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"],
        privileged_protected_attributes=privileged_group,
    )

    test_bld = BinaryLabelDataset(
        df=test_df.copy(deep=True),
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
        sensitive_attribute=DIR_SENSITIVE_ATTRIBUTE,
    )

    repaired_bld = dir_transformer.fit_transform(
        train_bld
    )

    repaired_df, _ = (
        repaired_bld.convert_to_dataframe()
    )

    X_train_repaired = repaired_df[
        X_train.columns
    ].copy()

    repaired_labels = (
        repaired_df["two_year_recid"]
        .astype(int)
        .to_numpy()
    )

    assert len(X_train_repaired) == len(X_train)

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

    repaired_feature_sha256 = array_sha256(
        X_train_repaired.to_numpy(dtype=np.float64),
        np.float64,
    )

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

    # Preserve the seeding protocol used in Run 024 while holding
    # the train/test split constant.
    np.random.seed(algorithm_seed)

    egr.fit(
        repaired_bld
    )

    np.random.seed(algorithm_seed)

    pred_egr_bld = egr.predict(
        test_bld
    )

    y_pred_egr = (
        pred_egr_bld.labels
        .ravel()
        .astype(int)
    )

    egr_scores = np.asarray(
        pred_egr_bld.scores,
        dtype=np.float64,
    ).ravel()

    egr_prediction_sha256 = array_sha256(
        y_pred_egr,
        np.int64,
    )

    egr_score_sha256 = array_sha256(
        egr_scores,
        np.float64,
    )

    pre_metric = ClassificationMetric(
        test_bld,
        pred_egr_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    pre_accuracy = float(
        accuracy_score(
            y_test,
            y_pred_egr,
        )
    )

    pre_tn, pre_fp, pre_fn, pre_tp = (
        confusion_matrix(
            y_test,
            y_pred_egr,
            labels=[0, 1],
        ).ravel()
    )

    pre_di = float(
        pre_metric.disparate_impact()
    )

    pre_spd = float(
        pre_metric.statistical_parity_difference()
    )

    pre_eod = float(
        pre_metric.equal_opportunity_difference()
    )

    pre_ped = float(
        pre_metric.false_positive_rate_difference()
    )

    # -------------------------------------------------------------
    # Stage 3: EO
    # -------------------------------------------------------------

    egr_labels_snapshot = (
        pred_egr_bld.labels.copy()
    )

    egr_scores_snapshot = (
        pred_egr_bld.scores.copy()
    )

    eo = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        seed=algorithm_seed,
    )

    pred_eo_bld = eo.fit_predict(
        test_bld,
        pred_egr_bld,
    )

    y_pred_eo = (
        pred_eo_bld.labels
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

    eo_changed_predictions = int(
        np.sum(
            y_pred_eo != y_pred_egr
        )
    )

    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

    final_metric = ClassificationMetric(
        test_bld,
        pred_eo_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    accuracy = float(
        accuracy_score(
            y_test,
            y_pred_eo,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_eo,
        labels=[0, 1],
    ).ravel()

    final_scores = np.asarray(
        pred_eo_bld.scores,
        dtype=np.float64,
    ).ravel()

    prediction_sha256 = array_sha256(
        y_pred_eo,
        np.int64,
    )

    score_sha256 = array_sha256(
        final_scores,
        np.float64,
    )

    # -------------------------------------------------------------
    # Store
    # -------------------------------------------------------------

    rows.append({
        "configuration": "DIR+EGR+EO",

        "split_seed": SPLIT_SEED,
        "algorithm_seed": algorithm_seed,

        "train_rows": len(train_ids),
        "test_rows": len(test_ids),

        "train_row_id_sha256": train_id_sha256,
        "test_row_id_sha256": test_id_sha256,

        "dir_repair_level": DIR_REPAIR_LEVEL,
        "dir_changed_nonprotected_cells": changed_cells,
        "dir_changed_nonprotected_features": len(changed_features),
        "dir_changed_feature_names": ";".join(changed_features),
        "dir_repaired_feature_sha256": repaired_feature_sha256,

        "egr_constraint": EGR_CONSTRAINT,
        "egr_eps": EGR_EPS,
        "egr_max_iter": EGR_MAX_ITER,
        "egr_drop_prot_attr": EGR_DROP_PROT_ATTR,
        "egr_fit_np_seed": algorithm_seed,
        "egr_predict_np_seed": algorithm_seed,

        "pre_eo_prediction_sha256": egr_prediction_sha256,
        "pre_eo_score_sha256": egr_score_sha256,

        "pre_eo_tn": int(pre_tn),
        "pre_eo_fp": int(pre_fp),
        "pre_eo_fn": int(pre_fn),
        "pre_eo_tp": int(pre_tp),

        "pre_eo_accuracy": pre_accuracy,
        "pre_eo_disparate_impact": pre_di,
        "pre_eo_statistical_parity_difference": pre_spd,
        "pre_eo_equal_opportunity_difference": pre_eod,
        "pre_eo_predictive_equality_difference": pre_ped,

        "eo_seed": algorithm_seed,
        "eo_test_calibration": True,
        "eo_changed_predictions": eo_changed_predictions,

        "prediction_sha256": prediction_sha256,
        "score_sha256": score_sha256,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "accuracy": accuracy,

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
    })


results = pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Run 036 integrity assertions
# ---------------------------------------------------------------------

assert len(results) == 20

assert results["split_seed"].nunique() == 1
assert results["split_seed"].iloc[0] == 101

assert results[
    "train_row_id_sha256"
].nunique() == 1

assert results[
    "test_row_id_sha256"
].nunique() == 1

# DIR is deterministic and sees exactly the same training data every time.
assert results[
    "dir_repaired_feature_sha256"
].nunique() == 1

assert results[
    "dir_changed_nonprotected_cells"
].nunique() == 1

assert results[
    "dir_changed_feature_names"
].nunique() == 1


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

    mean = float(values.mean())
    std = float(values.std(ddof=1))

    margin = float(
        t_critical
        * std
        / math.sqrt(n)
    )

    summary_rows.append({
        "configuration": "DIR+EGR+EO",
        "metric": metric_name,
        "n": n,
        "mean": mean,
        "sample_std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    })

summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

results_path = (
    METADATA_DIR
    / f"dir_egr_eo_per_seed_results_{tag}.csv"
)

summary_path = (
    METADATA_DIR
    / f"dir_egr_eo_aggregate_summary_{tag}.csv"
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
# Human-readable output
# ---------------------------------------------------------------------

print("=" * 100)
print("RUN 036 — DIR + EGR + EO — FIXED-SPLIT RANDOM-SEED SENSITIVITY")
print("=" * 100)

print()
print("ENVIRONMENT")
print("-----------")
print("NumPy:", importlib.metadata.version("numpy"))
print("pandas:", importlib.metadata.version("pandas"))
print("scikit-learn:", importlib.metadata.version("scikit-learn"))
print("AIF360:", importlib.metadata.version("aif360"))
print("Fairlearn:", importlib.metadata.version("fairlearn"))

print()
print("PROTOCOL")
print("--------")
print("Split seed:", SPLIT_SEED)
print("Algorithm seeds:", ALGORITHM_SEEDS)
print("Train rows:", len(train_ids))
print("Test rows:", len(test_ids))
print("Train row-ID SHA256:", train_id_sha256)
print("Test row-ID SHA256:", test_id_sha256)

print()
print("SEED-CONTROL DIAGNOSTICS")
print("------------------------")
print(
    "Unique DIR repaired-feature hashes:",
    results["dir_repaired_feature_sha256"].nunique(),
)
print(
    "Unique pre-EO EGR prediction hashes:",
    results["pre_eo_prediction_sha256"].nunique(),
)
print(
    "Unique pre-EO EGR score hashes:",
    results["pre_eo_score_sha256"].nunique(),
)
print(
    "Unique final prediction hashes:",
    results["prediction_sha256"].nunique(),
)
print(
    "Unique final score hashes:",
    results["score_sha256"].nunique(),
)

print()
print("PER-SEED RESULTS")
print("----------------")

for _, row in results.iterrows():
    print(
        "Algorithm seed {algorithm_seed:.0f} | "
        "EO_changed={eo_changed_predictions:.0f} | "
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
        "95% CI=[{ci95_lower:.12f}, {ci95_upper:.12f}]"
        .format(**row)
    )

print()
print("Saved:")
print(results_path)
print(summary_path)
