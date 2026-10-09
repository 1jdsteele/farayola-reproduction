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

from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
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
METADATA_DIR = RUN_ROOT / "metadata"


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)

data = pd.read_csv(DATA_PATH)

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

EGR_CONSTRAINT = "DemographicParity"
EGR_EPS = 0.02
EGR_MAX_ITER = 20
EGR_DROP_PROT_ATTR = False
BASE_LOGREG_MAX_ITER = 1000

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

    # Preserve exact source data before EGR.
    X_train_snapshot = X_train_reset.copy(deep=True)
    y_train_snapshot = y_train_reset.copy(deep=True)
    X_test_snapshot = X_test_reset.copy(deep=True)
    y_test_snapshot = y_test_reset.copy(deep=True)

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

    assert "race" in dataset_train_bld.feature_names
    assert "race" in dataset_test_bld.feature_names

    train_features_before = dataset_train_bld.features.copy()
    train_labels_before = dataset_train_bld.labels.copy()
    train_protected_before = (
        dataset_train_bld.protected_attributes.copy()
    )

    # -------------------------------------------------------------
    # Individual EGR intervention
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

    # Explicit reproducibility control for any training-side randomness.
    np.random.seed(seed)

    egr.fit(dataset_train_bld)

    # EGR prediction is randomized. Reset NumPy immediately before
    # prediction so A/B executions are reproducible.
    np.random.seed(seed)

    predictions_bld = egr.predict(dataset_test_bld)

    y_pred = (
        predictions_bld.labels
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
        dataset_train_bld.features,
        train_features_before,
    ), "EGR unexpectedly modified training features"

    assert np.array_equal(
        dataset_train_bld.labels,
        train_labels_before,
    ), "EGR unexpectedly modified training labels"

    assert np.array_equal(
        dataset_train_bld.protected_attributes,
        train_protected_before,
    ), "EGR unexpectedly modified protected attributes"

    assert len(y_pred) == len(y_test_reset), (
        "Unexpected number of EGR predictions"
    )

    unique_predictions = set(np.unique(y_pred).tolist())

    assert unique_predictions.issubset({0, 1}), (
        f"Unexpected EGR prediction labels: {unique_predictions}"
    )

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

    metric = ClassificationMetric(
        dataset_test_bld,
        predictions_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    accuracy = accuracy_score(
        y_test_reset,
        y_pred,
    )

    cm = confusion_matrix(
        y_test_reset,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

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

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "predicted_0": int((y_pred == 0).sum()),
        "predicted_1": int((y_pred == 1).sum()),

        "egr_constraint": EGR_CONSTRAINT,
        "egr_eps": EGR_EPS,
        "egr_max_iter": EGR_MAX_ITER,
        "egr_drop_prot_attr": EGR_DROP_PROT_ATTR,
        "egr_prediction_seed": seed,
        "base_logreg_max_iter": BASE_LOGREG_MAX_ITER,

        "egr_train_features_unchanged": True,
        "egr_train_labels_unchanged": True,
        "egr_race_unchanged": True,
        "egr_test_untouched": True,

        "accuracy": float(accuracy),

        "disparate_impact": float(
            metric.disparate_impact()
        ),

        "statistical_parity_difference": float(
            metric.statistical_parity_difference()
        ),

        "equal_opportunity_difference": float(
            metric.equal_opportunity_difference()
        ),

        "predictive_equality_difference": float(
            metric.false_positive_rate_difference()
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
print("RUN 011 — INDIVIDUAL EXPONENTIATED GRADIENT REDUCTION — COMPAS")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")
print(
    "Seeds 101-110 are a predetermined diagnostic set. "
    "They are NOT claimed to be the authors' exact ten seeds."
)
print(
    "EGR is the primary scientific change from Run 008."
)
print(
    "NumPy is explicitly reseeded immediately before EGR prediction "
    "to control randomized prediction behavior."
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
print("Fairness intervention: ExponentiatedGradientReduction")
print("Base classifier: LogisticRegression(max_iter=1000)")
print("Constraint:", EGR_CONSTRAINT)
print("eps:", EGR_EPS)
print("EGR max_iter:", EGR_MAX_ITER)
print("drop_prot_attr:", EGR_DROP_PROT_ATTR)
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("Seeds:", SEEDS)
print("Number of trials:", len(SEEDS))
print()

print("EGR INTEGRITY")
print("-------------")

for _, row in results.iterrows():
    print(
        "Seed {seed:.0f} | "
        "prediction_seed={egr_prediction_seed:.0f} | "
        "predicted_0={predicted_0:.0f} | "
        "predicted_1={predicted_1:.0f}"
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
