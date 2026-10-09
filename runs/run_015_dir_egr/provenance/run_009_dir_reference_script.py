from pathlib import Path
import argparse
import hashlib
import math

import numpy as np
import pandas as pd
import scipy
import sklearn
import aif360

from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.algorithms.preprocessing import DisparateImpactRemover
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

nonprotected_features = [
    column
    for column in X.columns
    if column != "race"
]


# ---------------------------------------------------------------------
# Experimental protocol
#
# Parent:
# Run 008 — standardized ten-seed NONE baseline.
#
# Held constant:
# - same reconstructed COMPAS dataset
# - same predictor matrix
# - same target
# - 70/30 train/test split
# - shuffle=True
# - stratify on race
# - seeds 101 through 110
# - LogisticRegression(max_iter=1000)
# - same AIF360 conventions and metrics
#
# Primary scientific change:
# - DisparateImpactRemover(
#       repair_level=1.0,
#       sensitive_attribute="race"
#   )
#   applied to TRAINING DATA ONLY.
#
# No StandardScaler is introduced.
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

REPAIR_LEVEL = 1.0
SENSITIVE_ATTRIBUTE = "race"

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

    X_test_snapshot = X_test.copy(deep=True)

    # -------------------------------------------------------------
    # Construct AIF360 training dataset
    # -------------------------------------------------------------

    train_bld_df = pd.concat(
        [
            X_train_reset,
            y_train_reset,
        ],
        axis=1,
    )

    dataset_train_bld = BinaryLabelDataset(
        df=train_bld_df,
        label_names=["two_year_recid"],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        protected_attribute_names=protected_attribute_names,
        privileged_protected_attributes=privileged_group,
    )

    # -------------------------------------------------------------
    # Apply individual DIR intervention
    # -------------------------------------------------------------

    dir_transformer = DisparateImpactRemover(
        repair_level=REPAIR_LEVEL,
        sensitive_attribute=SENSITIVE_ATTRIBUTE,
    )

    dataset_train_repaired = dir_transformer.fit_transform(
        dataset_train_bld
    )

    repaired_df, _ = dataset_train_repaired.convert_to_dataframe()

    X_train_repaired = repaired_df[X.columns].copy()

    repaired_labels = (
        repaired_df["two_year_recid"]
        .astype(int)
        .to_numpy()
    )

    original_labels = y_train_reset.astype(int).to_numpy()

    # -------------------------------------------------------------
    # DIR integrity checks
    # -------------------------------------------------------------

    assert len(X_train_repaired) == len(X_train_reset), (
        "DIR changed training row count"
    )

    assert np.array_equal(
        repaired_labels,
        original_labels,
    ), "DIR unexpectedly changed target labels"

    assert np.array_equal(
        X_train_repaired["race"].to_numpy(),
        X_train_reset["race"].to_numpy(),
    ), "DIR unexpectedly changed protected race attribute"

    feature_change_counts = {}

    for column in nonprotected_features:

        before = X_train_reset[column].to_numpy(dtype=float)
        after = X_train_repaired[column].to_numpy(dtype=float)

        changed = ~np.isclose(
            before,
            after,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

        feature_change_counts[column] = int(changed.sum())

    changed_feature_names = [
        column
        for column, count in feature_change_counts.items()
        if count > 0
    ]

    changed_nonprotected_cells = sum(
        feature_change_counts.values()
    )

    assert changed_nonprotected_cells > 0, (
        "DIR did not change any non-protected training feature values"
    )

    # Test data must remain completely untouched.
    pd.testing.assert_frame_equal(
        X_test,
        X_test_snapshot,
    )

    # -------------------------------------------------------------
    # Classifier
    # -------------------------------------------------------------

    classifier = LogisticRegression(max_iter=1000)

    classifier.fit(
        X_train_repaired,
        y_train_reset,
    )

    y_pred = classifier.predict(X_test)

    # -------------------------------------------------------------
    # AIF360 evaluation datasets
    # -------------------------------------------------------------

    test_df = pd.concat(
        [
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
        ],
        axis=1,
    )

    pred_df = test_df.copy()
    pred_df["two_year_recid"] = y_pred.astype(int)

    dataset_test_bld = BinaryLabelDataset(
        df=test_df,
        label_names=["two_year_recid"],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        protected_attribute_names=protected_attribute_names,
        privileged_protected_attributes=privileged_group,
    )

    predictions_bld = BinaryLabelDataset(
        df=pred_df,
        label_names=["two_year_recid"],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        protected_attribute_names=protected_attribute_names,
        privileged_protected_attributes=privileged_group,
    )

    metric = ClassificationMetric(
        dataset_test_bld,
        predictions_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # -------------------------------------------------------------
    # Split composition
    # -------------------------------------------------------------

    raw_test_df = X_test.copy()
    raw_test_df["two_year_recid"] = y_test

    test_cross = pd.crosstab(
        raw_test_df["race"],
        raw_test_df["two_year_recid"],
    )

    row = {
        "seed": seed,

        "train_rows": len(X_train),
        "test_rows": len(X_test),

        "train_race_0": int((X_train["race"] == 0).sum()),
        "train_race_1": int((X_train["race"] == 1).sum()),

        "test_race_0": int((X_test["race"] == 0).sum()),
        "test_race_1": int((X_test["race"] == 1).sum()),

        "train_target_0": int((y_train == 0).sum()),
        "train_target_1": int((y_train == 1).sum()),

        "test_target_0": int((y_test == 0).sum()),
        "test_target_1": int((y_test == 1).sum()),

        "test_race0_target0": int(test_cross.loc[0, 0]),
        "test_race0_target1": int(test_cross.loc[0, 1]),
        "test_race1_target0": int(test_cross.loc[1, 0]),
        "test_race1_target1": int(test_cross.loc[1, 1]),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "logreg_iterations": int(classifier.n_iter_[0]),

        "dir_repair_level": REPAIR_LEVEL,
        "dir_sensitive_attribute": SENSITIVE_ATTRIBUTE,

        "dir_race_unchanged": True,
        "dir_labels_unchanged": True,
        "dir_test_untouched": True,

        "dir_changed_nonprotected_cells": int(
            changed_nonprotected_cells
        ),

        "dir_changed_nonprotected_features": int(
            len(changed_feature_names)
        ),

        "dir_changed_feature_names": ";".join(
            changed_feature_names
        ),

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
#
# Preserve signed fairness metrics exactly as returned by AIF360.
#
# Sample standard deviation uses ddof=1.
# 95% confidence interval uses Student's t with df=9.
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
print("RUN 009 — INDIVIDUAL DISPARATE IMPACT REMOVER — COMPAS")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")
print(
    "Seeds 101-110 are a predetermined diagnostic set. "
    "They are NOT claimed to be the authors' exact ten seeds."
)
print(
    "DIR is the primary scientific change from Run 008."
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
print("Fairness intervention: DisparateImpactRemover")
print("Repair level:", REPAIR_LEVEL)
print("Sensitive attribute:", SENSITIVE_ATTRIBUTE)
print("DIR applied to: training data only")
print("Test data repaired: False")
print("StandardScaler added: False")
print("Classifier: LogisticRegression(max_iter=1000)")
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("Seeds:", SEEDS)
print("Number of trials:", len(SEEDS))
print()

print("DIR INTEGRITY")
print("-------------")

for _, row in results.iterrows():
    print(
        "Seed {seed:.0f} | "
        "changed cells={dir_changed_nonprotected_cells:.0f} | "
        "changed features={dir_changed_nonprotected_features:.0f} | "
        "features={dir_changed_feature_names}".format(**row)
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
        "PED={predictive_equality_difference:.12f}".format(**row)
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
        "{ci95_upper:.12f}]".format(**row)
    )

print()
print("SOFTWARE")
print("--------")
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("aif360:", aif360.__version__)
