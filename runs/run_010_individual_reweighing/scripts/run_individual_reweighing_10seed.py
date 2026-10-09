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

from aif360.algorithms.preprocessing import Reweighing
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
# - AIF360 Reweighing applied to TRAINING DATA ONLY.
# - resulting instance_weights are supplied to LogisticRegression
#   through sample_weight.
#
# Features and labels remain unchanged.
# Test data remain untouched.
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

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

    original_features = dataset_train_bld.features.copy()
    original_labels = dataset_train_bld.labels.copy()
    original_protected = dataset_train_bld.protected_attributes.copy()
    original_weights = dataset_train_bld.instance_weights.copy()

    # -------------------------------------------------------------
    # Apply individual Reweighing intervention
    # -------------------------------------------------------------

    reweigher = Reweighing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    dataset_train_reweighed = reweigher.fit_transform(
        dataset_train_bld
    )

    sample_weights = np.asarray(
        dataset_train_reweighed.instance_weights,
        dtype=float,
    )

    # -------------------------------------------------------------
    # Reweighing integrity checks
    # -------------------------------------------------------------

    assert len(sample_weights) == len(X_train_reset), (
        "Reweighing changed training row count"
    )

    assert np.array_equal(
        dataset_train_reweighed.features,
        original_features,
    ), "Reweighing unexpectedly changed feature values"

    assert np.array_equal(
        dataset_train_reweighed.labels,
        original_labels,
    ), "Reweighing unexpectedly changed labels"

    assert np.array_equal(
        dataset_train_reweighed.protected_attributes,
        original_protected,
    ), "Reweighing unexpectedly changed protected attributes"

    assert np.all(np.isfinite(sample_weights)), (
        "Reweighing produced non-finite instance weights"
    )

    assert np.all(sample_weights > 0), (
        "Reweighing produced non-positive instance weights"
    )

    assert not np.allclose(
        sample_weights,
        original_weights,
        rtol=0.0,
        atol=1e-12,
    ), "Reweighing did not modify instance weights"

    assert np.isclose(
        sample_weights.sum(),
        len(sample_weights),
        rtol=0.0,
        atol=1e-8,
    ), (
        "Unexpected total reweighted sample mass: "
        f"{sample_weights.sum()}"
    )

    # Test data must remain completely untouched.
    pd.testing.assert_frame_equal(
        X_test,
        X_test_snapshot,
    )

    # -------------------------------------------------------------
    # Record mean weights for the four race x target cells
    # -------------------------------------------------------------

    race_array = X_train_reset["race"].to_numpy()
    target_array = y_train_reset.to_numpy()

    weight_r0_y0 = sample_weights[
        (race_array == 0) & (target_array == 0)
    ].mean()

    weight_r0_y1 = sample_weights[
        (race_array == 0) & (target_array == 1)
    ].mean()

    weight_r1_y0 = sample_weights[
        (race_array == 1) & (target_array == 0)
    ].mean()

    weight_r1_y1 = sample_weights[
        (race_array == 1) & (target_array == 1)
    ].mean()

    # -------------------------------------------------------------
    # Classifier
    # -------------------------------------------------------------

    classifier = LogisticRegression(max_iter=1000)

    classifier.fit(
        X_train_reset,
        y_train_reset,
        sample_weight=sample_weights,
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

        "reweigh_features_unchanged": True,
        "reweigh_labels_unchanged": True,
        "reweigh_race_unchanged": True,
        "reweigh_test_untouched": True,

        "reweigh_weight_min": float(sample_weights.min()),
        "reweigh_weight_max": float(sample_weights.max()),
        "reweigh_weight_mean": float(sample_weights.mean()),
        "reweigh_weight_std": float(sample_weights.std(ddof=0)),
        "reweigh_weight_sum": float(sample_weights.sum()),

        "reweigh_race0_target0_weight": float(weight_r0_y0),
        "reweigh_race0_target1_weight": float(weight_r0_y1),
        "reweigh_race1_target0_weight": float(weight_r1_y0),
        "reweigh_race1_target1_weight": float(weight_r1_y1),

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
print("RUN 010 — INDIVIDUAL REWEIGHING — COMPAS")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")
print(
    "Seeds 101-110 are a predetermined diagnostic set. "
    "They are NOT claimed to be the authors' exact ten seeds."
)
print(
    "Reweighing is the primary scientific change from Run 008."
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
print("Fairness intervention: AIF360 Reweighing")
print("Protected attribute: race")
print("Applied to: training data only")
print("Weights passed to classifier: True")
print("Test data reweighted: False")
print("StandardScaler added: False")
print("Classifier: LogisticRegression(max_iter=1000)")
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("Seeds:", SEEDS)
print("Number of trials:", len(SEEDS))
print()

print("REWEIGHING INTEGRITY")
print("--------------------")

for _, row in results.iterrows():
    print(
        "Seed {seed:.0f} | "
        "min={reweigh_weight_min:.9f} | "
        "max={reweigh_weight_max:.9f} | "
        "mean={reweigh_weight_mean:.9f} | "
        "sum={reweigh_weight_sum:.6f} | "
        "r0y0={reweigh_race0_target0_weight:.9f} | "
        "r0y1={reweigh_race0_target1_weight:.9f} | "
        "r1y0={reweigh_race1_target0_weight:.9f} | "
        "r1y1={reweigh_race1_target1_weight:.9f}"
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

print()
print("SOFTWARE")
print("--------")
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("aif360:", aif360.__version__)
