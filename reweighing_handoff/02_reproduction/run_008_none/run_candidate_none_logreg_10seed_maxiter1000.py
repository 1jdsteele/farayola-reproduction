from pathlib import Path
import argparse
import hashlib
import math

import numpy as np
import pandas as pd
import sklearn
import aif360

from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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
    raise ValueError("Tag may contain only letters, numbers, underscores, or hyphens.")

tag = args.tag


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

data = pd.read_csv(DATA_PATH)

assert data.shape == (5278, 12), f"Unexpected dataset shape: {data.shape}"

assert data["two_year_recid"].value_counts().to_dict() == {
    0: 2795,
    1: 2483,
}, "Unexpected target counts"

assert data["race"].value_counts().to_dict() == {
    0: 3175,
    1: 2103,
}, "Unexpected race counts"

sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# Predictors and target
# ---------------------------------------------------------------------

X = data.drop(columns=["two_year_recid"])
y = data["two_year_recid"]


# ---------------------------------------------------------------------
# Experimental protocol
#
# This extends Run 006.
#
# Held constant:
# - 70/30 train/test split
# - shuffle=True
# - stratify on race
# - LogisticRegression()
# - same AIF360 conventions
# - no fairness intervention
#
# Sensitivity variable:
# - random_state evaluated over predetermined seeds 101 through 110
#
# These are NOT claimed to be the authors' exact ten seeds.
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

favorable_label = 1
unfavorable_label = 0

protected_attribute_names = ["race"]

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

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # -------------------------------------------------------------
    # AIF360 datasets
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

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # -------------------------------------------------------------
    # Split composition
    # -------------------------------------------------------------

    train_df = X_train.copy()
    train_df["two_year_recid"] = y_train

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

        "accuracy": float(accuracy),
        "disparate_impact": float(metric.disparate_impact()),
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

results_path = METADATA_DIR / f"per_seed_results_{tag}.csv"
summary_path = METADATA_DIR / f"aggregate_summary_{tag}.csv"

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
print("RUN 007 — CANDIDATE NONE BASELINE — 10-SEED SENSITIVITY")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")
print(
    "Seeds 101-110 are a predetermined sensitivity set. "
    "They are NOT claimed to be the authors' exact ten seeds."
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
print("Classifier: LogisticRegression()")
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("Seeds:", SEEDS)
print("Number of trials:", len(SEEDS))
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
        "95% CI=[{ci95_lower:.12f}, {ci95_upper:.12f}]".format(**row)
    )

print()
print("SOFTWARE")
print("--------")
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("aif360:", aif360.__version__)
