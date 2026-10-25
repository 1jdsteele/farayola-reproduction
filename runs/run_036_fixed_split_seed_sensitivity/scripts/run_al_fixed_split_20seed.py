from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import math

import numpy as np
import pandas as pd

from scipy.stats import t
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from aif360.algorithms.inprocessing import AdversarialDebiasing
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
# Paths / protocol
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

AL_ADVERSARY_LOSS_WEIGHT = 0.2
AL_NUM_EPOCHS = 50
AL_BATCH_SIZE = 128
AL_DEBIAS = True

# Not explicitly passed to AdversarialDebiasing.
# AIF360 0.6.1 default verified separately as 200.
AL_EFFECTIVE_HIDDEN_UNITS = 200
AL_HIDDEN_UNITS_EXPLICIT = False

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


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


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
# Fixed ordered data
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
# Run AL seeds
# ---------------------------------------------------------------------

rows = []

for algorithm_seed in ALGORITHM_SEEDS:

    # Fresh graph per seed is part of the established AL protocol.
    tf.reset_default_graph()

    # Fresh datasets prevent accidental mutation from carrying over.
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

    # Snapshot true labels/features for mutation checks.
    train_features_snapshot = train_bld.features.copy()
    train_labels_snapshot = train_bld.labels.copy()

    test_features_snapshot = test_bld.features.copy()
    test_labels_snapshot = test_bld.labels.copy()

    sess = tf.Session()

    try:
        model = AdversarialDebiasing(
            privileged_groups=privileged_group,
            unprivileged_groups=unprivileged_group,
            scope_name=f"run036_al_seed_{algorithm_seed}",
            sess=sess,
            seed=algorithm_seed,
            adversary_loss_weight=AL_ADVERSARY_LOSS_WEIGHT,
            num_epochs=AL_NUM_EPOCHS,
            batch_size=AL_BATCH_SIZE,
            debias=AL_DEBIAS,
        )

        model.fit(
            train_bld
        )

        pred_bld = model.predict(
            test_bld
        )

        labels = (
            pred_bld.labels
            .ravel()
            .astype(int)
        )

        scores = np.asarray(
            pred_bld.scores,
            dtype=np.float64,
        ).ravel()

        # -------------------------------------------------------------
        # Integrity
        # -------------------------------------------------------------

        assert np.array_equal(
            train_bld.features,
            train_features_snapshot,
        )

        assert np.array_equal(
            train_bld.labels,
            train_labels_snapshot,
        )

        assert np.array_equal(
            test_bld.features,
            test_features_snapshot,
        )

        assert np.array_equal(
            test_bld.labels,
            test_labels_snapshot,
        )

        assert len(labels) == len(y_test)

        assert set(
            np.unique(labels).tolist()
        ).issubset({0, 1})

        # -------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------

        metric = ClassificationMetric(
            test_bld,
            pred_bld,
            unprivileged_groups=unprivileged_group,
            privileged_groups=privileged_group,
        )

        accuracy = float(
            accuracy_score(
                y_test,
                labels,
            )
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            labels,
            labels=[0, 1],
        ).ravel()

        prediction_sha256 = array_sha256(
            labels,
            np.int64,
        )

        score_sha256 = array_sha256(
            scores,
            np.float64,
        )

        rows.append({
            "configuration": "AL",

            "split_seed": SPLIT_SEED,
            "algorithm_seed": algorithm_seed,

            "train_rows": len(train_ids),
            "test_rows": len(test_ids),

            "train_row_id_sha256": train_id_sha256,
            "test_row_id_sha256": test_id_sha256,

            "al_seed": algorithm_seed,
            "al_adversary_loss_weight":
                AL_ADVERSARY_LOSS_WEIGHT,
            "al_num_epochs":
                AL_NUM_EPOCHS,
            "al_batch_size":
                AL_BATCH_SIZE,
            "al_debias":
                AL_DEBIAS,
            "al_effective_hidden_units":
                AL_EFFECTIVE_HIDDEN_UNITS,
            "al_hidden_units_explicit":
                AL_HIDDEN_UNITS_EXPLICIT,

            "prediction_sha256":
                prediction_sha256,
            "score_sha256":
                score_sha256,

            "score_min":
                float(scores.min()),
            "score_max":
                float(scores.max()),
            "score_mean":
                float(scores.mean()),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),

            "accuracy":
                accuracy,

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
        })

    finally:
        sess.close()


results = pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Run 036 integrity
# ---------------------------------------------------------------------

assert len(results) == 20

assert results[
    "split_seed"
].nunique() == 1

assert int(
    results["split_seed"].iloc[0]
) == SPLIT_SEED

assert results[
    "train_row_id_sha256"
].nunique() == 1

assert results[
    "test_row_id_sha256"
].nunique() == 1

assert results[
    "al_adversary_loss_weight"
].nunique() == 1

assert float(
    results["al_adversary_loss_weight"].iloc[0]
) == 0.2

assert results[
    "al_num_epochs"
].nunique() == 1

assert results[
    "al_batch_size"
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

    mean = float(
        values.mean()
    )

    std = float(
        values.std(ddof=1)
    )

    margin = float(
        t_critical
        * std
        / math.sqrt(n)
    )

    summary_rows.append({
        "configuration": "AL",
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
    / f"al_per_seed_results_{tag}.csv"
)

summary_path = (
    METADATA_DIR
    / f"al_aggregate_summary_{tag}.csv"
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
print("RUN 036 — AL — FIXED-SPLIT RANDOM-SEED SENSITIVITY")
print("=" * 100)

print()
print("ENVIRONMENT")
print("-----------")
print("NumPy:", package_version("numpy"))
print("pandas:", package_version("pandas"))
print("scikit-learn:", package_version("scikit-learn"))
print("AIF360:", package_version("aif360"))
print("TensorFlow:", package_version("tensorflow"))
print(
    "tensorflow-macos:",
    package_version("tensorflow-macos"),
)

print()
print("PROTOCOL")
print("--------")
print("Configuration: AL")
print("Split seed:", SPLIT_SEED)
print("Algorithm seeds:", ALGORITHM_SEEDS)
print("Train rows:", len(train_ids))
print("Test rows:", len(test_ids))
print("Train row-ID SHA256:", train_id_sha256)
print("Test row-ID SHA256:", test_id_sha256)

print()
print("AL PARAMETERS")
print("-------------")
print(
    "adversary_loss_weight:",
    AL_ADVERSARY_LOSS_WEIGHT,
)
print(
    "num_epochs:",
    AL_NUM_EPOCHS,
)
print(
    "batch_size:",
    AL_BATCH_SIZE,
)
print(
    "debias:",
    AL_DEBIAS,
)
print(
    "effective hidden units:",
    AL_EFFECTIVE_HIDDEN_UNITS,
    "(AIF360 0.6.1 default; not explicitly passed)",
)

print()
print("SEED-CONTROL DIAGNOSTICS")
print("------------------------")
print(
    "Unique prediction hashes:",
    results["prediction_sha256"].nunique(),
)
print(
    "Unique score hashes:",
    results["score_sha256"].nunique(),
)

print()
print("PER-SEED RESULTS")
print("----------------")

for _, row in results.iterrows():
    print(
        "Algorithm seed {algorithm_seed:.0f} | "
        "Accuracy={accuracy:.12f} | "
        "DI={disparate_impact:.12f} | "
        "SPD={statistical_parity_difference:.12f} | "
        "EOD={equal_opportunity_difference:.12f} | "
        "PED={predictive_equality_difference:.12f} | "
        "CM=[{tn:.0f},{fp:.0f},{fn:.0f},{tp:.0f}]"
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
