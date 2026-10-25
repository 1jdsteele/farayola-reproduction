from pathlib import Path
import argparse
import hashlib
import math

import numpy as np
import pandas as pd

from scipy.stats import t

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tensorflow
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
# Paths / constants
# ---------------------------------------------------------------------

REPO_ROOT = Path.cwd()
RUN_ROOT = REPO_ROOT / "runs" / "run_026_individual_al"

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

METADATA_DIR = REPO_ROOT / "analysis" / "advisor_discrepancy_audit" / "al_loss_weight_diagnostic" / "repo_default_01"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)

SEEDS = [101]

AL_ADVERSARY_LOSS_WEIGHT = 0.1
AL_NUM_EPOCHS = 50
AL_BATCH_SIZE = 128
AL_DEBIAS = True

# This value is recorded for provenance only.
# It is NOT explicitly passed to AdversarialDebiasing.
AL_HIDDEN_UNITS_LIBRARY_DEFAULT = 200


privileged_group = [
    {"race": 1}
]

unprivileged_group = [
    {"race": 0}
]


# ---------------------------------------------------------------------
# Smoke-test seed-101 reference
# ---------------------------------------------------------------------

SMOKE_101 = {
    "accuracy": 0.679924242424,
    "disparate_impact": 1.162281926820,
    "statistical_parity_difference": 0.071496633369,
    "equal_opportunity_difference": 0.032577591900,
    "predictive_equality_difference": 0.020140332640,
    "tn": 596,
    "fp": 285,
    "fn": 222,
    "tp": 481,
}

SMOKE_METRIC_TOLERANCE = 5e-12


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


rows = []


# ---------------------------------------------------------------------
# Seed loop
# ---------------------------------------------------------------------

for seed in SEEDS:

    # -------------------------------------------------------------
    # Controlled split
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
    # BinaryLabelDatasets
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
    # Fresh graph + Session for this seed
    # -------------------------------------------------------------

    tf.reset_default_graph()

    sess = tf.Session()

    try:

        model = AdversarialDebiasing(
            unprivileged_groups=unprivileged_group,
            privileged_groups=privileged_group,
            scope_name=f"al_seed_{seed}",
            sess=sess,
            seed=seed,
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

    finally:
        sess.close()


    # -------------------------------------------------------------
    # Predictions / scores
    # -------------------------------------------------------------

    labels = (
        pred_bld.labels
        .ravel()
        .astype(int)
    )

    scores = (
        pred_bld.scores
        .ravel()
        .astype(float)
    )

    assert len(labels) == len(y_test)
    assert len(scores) == len(y_test)

    assert set(
        np.unique(labels).tolist()
    ).issubset({0, 1})

    assert np.all(
        np.isfinite(scores)
    )

    score_min = float(
        scores.min()
    )

    score_max = float(
        scores.max()
    )

    score_mean = float(
        scores.mean()
    )

    score_unique_count = int(
        np.unique(scores).size
    )

    scores_are_hard_labels = bool(
        np.all(
            np.isin(
                scores,
                [0.0, 1.0],
            )
        )
    )

    score_label_difference_count = int(
        np.sum(
            ~np.isclose(
                scores,
                labels.astype(float),
                rtol=0.0,
                atol=1e-15,
            )
        )
    )

    prediction_sha256 = array_sha256(
        labels,
        np.int64,
    )

    score_sha256 = array_sha256(
        scores,
        np.float64,
    )


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

    cm = confusion_matrix(
        y_test,
        labels,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy,

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


    # -------------------------------------------------------------
    # Smoke seed-101 diagnostic comparison
    # -------------------------------------------------------------

    smoke_cm_match = None
    smoke_metric_match = None

    if seed == 101:

        smoke_cm_match = bool(
            int(tn) == SMOKE_101["tn"]
            and int(fp) == SMOKE_101["fp"]
            and int(fn) == SMOKE_101["fn"]
            and int(tp) == SMOKE_101["tp"]
        )

        smoke_metric_match = True

        for name, value in metrics.items():

            if not np.isclose(
                value,
                SMOKE_101[name],
                rtol=0.0,
                atol=SMOKE_METRIC_TOLERANCE,
            ):
                smoke_metric_match = False


    # -------------------------------------------------------------
    # Source-data integrity
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


    # -------------------------------------------------------------
    # Record
    # -------------------------------------------------------------

    rows.append(
        {
            "seed": seed,

            # Protocol
            "al_adversary_loss_weight":
                AL_ADVERSARY_LOSS_WEIGHT,

            "al_num_epochs":
                AL_NUM_EPOCHS,

            "al_batch_size":
                AL_BATCH_SIZE,

            "al_debias":
                AL_DEBIAS,

            "al_hidden_units_library_default":
                AL_HIDDEN_UNITS_LIBRARY_DEFAULT,

            "al_hidden_units_explicitly_passed":
                False,

            "external_numpy_seed_added":
                False,

            "external_tensorflow_seed_added":
                False,

            "fresh_graph_per_seed":
                True,

            "fresh_session_per_seed":
                True,

            # Scores
            "score_min":
                score_min,

            "score_max":
                score_max,

            "score_mean":
                score_mean,

            "score_unique_count":
                score_unique_count,

            "scores_are_hard_labels":
                scores_are_hard_labels,

            "score_label_difference_count":
                score_label_difference_count,

            "prediction_sha256":
                prediction_sha256,

            "score_sha256":
                score_sha256,

            # Smoke diagnostic
            "smoke101_cm_match":
                smoke_cm_match,

            "smoke101_metric_match":
                smoke_metric_match,

            # Integrity
            "train_source_untouched":
                True,

            "test_source_untouched":
                True,

            # Confusion matrix
            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),

            "predicted_0":
                int(np.sum(labels == 0)),

            "predicted_1":
                int(np.sum(labels == 1)),

            # Metrics
            **metrics,
        }
    )


# ---------------------------------------------------------------------
# Per-seed output
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Save
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
print("RUN 026 — INDIVIDUAL ADVERSARIAL DEBIASING (AL) — COMPAS")
print("=" * 78)
print()

print("ENVIRONMENT")
print("-----------")
print(f"TensorFlow: {tensorflow.__version__}")
print("TensorFlow eager execution:", tensorflow.executing_eagerly())
print()

print("PROTOCOL")
print("--------")
print(
    f"adversary_loss_weight={AL_ADVERSARY_LOSS_WEIGHT}, "
    f"num_epochs={AL_NUM_EPOCHS}, "
    f"batch_size={AL_BATCH_SIZE}, "
    f"debias={AL_DEBIAS}"
)
print(
    "classifier_num_hidden_units=200 "
    "(AIF360 library default; not explicitly passed)"
)
print()

print("PER-SEED DIAGNOSTICS")
print("--------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "score_range=[{score_min:.9f},{score_max:.9f}] | "
        "score_mean={score_mean:.9f} | "
        "unique_scores={score_unique_count:.0f} | "
        "hard_scores={scores_are_hard_labels} | "
        "score_label_diff={score_label_difference_count:.0f}"
        .format(**row)
    )

print()
print("SEED-101 SMOKE COMPARISON")
print("-------------------------")

seed101 = results.loc[
    results["seed"] == 101
].iloc[0]

print(
    "Confusion matrix match:",
    seed101["smoke101_cm_match"],
)

print(
    "Metric match:",
    seed101["smoke101_metric_match"],
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
        "95% CI=[{ci95_lower:.12f}, "
        "{ci95_upper:.12f}]"
        .format(**row)
    )
