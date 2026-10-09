from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tensorflow
import tensorflow.compat.v1 as tf
import aif360

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------------
# TensorFlow 1.x compatibility required by legacy AIF360 AL
# ---------------------------------------------------------------------

tf.disable_v2_behavior()

from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tag",
    required=True,
    help="Smoke-test execution tag, e.g. A or B.",
)
args = parser.parse_args()

tag = args.tag

if not tag.replace("_", "").replace("-", "").isalnum():
    raise ValueError(
        "Tag may contain only letters, numbers, underscores, or hyphens."
    )


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DATA_PATH = REPO_ROOT / "data" / "compas_farayola_numeric.csv"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

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
# Predictors / target
# ---------------------------------------------------------------------

X = data.drop(columns=["two_year_recid"])
y = data["two_year_recid"]

EXPECTED_FEATURES = [
    "age",
    "age_cat",
    "sex",
    "race",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
    "score_text",
    "decile_score",
]

assert list(X.columns) == EXPECTED_FEATURES, (
    f"Unexpected feature columns/order: {list(X.columns)}"
)


# ---------------------------------------------------------------------
# Controlled experimental protocol
#
# This is a diagnostic smoke test, NOT Run 026.
#
# Held consistent with Runs 008-025:
# - 70/30 train/test
# - random_state=101
# - shuffle=True
# - stratify on race
# - race retained as predictor
# - favorable_label=1
# - race 0 unprivileged
# - race 1 privileged
#
# Paper-reported AL parameters:
# - adversary_loss_weight=0.2
# - num_epochs=50
# - batch_size=128
# - debias=True
#
# classifier_num_hidden_units is intentionally NOT supplied.
# Installed AIF360 0.6.1 default = 200.
#
# No StandardScaler.
# No class balancing.
# No external np.random.seed().
# No tf.set_random_seed().
#
# We are intentionally testing whether AdversarialDebiasing(seed=101)
# is sufficient for exact A/B reproducibility.
# ---------------------------------------------------------------------

SEED = 101

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=SEED,
    shuffle=True,
    stratify=X["race"],
)

X_train_reset = X_train.reset_index(drop=True)
X_test_reset = X_test.reset_index(drop=True)

y_train_reset = y_train.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)


# ---------------------------------------------------------------------
# Preserve exact source data for integrity checking
# ---------------------------------------------------------------------

X_train_snapshot = X_train_reset.copy(deep=True)
X_test_snapshot = X_test_reset.copy(deep=True)

y_train_snapshot = y_train_reset.copy(deep=True)
y_test_snapshot = y_test_reset.copy(deep=True)


# ---------------------------------------------------------------------
# AIF360 datasets
# ---------------------------------------------------------------------

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

favorable_label = 1
unfavorable_label = 0

protected_attribute_names = ["race"]

privileged_group = [{"race": 1}]
unprivileged_group = [{"race": 0}]

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
train_protected_before = dataset_train_bld.protected_attributes.copy()


# ---------------------------------------------------------------------
# Legacy AdversarialDebiasing
#
# A clean graph + one fresh session is used for this execution.
# The A/B reproducibility test runs this entire script in separate
# Python processes.
# ---------------------------------------------------------------------

tf.reset_default_graph()

sess = tf.Session()

try:
    model = AdversarialDebiasing(
        privileged_groups=privileged_group,
        unprivileged_groups=unprivileged_group,
        scope_name="al_smoke_seed101",
        sess=sess,
        seed=SEED,
        adversary_loss_weight=0.2,
        num_epochs=50,
        batch_size=128,
        debias=True,
        # classifier_num_hidden_units intentionally omitted:
        # installed AIF360 0.6.1 default is 200.
    )

    model.fit(dataset_train_bld)

    predictions_bld = model.predict(dataset_test_bld)

finally:
    sess.close()


# ---------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------

pd.testing.assert_frame_equal(
    X_train_reset,
    X_train_snapshot,
)

pd.testing.assert_frame_equal(
    X_test_reset,
    X_test_snapshot,
)

pd.testing.assert_series_equal(
    y_train_reset,
    y_train_snapshot,
)

pd.testing.assert_series_equal(
    y_test_reset,
    y_test_snapshot,
)

assert np.array_equal(
    dataset_train_bld.features,
    train_features_before,
), "AL unexpectedly modified training features"

assert np.array_equal(
    dataset_train_bld.labels,
    train_labels_before,
), "AL unexpectedly modified training labels"

assert np.array_equal(
    dataset_train_bld.protected_attributes,
    train_protected_before,
), "AL unexpectedly modified protected attributes"


# ---------------------------------------------------------------------
# Prediction diagnostics
# ---------------------------------------------------------------------

y_pred = predictions_bld.labels.ravel().astype(int)
scores = predictions_bld.scores.ravel().astype(np.float64)

assert len(y_pred) == len(y_test_reset), (
    "Unexpected AL prediction count"
)

assert len(scores) == len(y_test_reset), (
    "Unexpected AL score count"
)

unique_labels = sorted(np.unique(y_pred).tolist())

assert set(unique_labels).issubset({0, 1}), (
    f"Unexpected prediction labels: {unique_labels}"
)

score_unique_count = int(np.unique(scores).size)

scores_are_hard_binary = bool(
    np.isin(scores, [0.0, 1.0]).all()
)

scores_differ_from_labels_count = int(
    np.count_nonzero(
        ~np.isclose(
            scores,
            y_pred.astype(np.float64),
            rtol=0.0,
            atol=0.0,
        )
    )
)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Reproducibility hashes
# ---------------------------------------------------------------------

def array_sha256(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


prediction_hash = array_sha256(
    y_pred.astype(np.int64)
)

score_hash = array_sha256(
    scores.astype(np.float64)
)

train_index_hash = array_sha256(
    X_train.index.to_numpy(dtype=np.int64)
)

test_index_hash = array_sha256(
    X_test.index.to_numpy(dtype=np.int64)
)


# ---------------------------------------------------------------------
# Split composition
# ---------------------------------------------------------------------

raw_test_df = X_test_reset.copy()
raw_test_df["two_year_recid"] = y_test_reset

test_cross = pd.crosstab(
    raw_test_df["race"],
    raw_test_df["two_year_recid"],
)


# ---------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------

result = {
    "diagnostic": "AL seed-101 COMPAS smoke test",

    "input_sha256": sha256,
    "input_shape": list(data.shape),

    "seed": SEED,
    "test_size_fraction": 0.3,
    "shuffle": True,
    "stratified_on": "race",

    "train_rows": int(len(X_train_reset)),
    "test_rows": int(len(X_test_reset)),

    "train_race_0": int((X_train_reset["race"] == 0).sum()),
    "train_race_1": int((X_train_reset["race"] == 1).sum()),
    "test_race_0": int((X_test_reset["race"] == 0).sum()),
    "test_race_1": int((X_test_reset["race"] == 1).sum()),

    "train_target_0": int((y_train_reset == 0).sum()),
    "train_target_1": int((y_train_reset == 1).sum()),
    "test_target_0": int((y_test_reset == 0).sum()),
    "test_target_1": int((y_test_reset == 1).sum()),

    "test_race0_target0": int(test_cross.loc[0, 0]),
    "test_race0_target1": int(test_cross.loc[0, 1]),
    "test_race1_target0": int(test_cross.loc[1, 0]),
    "test_race1_target1": int(test_cross.loc[1, 1]),

    "adversary_loss_weight": 0.2,
    "num_epochs": 50,
    "batch_size": 128,
    "classifier_num_hidden_units": 200,
    "classifier_num_hidden_units_source": "AIF360 0.6.1 library default",
    "debias": True,

    "standard_scaler_used": False,
    "class_balancing_added": False,
    "external_numpy_seed_added": False,
    "tensorflow_global_seed_added": False,

    "prediction_count": int(len(y_pred)),
    "prediction_unique_labels": unique_labels,
    "predicted_0": int((y_pred == 0).sum()),
    "predicted_1": int((y_pred == 1).sum()),

    "score_count": int(len(scores)),
    "score_min": float(np.min(scores)),
    "score_max": float(np.max(scores)),
    "score_mean": float(np.mean(scores)),
    "score_unique_count": score_unique_count,
    "scores_are_hard_binary": scores_are_hard_binary,
    "scores_differ_from_labels_count": scores_differ_from_labels_count,

    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),

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

    "prediction_sha256": prediction_hash,
    "score_sha256": score_hash,
    "train_index_sha256": train_index_hash,
    "test_index_sha256": test_index_hash,

    "training_features_unchanged": True,
    "training_labels_unchanged": True,
    "training_protected_attributes_unchanged": True,

    "python": sys.version,
    "architecture": platform.machine(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "tensorflow": tensorflow.__version__,
    "aif360": aif360.__version__,
}


# ---------------------------------------------------------------------
# Preserve result
# ---------------------------------------------------------------------

result_path = SCRIPT_DIR / f"result_{tag}.json"

with result_path.open("w") as f:
    json.dump(
        result,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")


# ---------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("AL SEED-101 COMPAS SMOKE TEST")
print("=" * 78)

print()
print("STATUS")
print("------")
print("This is a diagnostic smoke test. It is NOT Run 026.")

print()
print("INPUT")
print("-----")
print("File:", DATA_PATH)
print("SHA256:", sha256)
print("Shape:", data.shape)

print()
print("PROTOCOL")
print("--------")
print("Seed:", SEED)
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("StandardScaler: False")
print("Class balancing added: False")
print("External NumPy seed: False")
print("TensorFlow global seed: False")

print()
print("AL")
print("--")
print("adversary_loss_weight: 0.2")
print("num_epochs: 50")
print("batch_size: 128")
print("classifier_num_hidden_units: 200 [library default]")
print("debias: True")

print()
print("PREDICTIONS")
print("-----------")
print("Prediction count:", len(y_pred))
print("Unique labels:", unique_labels)
print("Predicted 0:", int((y_pred == 0).sum()))
print("Predicted 1:", int((y_pred == 1).sum()))

print()
print("SCORES")
print("------")
print("Min:", repr(float(np.min(scores))))
print("Max:", repr(float(np.max(scores))))
print("Mean:", repr(float(np.mean(scores))))
print("Unique count:", score_unique_count)
print("Hard 0/1 only:", scores_are_hard_binary)
print(
    "Scores differing exactly from labels:",
    scores_differ_from_labels_count,
)

print()
print("CONFUSION MATRIX")
print("----------------")
print("TN:", int(tn))
print("FP:", int(fp))
print("FN:", int(fn))
print("TP:", int(tp))

print()
print("METRICS")
print("-------")
print(f"Accuracy: {accuracy:.12f}")
print(f"DI: {metric.disparate_impact():.12f}")
print(
    "SPD: "
    f"{metric.statistical_parity_difference():.12f}"
)
print(
    "EOD: "
    f"{metric.equal_opportunity_difference():.12f}"
)
print(
    "PED/FPR difference: "
    f"{metric.false_positive_rate_difference():.12f}"
)

print()
print("REPRODUCIBILITY HASHES")
print("----------------------")
print("Predictions:", prediction_hash)
print("Scores:", score_hash)
print("Train indices:", train_index_hash)
print("Test indices:", test_index_hash)

print()
print("RESULT FILE")
print("-----------")
print(result_path)
