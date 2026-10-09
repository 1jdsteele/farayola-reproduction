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
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
)
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
# Paths
# ---------------------------------------------------------------------

RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

RUN026_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_026_al_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)


# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

AL_ADVERSARY_LOSS_WEIGHT = 0.2
AL_NUM_EPOCHS = 50
AL_BATCH_SIZE = 128
AL_DEBIAS = True

AL_HIDDEN_UNITS_LIBRARY_DEFAULT = 200

ROC_METRIC_NAME = "Equal opportunity difference"
ROC_LOW = 0.3
ROC_HIGH = 0.8
ROC_NUM_THRESH = 100
ROC_NUM_MARGIN = 50
ROC_UB = 0.05
ROC_LB = -0.05

privileged_group = [
    {"race": 1}
]

unprivileged_group = [
    {"race": 0}
]


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

data = pd.read_csv(
    DATA_PATH
)

run026 = pd.read_csv(
    RUN026_PATH
)

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
    # Stage 1: AL
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

        pred_al_bld = model.predict(
            test_bld
        )

    finally:
        sess.close()


    # -------------------------------------------------------------
    # Pre-ROC AL predictions / scores
    # -------------------------------------------------------------

    al_labels = (
        pred_al_bld.labels
        .ravel()
        .astype(int)
    )

    al_scores = (
        pred_al_bld.scores
        .ravel()
        .astype(float)
    )

    assert len(al_labels) == len(y_test)
    assert len(al_scores) == len(y_test)

    assert set(
        np.unique(al_labels).tolist()
    ).issubset({0, 1})

    assert np.all(
        np.isfinite(al_scores)
    )

    al_scores_are_hard_labels = bool(
        np.all(
            np.isin(
                al_scores,
                [0.0, 1.0],
            )
        )
    )

    assert not al_scores_are_hard_labels

    al_score_min = float(
        al_scores.min()
    )

    al_score_max = float(
        al_scores.max()
    )

    al_score_mean = float(
        al_scores.mean()
    )

    al_score_unique_count = int(
        np.unique(al_scores).size
    )

    al_prediction_sha256 = array_sha256(
        al_labels,
        np.int64,
    )

    al_score_sha256 = array_sha256(
        al_scores,
        np.float64,
    )


    # -------------------------------------------------------------
    # Pre-ROC AL metrics
    # -------------------------------------------------------------

    pre_metric = ClassificationMetric(
        test_bld,
        pred_al_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    pre_accuracy = float(
        accuracy_score(
            y_test,
            al_labels,
        )
    )

    pre_cm = confusion_matrix(
        y_test,
        al_labels,
        labels=[0, 1],
    )

    pre_tn, pre_fp, pre_fn, pre_tp = pre_cm.ravel()

    pre_metrics = {
        "accuracy":
            pre_accuracy,

        "disparate_impact":
            float(
                pre_metric.disparate_impact()
            ),

        "statistical_parity_difference":
            float(
                pre_metric.statistical_parity_difference()
            ),

        "equal_opportunity_difference":
            float(
                pre_metric.equal_opportunity_difference()
            ),

        "predictive_equality_difference":
            float(
                pre_metric.false_positive_rate_difference()
            ),
    }


    # -------------------------------------------------------------
    # Critical Run 026 control
    # -------------------------------------------------------------

    r26 = run026.loc[
        run026["seed"] == seed
    ]

    assert len(r26) == 1

    r26 = r26.iloc[0]


    run026_prediction_hash_match = bool(
        al_prediction_sha256
        == str(r26["prediction_sha256"])
    )

    run026_score_hash_match = bool(
        al_score_sha256
        == str(r26["score_sha256"])
    )

    run026_cm_match = bool(
        int(r26["tn"]) == int(pre_tn)
        and int(r26["fp"]) == int(pre_fp)
        and int(r26["fn"]) == int(pre_fn)
        and int(r26["tp"]) == int(pre_tp)
    )

    run026_metric_match = True

    for name, value in pre_metrics.items():

        if not np.isclose(
            value,
            float(r26[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run026_metric_match = False


    run026_score_diagnostic_match = bool(
        np.isclose(
            al_score_min,
            float(r26["score_min"]),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            al_score_max,
            float(r26["score_max"]),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            al_score_mean,
            float(r26["score_mean"]),
            rtol=0.0,
            atol=1e-12,
        )
        and al_score_unique_count
            == int(r26["score_unique_count"])
    )


    # Critical controls.
    assert run026_prediction_hash_match
    assert run026_score_hash_match
    assert run026_cm_match
    assert run026_metric_match
    assert run026_score_diagnostic_match


    # -------------------------------------------------------------
    # Snapshot AL output before ROC
    # -------------------------------------------------------------

    al_labels_snapshot = (
        pred_al_bld.labels.copy()
    )

    al_scores_snapshot = (
        pred_al_bld.scores.copy()
    )


    # -------------------------------------------------------------
    # Stage 2: ROC
    # -------------------------------------------------------------

    roc = RejectOptionClassification(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
        metric_name=ROC_METRIC_NAME,
        low_class_thresh=ROC_LOW,
        high_class_thresh=ROC_HIGH,
        num_class_thresh=ROC_NUM_THRESH,
        num_ROC_margin=ROC_NUM_MARGIN,
        metric_ub=ROC_UB,
        metric_lb=ROC_LB,
    )

    pred_roc_bld = roc.fit_predict(
        test_bld,
        pred_al_bld,
    )


    # -------------------------------------------------------------
    # Confirm original AL predictions were not mutated
    # -------------------------------------------------------------

    al_labels_preserved = np.array_equal(
        pred_al_bld.labels,
        al_labels_snapshot,
    )

    al_scores_preserved = np.array_equal(
        pred_al_bld.scores,
        al_scores_snapshot,
    )

    assert al_labels_preserved
    assert al_scores_preserved


    # -------------------------------------------------------------
    # Final ROC predictions
    # -------------------------------------------------------------

    final_labels = (
        pred_roc_bld.labels
        .ravel()
        .astype(int)
    )

    assert len(final_labels) == len(y_test)

    assert set(
        np.unique(final_labels).tolist()
    ).issubset({0, 1})


    roc_changed_predictions = int(
        np.sum(
            final_labels != al_labels
        )
    )

    final_prediction_sha256 = array_sha256(
        final_labels,
        np.int64,
    )


    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

    final_metric = ClassificationMetric(
        test_bld,
        pred_roc_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    accuracy = float(
        accuracy_score(
            y_test,
            final_labels,
        )
    )

    cm = confusion_matrix(
        y_test,
        final_labels,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":
            accuracy,

        "disparate_impact":
            float(
                final_metric.disparate_impact()
            ),

        "statistical_parity_difference":
            float(
                final_metric.statistical_parity_difference()
            ),

        "equal_opportunity_difference":
            float(
                final_metric.equal_opportunity_difference()
            ),

        "predictive_equality_difference":
            float(
                final_metric.false_positive_rate_difference()
            ),
    }


    # -------------------------------------------------------------
    # Source integrity
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
            "seed":
                seed,

            # AL protocol
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

            # AL scores
            "al_score_min":
                al_score_min,

            "al_score_max":
                al_score_max,

            "al_score_mean":
                al_score_mean,

            "al_score_unique_count":
                al_score_unique_count,

            "al_scores_are_hard_labels":
                al_scores_are_hard_labels,

            "al_prediction_sha256":
                al_prediction_sha256,

            "al_score_sha256":
                al_score_sha256,

            # Run 026 control
            "run026_prediction_hash_match":
                run026_prediction_hash_match,

            "run026_score_hash_match":
                run026_score_hash_match,

            "run026_cm_match":
                run026_cm_match,

            "run026_metric_match":
                run026_metric_match,

            "run026_score_diagnostic_match":
                run026_score_diagnostic_match,

            # ROC
            "roc_metric_name":
                ROC_METRIC_NAME,

            "roc_selected_classification_threshold":
                float(
                    roc.classification_threshold
                ),

            "roc_selected_margin":
                float(
                    roc.ROC_margin
                ),

            "roc_changed_predictions":
                roc_changed_predictions,

            "roc_test_calibration":
                True,

            # AL preservation
            "al_labels_preserved":
                al_labels_preserved,

            "al_scores_preserved":
                al_scores_preserved,

            # Final prediction
            "final_prediction_sha256":
                final_prediction_sha256,

            # Integrity
            "train_source_untouched":
                True,

            "test_source_untouched":
                True,

            # Final confusion matrix
            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),

            "predicted_0":
                int(np.sum(final_labels == 0)),

            "predicted_1":
                int(np.sum(final_labels == 1)),

            # Final metrics
            **metrics,
        }
    )


# ---------------------------------------------------------------------
# Per-seed results
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
# Save outputs
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
print("RUN 030 — AL + REJECT OPTION CLASSIFICATION (AL + ROC) — COMPAS")
print("=" * 78)
print()

print("PRE-ROC RUN 026 CONTROLS")
print("------------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "pred_hash={run026_prediction_hash_match} | "
        "score_hash={run026_score_hash_match} | "
        "cm={run026_cm_match} | "
        "metrics={run026_metric_match} | "
        "score_diag={run026_score_diagnostic_match}"
        .format(**row)
    )

print()
print("ROC DIAGNOSTICS")
print("---------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "AL_unique_scores={al_score_unique_count:.0f} | "
        "AL_hard_scores={al_scores_are_hard_labels} | "
        "threshold={roc_selected_classification_threshold:.6f} | "
        "margin={roc_selected_margin:.6f} | "
        "changed={roc_changed_predictions:.0f} | "
        "AL_labels_preserved={al_labels_preserved} | "
        "AL_scores_preserved={al_scores_preserved}"
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
