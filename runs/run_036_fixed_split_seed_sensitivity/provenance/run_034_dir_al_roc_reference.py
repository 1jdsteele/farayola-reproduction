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

from aif360.algorithms.preprocessing import DisparateImpactRemover
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

RUN028_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_028_dir_al_per_seed_results_A.csv"
)

RUN030_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_030_al_roc_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)


# ---------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------

SEEDS = list(range(101, 111))

DIR_REPAIR_LEVEL = 1.0

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

data = pd.read_csv(DATA_PATH)
run028 = pd.read_csv(RUN028_PATH)
run030 = pd.read_csv(RUN030_PATH)

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

NONPROTECTED = [
    column
    for column in X.columns
    if column != "race"
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
    # Stage 1: DIR
    # -------------------------------------------------------------

    dir_transformer = DisparateImpactRemover(
        repair_level=DIR_REPAIR_LEVEL,
        sensitive_attribute="race",
    )

    repaired_bld = dir_transformer.fit_transform(
        train_bld
    )

    repaired_df, _ = repaired_bld.convert_to_dataframe()

    X_train_repaired = repaired_df[
        X.columns
    ].copy()

    repaired_labels = (
        repaired_df["two_year_recid"]
        .astype(int)
        .to_numpy()
    )


    # -------------------------------------------------------------
    # DIR integrity / activity
    # -------------------------------------------------------------

    dir_labels_unchanged = np.array_equal(
        repaired_labels,
        y_train.astype(int).to_numpy(),
    )

    dir_race_unchanged = np.array_equal(
        X_train_repaired["race"].to_numpy(),
        X_train["race"].to_numpy(),
    )

    assert dir_labels_unchanged
    assert dir_race_unchanged


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

    changed_feature_names = ";".join(
        changed_features
    )

    assert changed_cells > 0
    assert len(changed_features) > 0


    # -------------------------------------------------------------
    # Stage 2: AL
    # -------------------------------------------------------------

    tf.reset_default_graph()

    sess = tf.Session()

    try:

        model = AdversarialDebiasing(
            unprivileged_groups=unprivileged_group,
            privileged_groups=privileged_group,
            scope_name=f"dir_al_roc_seed_{seed}",
            sess=sess,
            seed=seed,
            adversary_loss_weight=AL_ADVERSARY_LOSS_WEIGHT,
            num_epochs=AL_NUM_EPOCHS,
            batch_size=AL_BATCH_SIZE,
            debias=AL_DEBIAS,
        )

        model.fit(
            repaired_bld
        )

        pred_al_bld = model.predict(
            test_bld
        )

    finally:
        sess.close()


    # -------------------------------------------------------------
    # Pre-ROC DIR+AL outputs
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


    al_prediction_sha256 = array_sha256(
        al_labels,
        np.int64,
    )

    al_score_sha256 = array_sha256(
        al_scores,
        np.float64,
    )

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


    # -------------------------------------------------------------
    # Pre-ROC metrics
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
    # Critical Run 028 control
    # -------------------------------------------------------------

    r28 = run028.loc[
        run028["seed"] == seed
    ]

    assert len(r28) == 1

    r28 = r28.iloc[0]


    run028_dir_changed_cells_match = bool(
        changed_cells
        == int(r28["dir_changed_nonprotected_cells"])
    )

    run028_dir_changed_features_match = bool(
        len(changed_features)
        == int(r28["dir_changed_nonprotected_features"])
    )

    run028_dir_changed_names_match = bool(
        changed_feature_names
        == str(r28["dir_changed_feature_names"])
    )

    run028_prediction_hash_match = bool(
        al_prediction_sha256
        == str(r28["prediction_sha256"])
    )

    run028_score_hash_match = bool(
        al_score_sha256
        == str(r28["score_sha256"])
    )

    run028_cm_match = bool(
        int(r28["tn"]) == int(pre_tn)
        and int(r28["fp"]) == int(pre_fp)
        and int(r28["fn"]) == int(pre_fn)
        and int(r28["tp"]) == int(pre_tp)
    )


    run028_metric_match = True

    for name, value in pre_metrics.items():

        if not np.isclose(
            value,
            float(r28[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run028_metric_match = False


    run028_score_diagnostic_match = bool(
        np.isclose(
            al_score_min,
            float(r28["score_min"]),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            al_score_max,
            float(r28["score_max"]),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            al_score_mean,
            float(r28["score_mean"]),
            rtol=0.0,
            atol=1e-12,
        )
        and al_score_unique_count
            == int(r28["score_unique_count"])
    )


    assert run028_dir_changed_cells_match
    assert run028_dir_changed_features_match
    assert run028_dir_changed_names_match
    assert run028_prediction_hash_match
    assert run028_score_hash_match
    assert run028_cm_match
    assert run028_metric_match
    assert run028_score_diagnostic_match


    # -------------------------------------------------------------
    # Snapshot DIR+AL output before ROC
    # -------------------------------------------------------------

    al_labels_snapshot = (
        pred_al_bld.labels.copy()
    )

    al_scores_snapshot = (
        pred_al_bld.scores.copy()
    )


    # -------------------------------------------------------------
    # Stage 3: ROC
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
    # Confirm ROC did not mutate original DIR+AL output
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
    # Final ROC outputs
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


    roc_threshold = float(
        roc.classification_threshold
    )

    roc_margin = float(
        roc.ROC_margin
    )

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
    # Diagnostic comparison with Run 030 AL+ROC
    #
    # Equality is NOT expected.
    # -------------------------------------------------------------

    r30 = run030.loc[
        run030["seed"] == seed
    ]

    assert len(r30) == 1

    r30 = r30.iloc[0]


    run030_threshold_match = bool(
        np.isclose(
            roc_threshold,
            float(r30["roc_selected_classification_threshold"]),
            rtol=0.0,
            atol=1e-12,
        )
    )

    run030_margin_match = bool(
        np.isclose(
            roc_margin,
            float(r30["roc_selected_margin"]),
            rtol=0.0,
            atol=1e-12,
        )
    )

    run030_changed_count_match = bool(
        roc_changed_predictions
        == int(r30["roc_changed_predictions"])
    )

    run030_final_prediction_hash_match = bool(
        final_prediction_sha256
        == str(r30["final_prediction_sha256"])
    )

    run030_final_cm_match = bool(
        int(r30["tn"]) == int(tn)
        and int(r30["fp"]) == int(fp)
        and int(r30["fn"]) == int(fn)
        and int(r30["tp"]) == int(tp)
    )


    run030_final_metric_match = True

    for name, value in metrics.items():

        if not np.isclose(
            value,
            float(r30[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            run030_final_metric_match = False


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

            # DIR
            "dir_repair_level":
                DIR_REPAIR_LEVEL,

            "dir_labels_unchanged":
                dir_labels_unchanged,

            "dir_race_unchanged":
                dir_race_unchanged,

            "dir_changed_nonprotected_cells":
                changed_cells,

            "dir_changed_nonprotected_features":
                len(changed_features),

            "dir_changed_feature_names":
                changed_feature_names,

            # Run 028 DIR controls
            "run028_dir_changed_cells_match":
                run028_dir_changed_cells_match,

            "run028_dir_changed_features_match":
                run028_dir_changed_features_match,

            "run028_dir_changed_names_match":
                run028_dir_changed_names_match,

            # AL
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

            "fresh_graph_per_seed":
                True,

            "fresh_session_per_seed":
                True,

            # AL score diagnostics
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

            # Run 028 AL controls
            "run028_prediction_hash_match":
                run028_prediction_hash_match,

            "run028_score_hash_match":
                run028_score_hash_match,

            "run028_cm_match":
                run028_cm_match,

            "run028_metric_match":
                run028_metric_match,

            "run028_score_diagnostic_match":
                run028_score_diagnostic_match,

            # ROC
            "roc_metric_name":
                ROC_METRIC_NAME,

            "roc_selected_classification_threshold":
                roc_threshold,

            "roc_selected_margin":
                roc_margin,

            "roc_changed_predictions":
                roc_changed_predictions,

            "roc_test_calibration":
                True,

            "al_labels_preserved":
                al_labels_preserved,

            "al_scores_preserved":
                al_scores_preserved,

            "final_prediction_sha256":
                final_prediction_sha256,

            # Run 030 diagnostic comparison
            "run030_threshold_match":
                run030_threshold_match,

            "run030_margin_match":
                run030_margin_match,

            "run030_changed_count_match":
                run030_changed_count_match,

            "run030_final_prediction_hash_match":
                run030_final_prediction_hash_match,

            "run030_final_cm_match":
                run030_final_cm_match,

            "run030_final_metric_match":
                run030_final_metric_match,

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

            # Final metrics
            **metrics,
        }
    )


# ---------------------------------------------------------------------
# Results
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
print("RUN 034 — DIR + AL + REJECT OPTION CLASSIFICATION — COMPAS")
print("=" * 78)
print()

print("DIR / PRE-ROC RUN 028 CONTROLS")
print("------------------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "DIR_cells={dir_changed_nonprotected_cells:.0f} | "
        "DIR_features={dir_changed_nonprotected_features:.0f} | "
        "Run028_DIR={run028_dir_changed_cells_match} | "
        "pred_hash={run028_prediction_hash_match} | "
        "score_hash={run028_score_hash_match} | "
        "cm={run028_cm_match} | "
        "metrics={run028_metric_match}"
        .format(**row)
    )

print()
print("ROC / RUN 030 DIAGNOSTICS")
print("-------------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "threshold={roc_selected_classification_threshold:.6f} | "
        "margin={roc_selected_margin:.6f} | "
        "changed={roc_changed_predictions:.0f} | "
        "Run030_threshold={run030_threshold_match} | "
        "Run030_margin={run030_margin_match} | "
        "Run030_changed={run030_changed_count_match} | "
        "Run030_pred_hash={run030_final_prediction_hash_match} | "
        "Run030_metrics={run030_final_metric_match}"
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
