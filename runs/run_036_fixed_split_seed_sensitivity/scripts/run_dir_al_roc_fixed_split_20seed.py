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

# 36-D provides an exact upstream DIR+AL control.
RUN036_D_PATH = (
    RUN_ROOT
    / "metadata"
    / "dir_al_eo_per_seed_results_A.csv"
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

AL_ADVERSARY_LOSS_WEIGHT = 0.2
AL_NUM_EPOCHS = 50
AL_BATCH_SIZE = 128
AL_DEBIAS = True
AL_EFFECTIVE_HIDDEN_UNITS = 200

ROC_METRIC_NAME = "Equal opportunity difference"
ROC_LOW_CLASS_THRESH = 0.3
ROC_HIGH_CLASS_THRESH = 0.8
ROC_NUM_CLASS_THRESH = 100
ROC_NUM_MARGIN = 50
ROC_METRIC_UB = 0.05
ROC_METRIC_LB = -0.05

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


def compute_metrics(true_bld, pred_bld, y_true):
    labels = pred_bld.labels.ravel().astype(int)

    metric = ClassificationMetric(
        true_bld,
        pred_bld,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        labels,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, labels)
        ),
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

data = pd.read_csv(DATA_PATH)
manifest = pd.read_csv(MANIFEST_PATH)
run036_d = pd.read_csv(RUN036_D_PATH)

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
    manifest.loc[
        manifest["split"] == "train"
    ]
    .sort_values("split_order")
)

test_manifest = (
    manifest.loc[
        manifest["split"] == "test"
    ]
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

    tf.reset_default_graph()

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

    test_features_snapshot = (
        test_bld.features.copy()
    )

    test_labels_snapshot = (
        test_bld.labels.copy()
    )

    # -------------------------------------------------------------
    # Stage 1: DIR
    # -------------------------------------------------------------

    dir_transformer = DisparateImpactRemover(
        repair_level=DIR_REPAIR_LEVEL,
        sensitive_attribute=DIR_SENSITIVE_ATTRIBUTE,
    )

    repaired_bld = (
        dir_transformer.fit_transform(
            train_bld
        )
    )

    repaired_df, _ = (
        repaired_bld.convert_to_dataframe()
    )

    repaired_labels = (
        repaired_df["two_year_recid"]
        .astype(int)
        .to_numpy()
    )

    X_train_repaired = repaired_df[
        X_train.columns
    ].copy()

    assert np.array_equal(
        repaired_labels,
        y_train.astype(int).to_numpy(),
    )

    assert np.array_equal(
        X_train_repaired[
            "race"
        ].to_numpy(),
        X_train[
            "race"
        ].to_numpy(),
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

        feature_change_counts[
            column
        ] = int(changed.sum())

    changed_features = [
        column
        for column, count
        in feature_change_counts.items()
        if count > 0
    ]

    changed_cells = int(
        sum(
            feature_change_counts.values()
        )
    )

    assert changed_cells > 0

    repaired_feature_sha256 = (
        array_sha256(
            X_train_repaired.to_numpy(
                dtype=np.float64
            ),
            np.float64,
        )
    )

    # -------------------------------------------------------------
    # Stage 2: AL
    # -------------------------------------------------------------

    sess = tf.Session()

    try:

        model = AdversarialDebiasing(
            privileged_groups=privileged_group,
            unprivileged_groups=unprivileged_group,
            scope_name=(
                f"run036_dir_al_roc_seed_"
                f"{algorithm_seed}"
            ),
            sess=sess,
            seed=algorithm_seed,
            adversary_loss_weight=(
                AL_ADVERSARY_LOSS_WEIGHT
            ),
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

        y_pred_al = (
            pred_al_bld.labels
            .ravel()
            .astype(int)
        )

        al_scores = np.asarray(
            pred_al_bld.scores,
            dtype=np.float64,
        ).ravel()

        pre = compute_metrics(
            test_bld,
            pred_al_bld,
            y_test,
        )

        pre_prediction_sha256 = (
            array_sha256(
                y_pred_al,
                np.int64,
            )
        )

        pre_score_sha256 = (
            array_sha256(
                al_scores,
                np.float64,
            )
        )

        # ---------------------------------------------------------
        # Critical 36-D upstream-control check
        #
        # DIR+AL must be exactly identical before EO/ROC diverge.
        # ---------------------------------------------------------

        reference = run036_d.loc[
            run036_d[
                "algorithm_seed"
            ] == algorithm_seed
        ]

        assert len(reference) == 1

        reference = reference.iloc[0]

        assert (
            pre_prediction_sha256
            == reference[
                "pre_eo_prediction_sha256"
            ]
        )

        assert (
            pre_score_sha256
            == reference[
                "pre_eo_score_sha256"
            ]
        )

        metric_pairs = {
            "accuracy":
                "pre_eo_accuracy",
            "disparate_impact":
                "pre_eo_disparate_impact",
            "statistical_parity_difference":
                "pre_eo_statistical_parity_difference",
            "equal_opportunity_difference":
                "pre_eo_equal_opportunity_difference",
            "predictive_equality_difference":
                "pre_eo_predictive_equality_difference",
        }

        for current_name, reference_name in (
            metric_pairs.items()
        ):
            assert np.isclose(
                pre[current_name],
                float(
                    reference[
                        reference_name
                    ]
                ),
                rtol=0.0,
                atol=1e-12,
            )

        # ---------------------------------------------------------
        # Stage 3: ROC
        # ---------------------------------------------------------

        al_labels_snapshot = (
            pred_al_bld.labels.copy()
        )

        al_scores_snapshot = (
            pred_al_bld.scores.copy()
        )

        roc = RejectOptionClassification(
            unprivileged_groups=(
                unprivileged_group
            ),
            privileged_groups=(
                privileged_group
            ),
            metric_name=ROC_METRIC_NAME,
            low_class_thresh=(
                ROC_LOW_CLASS_THRESH
            ),
            high_class_thresh=(
                ROC_HIGH_CLASS_THRESH
            ),
            num_class_thresh=(
                ROC_NUM_CLASS_THRESH
            ),
            num_ROC_margin=(
                ROC_NUM_MARGIN
            ),
            metric_ub=ROC_METRIC_UB,
            metric_lb=ROC_METRIC_LB,
        )

        pred_roc_bld = roc.fit_predict(
            test_bld,
            pred_al_bld,
        )

        y_pred_roc = (
            pred_roc_bld.labels
            .ravel()
            .astype(int)
        )

        assert np.array_equal(
            pred_al_bld.labels,
            al_labels_snapshot,
        )

        assert np.array_equal(
            pred_al_bld.scores,
            al_scores_snapshot,
        )

        assert np.array_equal(
            test_bld.features,
            test_features_snapshot,
        )

        assert np.array_equal(
            test_bld.labels,
            test_labels_snapshot,
        )

        roc_changed_predictions = int(
            np.sum(
                y_pred_roc != y_pred_al
            )
        )

        final_scores = np.asarray(
            pred_roc_bld.scores,
            dtype=np.float64,
        ).ravel()

        final = compute_metrics(
            test_bld,
            pred_roc_bld,
            y_test,
        )

        final_prediction_sha256 = (
            array_sha256(
                y_pred_roc,
                np.int64,
            )
        )

        final_score_sha256 = (
            array_sha256(
                final_scores,
                np.float64,
            )
        )

        classification_threshold = float(
            roc.classification_threshold
        )

        roc_margin = float(
            roc.ROC_margin
        )

        threshold_at_low_boundary = bool(
            np.isclose(
                classification_threshold,
                ROC_LOW_CLASS_THRESH,
                rtol=0.0,
                atol=1e-12,
            )
        )

        threshold_at_high_boundary = bool(
            np.isclose(
                classification_threshold,
                ROC_HIGH_CLASS_THRESH,
                rtol=0.0,
                atol=1e-12,
            )
        )

        margin_is_zero = bool(
            np.isclose(
                roc_margin,
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        )

        # ---------------------------------------------------------
        # Store
        # ---------------------------------------------------------

        rows.append({
            "configuration":
                "DIR+AL+ROC",

            "split_seed":
                SPLIT_SEED,

            "algorithm_seed":
                algorithm_seed,

            "train_rows":
                len(train_ids),

            "test_rows":
                len(test_ids),

            "train_row_id_sha256":
                train_id_sha256,

            "test_row_id_sha256":
                test_id_sha256,

            # DIR
            "dir_repair_level":
                DIR_REPAIR_LEVEL,

            "dir_changed_nonprotected_cells":
                changed_cells,

            "dir_changed_nonprotected_features":
                len(changed_features),

            "dir_changed_feature_names":
                ";".join(changed_features),

            "dir_repaired_feature_sha256":
                repaired_feature_sha256,

            # AL
            "al_seed":
                algorithm_seed,

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

            # Pre-ROC DIR+AL
            "pre_roc_prediction_sha256":
                pre_prediction_sha256,

            "pre_roc_score_sha256":
                pre_score_sha256,

            "pre_roc_matches_36d":
                True,

            "pre_roc_tn":
                pre["tn"],

            "pre_roc_fp":
                pre["fp"],

            "pre_roc_fn":
                pre["fn"],

            "pre_roc_tp":
                pre["tp"],

            "pre_roc_accuracy":
                pre["accuracy"],

            "pre_roc_disparate_impact":
                pre["disparate_impact"],

            "pre_roc_statistical_parity_difference":
                pre[
                    "statistical_parity_difference"
                ],

            "pre_roc_equal_opportunity_difference":
                pre[
                    "equal_opportunity_difference"
                ],

            "pre_roc_predictive_equality_difference":
                pre[
                    "predictive_equality_difference"
                ],

            # ROC search protocol
            "roc_metric_name":
                ROC_METRIC_NAME,

            "roc_low_class_thresh":
                ROC_LOW_CLASS_THRESH,

            "roc_high_class_thresh":
                ROC_HIGH_CLASS_THRESH,

            "roc_num_class_thresh":
                ROC_NUM_CLASS_THRESH,

            "roc_num_margin":
                ROC_NUM_MARGIN,

            "roc_metric_ub":
                ROC_METRIC_UB,

            "roc_metric_lb":
                ROC_METRIC_LB,

            # ROC fitted solution
            "roc_classification_threshold":
                classification_threshold,

            "roc_margin":
                roc_margin,

            "roc_threshold_at_low_boundary":
                threshold_at_low_boundary,

            "roc_threshold_at_high_boundary":
                threshold_at_high_boundary,

            "roc_margin_is_zero":
                margin_is_zero,

            "roc_changed_predictions":
                roc_changed_predictions,

            # Final hashes / CM / outcomes
            "prediction_sha256":
                final_prediction_sha256,

            "score_sha256":
                final_score_sha256,

            "tn":
                final["tn"],

            "fp":
                final["fp"],

            "fn":
                final["fn"],

            "tp":
                final["tp"],

            "accuracy":
                final["accuracy"],

            "disparate_impact":
                final["disparate_impact"],

            "statistical_parity_difference":
                final[
                    "statistical_parity_difference"
                ],

            "equal_opportunity_difference":
                final[
                    "equal_opportunity_difference"
                ],

            "predictive_equality_difference":
                final[
                    "predictive_equality_difference"
                ],
        })

    finally:
        sess.close()


results = pd.DataFrame(
    rows
)


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

# DIR is deterministic.
assert results[
    "dir_repaired_feature_sha256"
].nunique() == 1

# Every pre-ROC DIR+AL realization must exactly match 36-D.
assert results[
    "pre_roc_matches_36d"
].all()


# ---------------------------------------------------------------------
# Aggregate summaries: pre-ROC and final
# ---------------------------------------------------------------------

metric_names = [
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

for stage, prefix in [
    ("DIR+AL pre-ROC", "pre_roc_"),
    ("DIR+AL+ROC final", ""),
]:

    for metric_name in metric_names:

        column = (
            f"{prefix}{metric_name}"
        )

        values = results[
            column
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
            "configuration":
                "DIR+AL+ROC",

            "stage":
                stage,

            "metric":
                metric_name,

            "n":
                n,

            "mean":
                mean,

            "sample_std":
                std,

            "min":
                float(values.min()),

            "max":
                float(values.max()),

            "ci95_lower":
                mean - margin,

            "ci95_upper":
                mean + margin,
        })

summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

results_path = (
    METADATA_DIR
    / f"dir_al_roc_per_seed_results_{tag}.csv"
)

summary_path = (
    METADATA_DIR
    / f"dir_al_roc_aggregate_summary_{tag}.csv"
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

print(
    "RUN 036-E — DIR + AL + ROC — "
    "FIXED-SPLIT RANDOM-SEED SENSITIVITY"
)

print("=" * 100)

print()
print("ENVIRONMENT")
print("-----------")
print(
    "NumPy:",
    package_version("numpy"),
)
print(
    "pandas:",
    package_version("pandas"),
)
print(
    "scikit-learn:",
    package_version("scikit-learn"),
)
print(
    "AIF360:",
    package_version("aif360"),
)
print(
    "TensorFlow:",
    package_version("tensorflow"),
)
print(
    "tensorflow-macos:",
    package_version("tensorflow-macos"),
)

print()
print("PROTOCOL")
print("--------")
print(
    "Split seed:",
    SPLIT_SEED,
)
print(
    "Algorithm seeds:",
    ALGORITHM_SEEDS,
)
print(
    "Train rows:",
    len(train_ids),
)
print(
    "Test rows:",
    len(test_ids),
)
print(
    "Train row-ID SHA256:",
    train_id_sha256,
)
print(
    "Test row-ID SHA256:",
    test_id_sha256,
)

print()
print("ROC PARAMETERS")
print("--------------")
print(
    "metric:",
    ROC_METRIC_NAME,
)
print(
    "classification threshold range:",
    ROC_LOW_CLASS_THRESH,
    "to",
    ROC_HIGH_CLASS_THRESH,
)
print(
    "threshold points:",
    ROC_NUM_CLASS_THRESH,
)
print(
    "ROC margins:",
    ROC_NUM_MARGIN,
)
print(
    "fairness bounds:",
    ROC_METRIC_LB,
    "to",
    ROC_METRIC_UB,
)

print()
print("SEED-CONTROL DIAGNOSTICS")
print("------------------------")
print(
    "Unique DIR repaired-feature hashes:",
    results[
        "dir_repaired_feature_sha256"
    ].nunique(),
)
print(
    "Unique pre-ROC AL prediction hashes:",
    results[
        "pre_roc_prediction_sha256"
    ].nunique(),
)
print(
    "Unique pre-ROC AL score hashes:",
    results[
        "pre_roc_score_sha256"
    ].nunique(),
)
print(
    "Pre-ROC matches 36-D:",
    bool(
        results[
            "pre_roc_matches_36d"
        ].all()
    ),
)
print(
    "Unique final prediction hashes:",
    results[
        "prediction_sha256"
    ].nunique(),
)
print(
    "Unique final score hashes:",
    results[
        "score_sha256"
    ].nunique(),
)

print()
print("ROC SOLUTION DIAGNOSTICS")
print("------------------------")
print(
    "Unique selected thresholds:",
    results[
        "roc_classification_threshold"
    ].nunique(),
)
print(
    "Unique selected margins:",
    results[
        "roc_margin"
    ].nunique(),
)
print(
    "Low-boundary threshold seeds:",
    int(
        results[
            "roc_threshold_at_low_boundary"
        ].sum()
    ),
    "/ 20",
)
print(
    "High-boundary threshold seeds:",
    int(
        results[
            "roc_threshold_at_high_boundary"
        ].sum()
    ),
    "/ 20",
)
print(
    "Zero-margin seeds:",
    int(
        results[
            "roc_margin_is_zero"
        ].sum()
    ),
    "/ 20",
)

print()
print("PER-SEED RESULTS")
print("----------------")

for _, row in results.iterrows():

    print(
        "Algorithm seed {algorithm_seed:.0f} | "
        "preEOD={pre_roc_equal_opportunity_difference:.6f} | "
        "threshold={roc_classification_threshold:.6f} | "
        "margin={roc_margin:.6f} | "
        "ROC_changed={roc_changed_predictions:.0f} | "
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

for stage in summary[
    "stage"
].unique():

    print()
    print(stage)

    subset = summary.loc[
        summary["stage"] == stage
    ]

    for _, row in subset.iterrows():

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
print("Saved:")
print(results_path)
print(summary_path)
