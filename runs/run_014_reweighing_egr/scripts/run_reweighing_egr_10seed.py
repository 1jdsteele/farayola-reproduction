from pathlib import Path
import argparse
import hashlib
import inspect
import math

import numpy as np
import pandas as pd

from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import (
    ExponentiatedGradientReduction,
)
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

DATA_PATH = (
    RUN_ROOT
    / "data"
    / "compas_farayola_numeric.csv"
)

RUN011_RESULTS_PATH = (
    RUN_ROOT
    / "provenance"
    / "run_011_parent_per_seed_results_A.csv"
)

METADATA_DIR = RUN_ROOT / "metadata"


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)

data = pd.read_csv(DATA_PATH)
run011_results = pd.read_csv(RUN011_RESULTS_PATH)

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

sha256 = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

assert sha256 == EXPECTED_SHA256, (
    f"Unexpected input SHA256: {sha256}"
)


# ---------------------------------------------------------------------
# Confirm installed AIF360 integration behavior
# ---------------------------------------------------------------------

egr_fit_source = inspect.getsource(
    ExponentiatedGradientReduction.fit
)

EGR_FIT_REFERENCES_INSTANCE_WEIGHTS = (
    "instance_weights" in egr_fit_source
)

assert not EGR_FIT_REFERENCES_INSTANCE_WEIGHTS, (
    "Installed AIF360 EGR fit() now appears to reference "
    "instance_weights. Re-evaluate Run 014 integration before proceeding."
)


# ---------------------------------------------------------------------
# Predictors and target
# ---------------------------------------------------------------------

X = data.drop(columns=["two_year_recid"])
y = data["two_year_recid"]

assert "race" in X.columns


# ---------------------------------------------------------------------
# Protocol
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

    X_test_snapshot = X_test_reset.copy(deep=True)
    y_test_snapshot = y_test_reset.copy(deep=True)

    # -------------------------------------------------------------
    # Build AIF360 datasets
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

    original_features = dataset_train_bld.features.copy()
    original_labels = dataset_train_bld.labels.copy()
    original_protected = (
        dataset_train_bld.protected_attributes.copy()
    )
    original_weights = (
        dataset_train_bld.instance_weights.copy()
    )

    # -------------------------------------------------------------
    # Stage 1: Reweighing
    # -------------------------------------------------------------

    reweigher = Reweighing(
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group,
    )

    re_dataset = reweigher.fit_transform(
        dataset_train_bld
    )

    re_weights = np.asarray(
        re_dataset.instance_weights,
        dtype=float,
    )

    # -------------------------------------------------------------
    # Reweighing integrity
    # -------------------------------------------------------------

    assert np.array_equal(
        re_dataset.features,
        original_features,
    ), "Reweighing unexpectedly changed features"

    assert np.array_equal(
        re_dataset.labels,
        original_labels,
    ), "Reweighing unexpectedly changed labels"

    assert np.array_equal(
        re_dataset.protected_attributes,
        original_protected,
    ), "Reweighing unexpectedly changed race"

    assert np.all(
        np.isfinite(re_weights)
    ), "Non-finite Reweighing weights"

    assert np.all(
        re_weights > 0
    ), "Non-positive Reweighing weights"

    assert not np.allclose(
        re_weights,
        original_weights,
        rtol=0.0,
        atol=1e-12,
    ), "Reweighing did not change instance weights"

    assert np.isclose(
        re_weights.sum(),
        len(re_weights),
        rtol=0.0,
        atol=1e-8,
    ), (
        "Unexpected total Reweighing sample mass: "
        f"{re_weights.sum()}"
    )

    # -------------------------------------------------------------
    # Record race x target Reweighing weights
    # -------------------------------------------------------------

    race_array = X_train_reset["race"].to_numpy()
    target_array = y_train_reset.to_numpy()

    weight_r0_y0 = re_weights[
        (race_array == 0)
        & (target_array == 0)
    ].mean()

    weight_r0_y1 = re_weights[
        (race_array == 0)
        & (target_array == 1)
    ].mean()

    weight_r1_y0 = re_weights[
        (race_array == 1)
        & (target_array == 0)
    ].mean()

    weight_r1_y1 = re_weights[
        (race_array == 1)
        & (target_array == 1)
    ].mean()

    # -------------------------------------------------------------
    # Stage 2: EGR
    #
    # IMPORTANT:
    # This deliberately mirrors the pinned repository:
    #
    # re_dataset = RW.fit_transform(dataset_train_bld)
    # EGR.fit(re_dataset)
    #
    # Installed AIF360 0.6.1's legacy EGR fit() does not forward
    # re_dataset.instance_weights to the underlying sklearn EGR.
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

    np.random.seed(seed)

    egr.fit(re_dataset)

    np.random.seed(seed)

    predictions_bld = egr.predict(
        dataset_test_bld
    )

    y_pred = (
        predictions_bld.labels
        .ravel()
        .astype(int)
    )

    # -------------------------------------------------------------
    # Test integrity
    # -------------------------------------------------------------

    pd.testing.assert_frame_equal(
        X_test_reset,
        X_test_snapshot,
    )

    pd.testing.assert_series_equal(
        y_test_reset,
        y_test_snapshot,
    )

    assert len(y_pred) == len(y_test_reset)

    assert set(
        np.unique(y_pred).tolist()
    ).issubset({0, 1})

    # -------------------------------------------------------------
    # Metrics
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

    metric_values = {
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

    # -------------------------------------------------------------
    # Compare with frozen Run 011 individual EGR
    #
    # If the reweighting weights are ignored by EGR, Run 014 should
    # reproduce Run 011 under identical seed controls.
    # -------------------------------------------------------------

    run011_row = run011_results.loc[
        run011_results["seed"] == seed
    ]

    assert len(run011_row) == 1

    run011_row = run011_row.iloc[0]

    run011_cm_match = (
        int(run011_row["tn"]) == int(tn)
        and int(run011_row["fp"]) == int(fp)
        and int(run011_row["fn"]) == int(fn)
        and int(run011_row["tp"]) == int(tp)
    )

    run011_metric_match = True

    for metric_name, value in metric_values.items():

        reference = float(
            run011_row[metric_name]
        )

        if not np.isclose(
            value,
            reference,
            rtol=0.0,
            atol=1e-12,
        ):
            run011_metric_match = False

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

        # Reweighing diagnostics
        "re_features_unchanged": True,
        "re_labels_unchanged": True,
        "re_race_unchanged": True,
        "re_weights_changed": True,

        "re_weight_min": float(
            re_weights.min()
        ),

        "re_weight_max": float(
            re_weights.max()
        ),

        "re_weight_mean": float(
            re_weights.mean()
        ),

        "re_weight_sum": float(
            re_weights.sum()
        ),

        "re_race0_target0_weight": float(
            weight_r0_y0
        ),

        "re_race0_target1_weight": float(
            weight_r0_y1
        ),

        "re_race1_target0_weight": float(
            weight_r1_y0
        ),

        "re_race1_target1_weight": float(
            weight_r1_y1
        ),

        # EGR configuration
        "egr_constraint": EGR_CONSTRAINT,
        "egr_eps": EGR_EPS,
        "egr_max_iter": EGR_MAX_ITER,
        "egr_drop_prot_attr": EGR_DROP_PROT_ATTR,
        "egr_prediction_seed": seed,
        "base_logreg_max_iter": BASE_LOGREG_MAX_ITER,

        # Critical integration diagnostic
        "egr_fit_references_instance_weights":
            EGR_FIT_REFERENCES_INSTANCE_WEIGHTS,

        "run011_confusion_matrix_match":
            run011_cm_match,

        "run011_metric_match":
            run011_metric_match,

        # Test integrity
        "test_untouched": True,

        # Outcomes
        **metric_values,
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
    minimum = values.min()
    maximum = values.max()

    standard_error = (
        std / math.sqrt(n)
    )

    margin = (
        t_critical * standard_error
    )

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

summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Outputs
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
# Human-readable stdout
# ---------------------------------------------------------------------

print("=" * 78)
print("RUN 014 — REWEIGHING + EGR (PI) — COMPAS")
print("=" * 78)
print()

print("IMPORTANT")
print("---------")
print(
    "This run reproduces the pinned repository structure:"
)
print(
    "Reweighing.fit_transform(training) -> EGR.fit(reweighted_dataset)"
)
print()

print(
    "Installed AIF360 EGR fit() references instance_weights:",
    EGR_FIT_REFERENCES_INSTANCE_WEIGHTS,
)

print(
    "Therefore Reweighing weights are computed but are not forwarded "
    "into EGR training by this legacy wrapper."
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
print("Stage 1: Reweighing")
print("Stage 2: ExponentiatedGradientReduction")
print("Base classifier: LogisticRegression(max_iter=1000)")
print("Constraint:", EGR_CONSTRAINT)
print("eps:", EGR_EPS)
print("EGR max_iter:", EGR_MAX_ITER)
print("drop_prot_attr:", EGR_DROP_PROT_ATTR)
print("Test size: 0.3")
print("Shuffle: True")
print("Stratified on: race")
print("Seeds:", SEEDS)
print()

print("INTEGRATION DIAGNOSTICS")
print("-----------------------")

for _, row in results.iterrows():

    print(
        "Seed {seed:.0f} | "
        "weights_changed={re_weights_changed} | "
        "weight_min={re_weight_min:.6f} | "
        "weight_max={re_weight_max:.6f} | "
        "EGR_uses_weights={egr_fit_references_instance_weights} | "
        "Run011_CM_match={run011_confusion_matrix_match} | "
        "Run011_metrics_match={run011_metric_match}"
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
