from pathlib import Path
import argparse
import hashlib
import math

import numpy as np
import pandas as pd

from scipy.stats import t

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import (
    Reweighing,
    DisparateImpactRemover,
)
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
)


# ---------------------------------------------------------------------
# CLI
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

RUN_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = RUN_ROOT / "data" / "compas_farayola_numeric.csv"

REF18_PATH = RUN_ROOT / "provenance" / "run018_per_seed_results_A.csv"
REF19_PATH = RUN_ROOT / "provenance" / "run019_per_seed_results_A.csv"
REF20_PATH = RUN_ROOT / "provenance" / "run020_per_seed_results_A.csv"
REF21_PATH = RUN_ROOT / "provenance" / "run021_per_seed_results_A.csv"

METADATA_DIR = RUN_ROOT / "metadata"

EXPECTED_SHA256 = (
    "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
)

SEEDS = list(range(101, 111))

PRIV = [{"race": 1}]
UNPRIV = [{"race": 0}]

ROC_ARGS = dict(
    unprivileged_groups=UNPRIV,
    privileged_groups=PRIV,
    metric_name="Equal opportunity difference",
    low_class_thresh=0.3,
    high_class_thresh=0.8,
    num_class_thresh=100,
    num_ROC_margin=50,
    metric_ub=0.05,
    metric_lb=-0.05,
)

METRIC_NAMES = [
    "accuracy",
    "disparate_impact",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "predictive_equality_difference",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def arr_hash(a, dtype=np.int64):
    x = np.ascontiguousarray(np.asarray(a, dtype=dtype))
    return hashlib.sha256(x.tobytes()).hexdigest()


def make_bld(X_df, y_values):
    df = X_df.reset_index(drop=True).copy()

    y_series = pd.Series(
        np.asarray(y_values).reshape(-1),
        name="two_year_recid",
    )

    df["two_year_recid"] = y_series.to_numpy()

    return BinaryLabelDataset(
        df=df,
        label_names=["two_year_recid"],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"],
        privileged_protected_attributes=PRIV,
    )


def make_pred_bld(X_df, labels):
    return make_bld(
        X_df,
        np.asarray(labels, dtype=int),
    )


def evaluate(true_bld, pred_bld, y_true):
    labels = pred_bld.labels.ravel().astype(int)

    cm_obj = ClassificationMetric(
        true_bld,
        pred_bld,
        unprivileged_groups=UNPRIV,
        privileged_groups=PRIV,
    )

    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).reshape(-1),
        labels,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(
                np.asarray(y_true).reshape(-1),
                labels,
            )
        ),
        "disparate_impact": float(
            cm_obj.disparate_impact()
        ),
        "statistical_parity_difference": float(
            cm_obj.statistical_parity_difference()
        ),
        "equal_opportunity_difference": float(
            cm_obj.equal_opportunity_difference()
        ),
        "predictive_equality_difference": float(
            cm_obj.false_positive_rate_difference()
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fit_lr(X_train, y_train, X_test, scaled):
    if scaled:
        scaler = StandardScaler()

        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)

    else:
        X_train_use = np.asarray(X_train, dtype=float)
        X_test_use = np.asarray(X_test, dtype=float)

    model = LogisticRegression(
        max_iter=1000,
    )

    model.fit(
        X_train_use,
        np.asarray(y_train).reshape(-1),
    )

    pred = model.predict(
        X_test_use
    ).astype(int)

    return pred, int(model.n_iter_[0])


def postprocess(X_test_df, y_test, parent_labels, seed):
    true_bld = make_bld(
        X_test_df,
        y_test,
    )

    # Fresh prediction BLD for EO.
    pred_eo_input = make_pred_bld(
        X_test_df,
        parent_labels,
    )

    eo = EqOddsPostprocessing(
        unprivileged_groups=UNPRIV,
        privileged_groups=PRIV,
        seed=seed,
    )

    eo_out = eo.fit_predict(
        true_bld,
        pred_eo_input,
    )

    eo_labels = (
        eo_out.labels
        .ravel()
        .astype(int)
    )

    # Fresh prediction BLD for ROC.
    pred_roc_input = make_pred_bld(
        X_test_df,
        parent_labels,
    )

    roc = RejectOptionClassification(
        **ROC_ARGS
    )

    roc_out = roc.fit_predict(
        true_bld,
        pred_roc_input,
    )

    roc_labels = (
        roc_out.labels
        .ravel()
        .astype(int)
    )

    return {
        "true_bld": true_bld,
        "eo_bld": eo_out,
        "roc_bld": roc_out,
        "eo_labels": eo_labels,
        "roc_labels": roc_labels,
        "eo_changed": int(
            np.sum(eo_labels != parent_labels)
        ),
        "roc_changed": int(
            np.sum(roc_labels != parent_labels)
        ),
        "roc_threshold": float(
            roc.classification_threshold
        ),
        "roc_margin": float(
            roc.ROC_margin
        ),
    }


def reference_metric_match(ref, seed, metrics):
    row = ref.loc[
        ref["seed"] == seed
    ]

    assert len(row) == 1
    row = row.iloc[0]

    for name in METRIC_NAMES:
        if not np.isclose(
            float(metrics[name]),
            float(row[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            return False

    return True


def metric_vector_match(a, b):
    return all(
        np.isclose(
            float(a[name]),
            float(b[name]),
            rtol=0.0,
            atol=1e-12,
        )
        for name in METRIC_NAMES
    )


# ---------------------------------------------------------------------
# Load and verify
# ---------------------------------------------------------------------

data = pd.read_csv(DATA_PATH)

assert data.shape == (5278, 12)

sha = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

assert sha == EXPECTED_SHA256

ref18 = pd.read_csv(REF18_PATH)
ref19 = pd.read_csv(REF19_PATH)
ref20 = pd.read_csv(REF20_PATH)
ref21 = pd.read_csv(REF21_PATH)

X = data.drop(
    columns=["two_year_recid"]
)

y = data["two_year_recid"]

feature_names = list(X.columns)
race_idx = feature_names.index("race")


result_rows = []
diag_rows = []


# ---------------------------------------------------------------------
# Seed loop
# ---------------------------------------------------------------------

for seed in SEEDS:

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

    train_bld = make_bld(
        X_train,
        y_train,
    )

    test_bld = make_bld(
        X_test,
        y_test,
    )


    # =============================================================
    # REWEIGHING
    # =============================================================

    rw = Reweighing(
        unprivileged_groups=UNPRIV,
        privileged_groups=PRIV,
    )

    re_bld = rw.fit_transform(
        train_bld
    )

    re_weights_changed = bool(
        not np.allclose(
            re_bld.instance_weights,
            train_bld.instance_weights,
            rtol=0.0,
            atol=1e-12,
        )
    )

    re_features_unchanged = bool(
        np.array_equal(
            re_bld.features,
            train_bld.features,
        )
    )

    re_labels_unchanged = bool(
        np.array_equal(
            re_bld.labels,
            train_bld.labels,
        )
    )

    assert re_weights_changed
    assert re_features_unchanged
    assert re_labels_unchanged


    # -------------------------------------------------------------
    # Re branch: unscaled control
    # -------------------------------------------------------------

    re_unscaled_parent, re_unscaled_iter = fit_lr(
        re_bld.features,
        re_bld.labels.ravel(),
        test_bld.features,
        scaled=False,
    )

    re_unscaled_pp = postprocess(
        X_test,
        y_test,
        re_unscaled_parent,
        seed,
    )

    re_unscaled_eo_metrics = evaluate(
        re_unscaled_pp["true_bld"],
        re_unscaled_pp["eo_bld"],
        y_test,
    )

    re_unscaled_roc_metrics = evaluate(
        re_unscaled_pp["true_bld"],
        re_unscaled_pp["roc_bld"],
        y_test,
    )

    run018_match = reference_metric_match(
        ref18,
        seed,
        re_unscaled_eo_metrics,
    )

    run019_match = reference_metric_match(
        ref19,
        seed,
        re_unscaled_roc_metrics,
    )

    assert run018_match
    assert run019_match


    # -------------------------------------------------------------
    # Re branch: scaled audit
    # -------------------------------------------------------------

    re_scaled_parent, re_scaled_iter = fit_lr(
        re_bld.features,
        re_bld.labels.ravel(),
        test_bld.features,
        scaled=True,
    )

    # Scaled baseline without Re.
    baseline_scaled_parent, baseline_scaled_iter = fit_lr(
        train_bld.features,
        train_bld.labels.ravel(),
        test_bld.features,
        scaled=True,
    )

    re_scaled_parent_equals_baseline = bool(
        np.array_equal(
            re_scaled_parent,
            baseline_scaled_parent,
        )
    )

    assert re_scaled_parent_equals_baseline


    re_scaled_pp = postprocess(
        X_test,
        y_test,
        re_scaled_parent,
        seed,
    )

    baseline_scaled_pp = postprocess(
        X_test,
        y_test,
        baseline_scaled_parent,
        seed,
    )


    re_scaled_eo_equals_baseline = bool(
        np.array_equal(
            re_scaled_pp["eo_labels"],
            baseline_scaled_pp["eo_labels"],
        )
    )

    re_scaled_roc_equals_baseline = bool(
        np.array_equal(
            re_scaled_pp["roc_labels"],
            baseline_scaled_pp["roc_labels"],
        )
    )

    assert re_scaled_eo_equals_baseline
    assert re_scaled_roc_equals_baseline


    re_scaled_eo_metrics = evaluate(
        re_scaled_pp["true_bld"],
        re_scaled_pp["eo_bld"],
        y_test,
    )

    re_scaled_roc_metrics = evaluate(
        re_scaled_pp["true_bld"],
        re_scaled_pp["roc_bld"],
        y_test,
    )


    # =============================================================
    # DIR
    # =============================================================

    dir_alg = DisparateImpactRemover(
        repair_level=1.0,
        sensitive_attribute="race",
    )

    dir_bld = dir_alg.fit_transform(
        train_bld
    )

    dir_labels_unchanged = bool(
        np.array_equal(
            dir_bld.labels,
            train_bld.labels,
        )
    )

    dir_race_unchanged = bool(
        np.array_equal(
            dir_bld.features[:, race_idx],
            train_bld.features[:, race_idx],
        )
    )

    assert dir_labels_unchanged
    assert dir_race_unchanged


    changed_counts = {}

    for idx, name in enumerate(feature_names):
        if name == "race":
            continue

        changed = ~np.isclose(
            dir_bld.features[:, idx],
            train_bld.features[:, idx],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

        changed_counts[name] = int(
            changed.sum()
        )


    dir_changed_features = [
        name
        for name, count in changed_counts.items()
        if count > 0
    ]

    dir_changed_cells = int(
        sum(changed_counts.values())
    )

    assert dir_changed_cells > 0
    assert len(dir_changed_features) > 0


    # -------------------------------------------------------------
    # DIR unscaled control
    # -------------------------------------------------------------

    dir_unscaled_parent, dir_unscaled_iter = fit_lr(
        dir_bld.features,
        dir_bld.labels.ravel(),
        test_bld.features,
        scaled=False,
    )

    dir_unscaled_pp = postprocess(
        X_test,
        y_test,
        dir_unscaled_parent,
        seed,
    )

    dir_unscaled_eo_metrics = evaluate(
        dir_unscaled_pp["true_bld"],
        dir_unscaled_pp["eo_bld"],
        y_test,
    )

    dir_unscaled_roc_metrics = evaluate(
        dir_unscaled_pp["true_bld"],
        dir_unscaled_pp["roc_bld"],
        y_test,
    )

    run020_match = reference_metric_match(
        ref20,
        seed,
        dir_unscaled_eo_metrics,
    )

    run021_match = reference_metric_match(
        ref21,
        seed,
        dir_unscaled_roc_metrics,
    )

    assert run020_match
    assert run021_match


    # -------------------------------------------------------------
    # DIR scaled audit
    # -------------------------------------------------------------

    dir_scaled_parent, dir_scaled_iter = fit_lr(
        dir_bld.features,
        dir_bld.labels.ravel(),
        test_bld.features,
        scaled=True,
    )

    dir_scaled_pp = postprocess(
        X_test,
        y_test,
        dir_scaled_parent,
        seed,
    )

    dir_scaled_eo_metrics = evaluate(
        dir_scaled_pp["true_bld"],
        dir_scaled_pp["eo_bld"],
        y_test,
    )

    dir_scaled_roc_metrics = evaluate(
        dir_scaled_pp["true_bld"],
        dir_scaled_pp["roc_bld"],
        y_test,
    )


    # =============================================================
    # Save four scaled audit configurations
    # =============================================================

    configs = [
        (
            "Re+EO_scaled",
            re_scaled_eo_metrics,
            re_scaled_pp["eo_labels"],
            re_scaled_parent,
            re_unscaled_pp["eo_labels"],
            re_scaled_pp["eo_changed"],
            np.nan,
            np.nan,
        ),
        (
            "Re+ROC_scaled",
            re_scaled_roc_metrics,
            re_scaled_pp["roc_labels"],
            re_scaled_parent,
            re_unscaled_pp["roc_labels"],
            re_scaled_pp["roc_changed"],
            re_scaled_pp["roc_threshold"],
            re_scaled_pp["roc_margin"],
        ),
        (
            "DIR+EO_scaled",
            dir_scaled_eo_metrics,
            dir_scaled_pp["eo_labels"],
            dir_scaled_parent,
            dir_unscaled_pp["eo_labels"],
            dir_scaled_pp["eo_changed"],
            np.nan,
            np.nan,
        ),
        (
            "DIR+ROC_scaled",
            dir_scaled_roc_metrics,
            dir_scaled_pp["roc_labels"],
            dir_scaled_parent,
            dir_unscaled_pp["roc_labels"],
            dir_scaled_pp["roc_changed"],
            dir_scaled_pp["roc_threshold"],
            dir_scaled_pp["roc_margin"],
        ),
    ]


    for (
        model_name,
        metric_dict,
        final_labels,
        parent_labels,
        old_final_labels,
        post_changed,
        roc_threshold,
        roc_margin,
    ) in configs:

        result_rows.append(
            {
                "model": model_name,
                "seed": seed,

                "parent_prediction_sha256":
                    arr_hash(parent_labels),

                "final_prediction_sha256":
                    arr_hash(final_labels),

                "scaled_vs_unscaled_final_changed_predictions":
                    int(
                        np.sum(
                            final_labels
                            != old_final_labels
                        )
                    ),

                "postprocessor_changed_predictions":
                    int(post_changed),

                "roc_selected_classification_threshold":
                    roc_threshold,

                "roc_selected_margin":
                    roc_margin,

                **{
                    k: metric_dict[k]
                    for k in METRIC_NAMES
                },

                "tn": metric_dict["tn"],
                "fp": metric_dict["fp"],
                "fn": metric_dict["fn"],
                "tp": metric_dict["tp"],
            }
        )


    # =============================================================
    # Diagnostics
    # =============================================================

    diag_rows.append(
        {
            "seed": seed,

            "run018_unscaled_reference_match":
                run018_match,

            "run019_unscaled_reference_match":
                run019_match,

            "run020_unscaled_reference_match":
                run020_match,

            "run021_unscaled_reference_match":
                run021_match,

            "re_weights_changed":
                re_weights_changed,

            "re_features_unchanged":
                re_features_unchanged,

            "re_labels_unchanged":
                re_labels_unchanged,

            "re_scaled_parent_equals_scaled_baseline":
                re_scaled_parent_equals_baseline,

            "re_scaled_eo_equals_scaled_baseline_eo":
                re_scaled_eo_equals_baseline,

            "re_scaled_roc_equals_scaled_baseline_roc":
                re_scaled_roc_equals_baseline,

            "re_parent_scaled_vs_unscaled_changes":
                int(
                    np.sum(
                        re_scaled_parent
                        != re_unscaled_parent
                    )
                ),

            "re_eo_scaled_vs_unscaled_changes":
                int(
                    np.sum(
                        re_scaled_pp["eo_labels"]
                        != re_unscaled_pp["eo_labels"]
                    )
                ),

            "re_roc_scaled_vs_unscaled_changes":
                int(
                    np.sum(
                        re_scaled_pp["roc_labels"]
                        != re_unscaled_pp["roc_labels"]
                    )
                ),

            "re_scaled_eo_changes_from_parent":
                re_scaled_pp["eo_changed"],

            "re_scaled_roc_changes_from_parent":
                re_scaled_pp["roc_changed"],

            "re_unscaled_lr_iterations":
                re_unscaled_iter,

            "re_scaled_lr_iterations":
                re_scaled_iter,

            "baseline_scaled_lr_iterations":
                baseline_scaled_iter,

            "dir_labels_unchanged":
                dir_labels_unchanged,

            "dir_race_unchanged":
                dir_race_unchanged,

            "dir_changed_nonprotected_cells":
                dir_changed_cells,

            "dir_changed_nonprotected_features":
                len(dir_changed_features),

            "dir_changed_feature_names":
                ";".join(dir_changed_features),

            "dir_parent_scaled_vs_unscaled_changes":
                int(
                    np.sum(
                        dir_scaled_parent
                        != dir_unscaled_parent
                    )
                ),

            "dir_eo_scaled_vs_unscaled_changes":
                int(
                    np.sum(
                        dir_scaled_pp["eo_labels"]
                        != dir_unscaled_pp["eo_labels"]
                    )
                ),

            "dir_roc_scaled_vs_unscaled_changes":
                int(
                    np.sum(
                        dir_scaled_pp["roc_labels"]
                        != dir_unscaled_pp["roc_labels"]
                    )
                ),

            "dir_scaled_eo_changes_from_parent":
                dir_scaled_pp["eo_changed"],

            "dir_scaled_roc_changes_from_parent":
                dir_scaled_pp["roc_changed"],

            "dir_unscaled_lr_iterations":
                dir_unscaled_iter,

            "dir_scaled_lr_iterations":
                dir_scaled_iter,
        }
    )


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------

results = pd.DataFrame(result_rows)
diagnostics = pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------

summary_rows = []

for model_name, group in results.groupby(
    "model",
    sort=False,
):

    n = len(group)

    tcrit = t.ppf(
        0.975,
        df=n - 1,
    )

    for metric_name in METRIC_NAMES:

        values = group[
            metric_name
        ].astype(float)

        mean = values.mean()
        std = values.std(ddof=1)

        margin = (
            tcrit
            * std
            / math.sqrt(n)
        )

        summary_rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "n": n,
                "mean": mean,
                "sample_std": std,
                "min": values.min(),
                "max": values.max(),
                "ci95_lower": mean - margin,
                "ci95_upper": mean + margin,
            }
        )


summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------

results.to_csv(
    METADATA_DIR / f"per_seed_results_{tag}.csv",
    index=False,
    float_format="%.15g",
)

diagnostics.to_csv(
    METADATA_DIR / f"audit_diagnostics_{tag}.csv",
    index=False,
    float_format="%.15g",
)

summary.to_csv(
    METADATA_DIR / f"aggregate_summary_{tag}.csv",
    index=False,
    float_format="%.15g",
)


# ---------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------

print("=" * 78)
print("RUN 035 — COMPAS PP STANDARDSCALER AUDIT")
print("=" * 78)

print()
print("UNSCALED REFERENCE REPRODUCTION")
print("-------------------------------")

for c in [
    "run018_unscaled_reference_match",
    "run019_unscaled_reference_match",
    "run020_unscaled_reference_match",
    "run021_unscaled_reference_match",
]:
    print(
        f"{c}:",
        bool(diagnostics[c].all()),
    )


print()
print("REWEIGHING INERTNESS WITH SCALING")
print("---------------------------------")

for c in [
    "re_weights_changed",
    "re_features_unchanged",
    "re_labels_unchanged",
    "re_scaled_parent_equals_scaled_baseline",
    "re_scaled_eo_equals_scaled_baseline_eo",
    "re_scaled_roc_equals_scaled_baseline_roc",
]:
    print(
        f"{c}:",
        bool(diagnostics[c].all()),
    )


print()
print("DIR ACTIVITY")
print("------------")

print(
    "dir_labels_unchanged:",
    bool(
        diagnostics[
            "dir_labels_unchanged"
        ].all()
    ),
)

print(
    "dir_race_unchanged:",
    bool(
        diagnostics[
            "dir_race_unchanged"
        ].all()
    ),
)

print(
    "DIR active every seed:",
    bool(
        (
            diagnostics[
                "dir_changed_nonprotected_cells"
            ] > 0
        ).all()
    ),
)


print()
print("SCALING EFFECT — CHANGED PREDICTIONS")
print("------------------------------------")

change_cols = [
    "re_parent_scaled_vs_unscaled_changes",
    "re_eo_scaled_vs_unscaled_changes",
    "re_roc_scaled_vs_unscaled_changes",
    "dir_parent_scaled_vs_unscaled_changes",
    "dir_eo_scaled_vs_unscaled_changes",
    "dir_roc_scaled_vs_unscaled_changes",
]

for c in change_cols:
    print(
        f"{c}: "
        f"min={int(diagnostics[c].min())}, "
        f"max={int(diagnostics[c].max())}, "
        f"mean={diagnostics[c].mean():.2f}"
    )


print()
print("ROC ACTIVITY AFTER SCALED HARD PREDICTIONS")
print("------------------------------------------")

print(
    "Re+ROC changed from parent:",
    diagnostics[
        "re_scaled_roc_changes_from_parent"
    ].tolist(),
)

print(
    "DIR+ROC changed from parent:",
    diagnostics[
        "dir_scaled_roc_changes_from_parent"
    ].tolist(),
)


print()
print("AGGREGATE SUMMARY")
print("-----------------")

for _, row in summary.iterrows():
    print(
        f"{row['model']} | "
        f"{row['metric']} | "
        f"mean={row['mean']:.12f} | "
        f"sd={row['sample_std']:.12f} | "
        f"range=[{row['min']:.12f}, "
        f"{row['max']:.12f}]"
    )
