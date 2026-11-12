"""Run 037: isolate the sample_weight fix in Reweighing + Equalized Odds.

This is a direct diagnostic of the repository-style test-set calibrated experiment,
NOT an unbiased generalization evaluation. The untouched Run 018 remains the control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import aif360
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "compas_farayola_numeric.csv"
PROV = ROOT / "provenance"
OUT = ROOT / "metadata"
SHA256 = "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
SEEDS = range(101, 111)
UNPRIVILEGED = [{"race": 0}]
PRIVILEGED = [{"race": 1}]
METRICS = [
    "accuracy", "disparate_impact", "statistical_parity_difference",
    "equal_opportunity_difference", "predictive_equality_difference",
]


def check_reference(reference: pd.DataFrame, seed: int, values: dict,
                    cm: np.ndarray, label: str) -> None:
    """Fail fast when the local environment cannot replicate frozen controls."""
    found = reference.loc[reference.seed == seed]
    assert len(found) == 1, f"{label}: expected one row for seed {seed}"
    row = found.iloc[0]
    cols = ("tn", "fp", "fn", "tp")
    for k, actual in zip(cols, cm):
        assert int(row[k]) == int(actual), (
            f"{label}, seed {seed}: {k} expected {int(row[k])}, got {int(actual)}. "
            "Check package versions and model feature preprocessing before proceeding."
        )
    for k in METRICS:
        assert np.isclose(float(row[k]), float(values[k]), rtol=0.0, atol=1e-12), (
            f"{label}, seed {seed}: {k} expected {row[k]}, got {values[k]}. "
            "Stop: this environment does not reproduce the frozen control."
        )


def values_for(true: BinaryLabelDataset, predictions: np.ndarray) -> tuple[dict, np.ndarray]:
    pred_bld = true.copy(deepcopy=True)
    pred_bld.labels = np.asarray(predictions, dtype=float).reshape(-1, 1)
    metrics = ClassificationMetric(
        true, pred_bld,
        unprivileged_groups=UNPRIVILEGED,
        privileged_groups=PRIVILEGED,
    )
    ytrue = true.labels.ravel().astype(int)
    ypred = pred_bld.labels.ravel().astype(int)
    cm = confusion_matrix(ytrue, ypred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(ytrue, ypred)),
        "disparate_impact": float(metrics.disparate_impact()),
        "statistical_parity_difference": float(metrics.statistical_parity_difference()),
        "equal_opportunity_difference": float(metrics.equal_opportunity_difference()),
        "predictive_equality_difference": float(metrics.false_positive_rate_difference()),
    }, cm


def pred_dataset(true: BinaryLabelDataset, ypred: np.ndarray) -> BinaryLabelDataset:
    pred = true.copy(deepcopy=True)
    pred.labels = np.asarray(ypred, dtype=float).reshape(-1, 1)
    return pred


def aggregate(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    critical = float(t.ppf(0.975, len(frame) - 1))
    for k in columns:
        v = frame[k].to_numpy(dtype=float)
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        margin = critical * sd / math.sqrt(len(v))
        rows.append({"metric": k, "n": len(v), "mean": mean,
                     "sample_std": sd, "min": float(v.min()), "max": float(v.max()),
                     "ci95_lower": mean - margin, "ci95_upper": mean + margin})
    return pd.DataFrame(rows)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="A", help="Output suffix, e.g. A or B")
    args = parser.parse_args()
    tag = args.tag
    if not tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid tag; use alphanumeric characters, _ or -")

    OUT.mkdir(parents=True, exist_ok=True)
    assert digest(DATA_PATH) == SHA256, "Input dataset has changed"
    data = pd.read_csv(DATA_PATH)
    assert data.shape == (5278, 12)
    assert data.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    assert data.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    X = data.drop(columns="two_year_recid")
    y = data.two_year_recid

    ref_paths = {
        "008": PROV / "run_008_none_per_seed_results_A.csv",
        "010": PROV / "run_010_reweighing_per_seed_results_A.csv",
        "012": PROV / "run_012_eo_per_seed_results_A.csv",
        "018": PROV / "run_018_re_eo_per_seed_results_A.csv",
    }
    refs = {name: pd.read_csv(path) for name, path in ref_paths.items()}
    for label, ref in refs.items():
        assert sorted(ref.seed.tolist()) == list(SEEDS), f"Unexpected seed list in {label}"

    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, shuffle=True, stratify=X.race
        )
        test_original_ids = X_test.index.to_numpy(copy=True)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        train_bld = BinaryLabelDataset(
            df=pd.concat([X_train, y_train], axis=1),
            label_names=["two_year_recid"], favorable_label=1, unfavorable_label=0,
            protected_attribute_names=["race"],
            privileged_protected_attributes=PRIVILEGED,
        )
        test_bld = BinaryLabelDataset(
            df=pd.concat([X_test, y_test], axis=1),
            label_names=["two_year_recid"], favorable_label=1, unfavorable_label=0,
            protected_attribute_names=["race"],
            privileged_protected_attributes=PRIVILEGED,
        )
        # Preserve the exact feature extraction of Run 018.
        assert train_bld.feature_names == test_bld.feature_names
        assert set(train_bld.feature_names) == set(X_train.columns)
        re_dataset = Reweighing(
            unprivileged_groups=UNPRIVILEGED,
            privileged_groups=PRIVILEGED,
        ).fit_transform(train_bld)
        weights = np.asarray(re_dataset.instance_weights, dtype=float).reshape(-1)
        initial_weights = np.asarray(train_bld.instance_weights, dtype=float).reshape(-1)

        assert len(weights) == len(X_train)
        assert np.all(np.isfinite(weights)) and np.all(weights > 0)
        assert not np.allclose(weights, initial_weights, atol=1e-12, rtol=0)
        assert np.isclose(weights.sum(), len(weights), atol=1e-8, rtol=0)
        assert np.array_equal(train_bld.features, re_dataset.features)
        assert np.array_equal(train_bld.labels, re_dataset.labels)
        assert np.array_equal(train_bld.protected_attributes, re_dataset.protected_attributes)
        assert np.array_equal(test_bld.labels.ravel().astype(int), y_test.to_numpy())

        ref010 = refs["010"].loc[refs["010"].seed == seed].iloc[0]
        for p, value in [
            ("reweigh_weight_min", weights.min()),
            ("reweigh_weight_max", weights.max()),
            ("reweigh_weight_sum", weights.sum()),
        ]:
            assert np.isclose(float(ref010[p]), float(value), atol=1e-10, rtol=0), (
                f"Seed {seed}: weight mismatch with Run 010 in {p}"
            )

        # Same classifier and feature arrays as Run 018, with the only
        # experimental difference being sample_weight in the second fit.
        base_lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        base_lr.fit(re_dataset.features, re_dataset.labels.ravel())
        weighted_lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        weighted_lr.fit(re_dataset.features, re_dataset.labels.ravel(), sample_weight=weights)

        base_pre = base_lr.predict(test_bld.features).astype(int)
        weighted_pre = weighted_lr.predict(test_bld.features).astype(int)
        base_values, base_cm = values_for(test_bld, base_pre)
        weighted_pre_values, weighted_pre_cm = values_for(test_bld, weighted_pre)
        check_reference(refs["008"], seed, base_values, base_cm, "Run 008 (pre-EO)")

        # Run 010 uses a pandas DataFrame rather than AIF360 NumPy arrays;
        # record its comparability, do not silently assert exact equivalence.
        run010_cm_equal = bool(np.array_equal(
            weighted_pre_cm, ref010[["tn", "fp", "fn", "tp"]].to_numpy(dtype=int)
        ))
        run010_metric_equal = bool(all(
            np.isclose(weighted_pre_values[k], float(ref010[k]), atol=1e-12, rtol=0)
            for k in METRICS
        ))

        # Repository-style test-set EO calibration, deliberately retained to
        # isolate weight propagation. NOT a generalization-valid test.
        control_eo = EqOddsPostprocessing(
            unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED, seed=seed
        )
        weighted_eo = EqOddsPostprocessing(
            unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED, seed=seed
        )
        base_pred_bld = pred_dataset(test_bld, base_pre)
        weighted_pred_bld = pred_dataset(test_bld, weighted_pre)
        base_snapshot = base_pred_bld.labels.copy()
        weighted_snapshot = weighted_pred_bld.labels.copy()

        base_post = control_eo.fit_predict(test_bld, base_pred_bld).labels.ravel().astype(int)
        weighted_post = weighted_eo.fit_predict(test_bld, weighted_pred_bld).labels.ravel().astype(int)
        assert np.array_equal(base_pred_bld.labels, base_snapshot)
        assert np.array_equal(weighted_pred_bld.labels, weighted_snapshot)
        assert len(base_post) == len(weighted_post) == len(y_test)
        assert set(np.unique(weighted_post).tolist()).issubset({0, 1})

        base_post_values, base_post_cm = values_for(test_bld, base_post)
        weighted_post_values, weighted_post_cm = values_for(test_bld, weighted_post)
        check_reference(refs["012"], seed, base_post_values, base_post_cm, "Run 012 EO")
        check_reference(refs["018"], seed, base_post_values, base_post_cm, "Run 018 Re+EO")

        row = {
            "seed": seed,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "re_weights_changed": True, "weight_min": float(weights.min()),
            "weight_max": float(weights.max()), "weight_std": float(weights.std()),
            "weight_sum": float(weights.sum()),
            "feature_names_match": True,
            "lr_used_instance_weights": True,
            "run008_pre_eo_match": True, "run012_post_eo_match": True,
            "run018_post_eo_match": True,
            "run010_weighted_pre_eo_cm_match": run010_cm_equal,
            "run010_weighted_pre_eo_metrics_match": run010_metric_equal,
            "base_lr_n_iter": int(base_lr.n_iter_[0]),
            "weighted_lr_n_iter": int(weighted_lr.n_iter_[0]),
            "coefficient_difference_l2": float(
                np.linalg.norm(base_lr.coef_ - weighted_lr.coef_)
            ),
            "base_vs_weighted_pre_eo_changed_predictions": int(np.sum(base_pre != weighted_pre)),
            "base_vs_weighted_post_eo_changed_predictions": int(np.sum(base_post != weighted_post)),
            "control_eo_changed_predictions": int(np.sum(base_pre != base_post)),
            "weighted_eo_changed_predictions": int(np.sum(weighted_pre != weighted_post)),
            "eo_test_calibration": True,
            "tn": int(weighted_post_cm[0]), "fp": int(weighted_post_cm[1]),
            "fn": int(weighted_post_cm[2]), "tp": int(weighted_post_cm[3]),
            "pre_eo_tn": int(weighted_pre_cm[0]), "pre_eo_fp": int(weighted_pre_cm[1]),
            "pre_eo_fn": int(weighted_pre_cm[2]), "pre_eo_tp": int(weighted_pre_cm[3]),
        }
        row.update(weighted_post_values)  # canonical final Run 037 metrics
        row.update({f"pre_eo_{k}": v for k, v in weighted_pre_values.items()})
        row.update({f"control_pre_eo_{k}": v for k, v in base_values.items()})
        row.update({f"control_post_eo_{k}": v for k, v in base_post_values.items()})
        row.update({f"delta_vs_018_{k}": weighted_post_values[k] - base_post_values[k]
                    for k in METRICS})
        row.update({f"delta_pre_eo_{k}": weighted_pre_values[k] - base_values[k]
                    for k in METRICS})
        result_rows.append(row)

        prediction_frames.append(pd.DataFrame({
            "seed": seed, "original_row_id": test_original_ids,
            "race": X_test.race.to_numpy(), "true_label": y_test.to_numpy(),
            "control_pre_eo": base_pre, "weighted_pre_eo": weighted_pre,
            "control_post_eo": base_post, "weighted_post_eo": weighted_post,
            "control_score": base_lr.predict_proba(test_bld.features)[:, 1],
            "weighted_score": weighted_lr.predict_proba(test_bld.features)[:, 1],
        }))
        print(f"Seed {seed}: weight-aware LR changed "
              f"{row['base_vs_weighted_pre_eo_changed_predictions']} / {len(y_test)} "
              "pre-EO predictions; final EO changed "
              f"{row['base_vs_weighted_post_eo_changed_predictions']} / {len(y_test)} "
              "relative to original Re+EO; "
              f"accuracy delta={row['delta_vs_018_accuracy']:+.6f}; "
              f"Run010 pre-EO match={run010_metric_equal}", flush=True)

    results = pd.DataFrame(result_rows)
    aggregate(results, METRICS).to_csv(
        OUT / f"aggregate_summary_{tag}.csv", index=False, float_format="%.15g")
    deltas = [f"delta_vs_018_{k}" for k in METRICS] + [
        f"delta_pre_eo_{k}" for k in METRICS]
    aggregate(results, deltas).to_csv(
        OUT / f"paired_deltas_{tag}.csv", index=False, float_format="%.15g")
    results.to_csv(OUT / f"per_seed_results_{tag}.csv", index=False, float_format="%.15g")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUT / f"prediction_level_{tag}.csv", index=False, float_format="%.15g")
    versions = {
        "experiment": "037_corrected_re_eo",
        "data_sha256": digest(DATA_PATH),
        "reference_sha256": {label: digest(path) for label, path in ref_paths.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__,
                     "aif360": getattr(aif360, "__version__", "unreported")},
        "seeds": list(SEEDS), "test_fraction": 0.3, "stratify": "race",
        "eo_seed": "experimental seed", "eo_calibration": "test (leakage; intentional)",
        "classifier": "LogisticRegression(solver='lbfgs', max_iter=1000)",
        "scientific_change": "pass sample_weight=re_dataset.instance_weights to LR.fit",
    }
    (OUT / f"run_manifest_{tag}.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    print("\nRUN 037 COMPLETE. Outputs in", OUT, flush=True)
    print(results[["seed", "base_vs_weighted_pre_eo_changed_predictions",
                   "base_vs_weighted_post_eo_changed_predictions",
                   "accuracy", "statistical_parity_difference", "disparate_impact"]].to_string(index=False))
    print("\nFinal aggregate:\n", aggregate(results, METRICS).to_string(index=False))
    print("\nPaired deltas (corrected - original):\n",
          aggregate(results, deltas).to_string(index=False))


if __name__ == "__main__":
    main()
