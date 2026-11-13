"""Run 042 — Approach A: combine Reweighing weights with EGR oracle weights.

Does NOT externally weight Fairlearn's objective/fairness moments. Existing
AIF360 EGR is the control; a unity-weight oracle must be prediction-identical.
Run from any directory. Use --seeds 101 --tag SMOKE before --tag A.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from weighted_oracle import ExternalWeightOracleLR, features_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/compas_farayola_numeric.csv"
OUT = ROOT / "metadata"
P011 = ROOT / "provenance/run_011_egr_per_seed_results_A.csv"
P014 = ROOT / "provenance/run_014_re_egr_per_seed_results_A.csv"
EXPECTED_DATA_SHA256 = "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
METRICS = ["accuracy", "disparate_impact", "statistical_parity_difference",
           "equal_opportunity_difference", "predictive_equality_difference"]
CONSTRAINT = "DemographicParity"
EGR_EPS = 0.02
EGR_MAX_ITER = 20
BASE_LOGREG_MAX_ITER = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 111)))
    args = parser.parse_args()
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("Tag may contain only alphanumeric, underscore and hyphen")
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(set(range(101, 111))):
        parser.error("Seeds must be a unique subset of 101..110")
    return args


def check_versions():
    installed = {p: version(p) for p in ("aif360", "fairlearn", "scikit-learn", "pandas", "numpy")}
    for p, expected in (("aif360", "0.6.1"), ("fairlearn", "0.14.0"), ("scikit-learn", "1.9.1")):
        if installed[p] != expected:
            raise RuntimeError(f"Run 042 requires {p}=={expected}; found {installed[p]}. "
                               "Do not silently change the reference environment.")
    source = inspect.getsource(ExponentiatedGradientReduction.fit)
    if "instance_weights" in source:
        raise RuntimeError("Installed AIF360 wrapper appears to forward instance weights: audit again")
    # Fairlearn 0.14.0 uses these internal names; insist on the audited interface.
    import fairlearn.reductions._exponentiated_gradient._lagrangian as lag
    call_source = inspect.getsource(lag._Lagrangian._call_oracle)
    if "redW" not in call_source or "sample_weight_name" not in call_source:
        raise RuntimeError("Fairlearn oracle interface changed; audit before running")
    print("Pinned implementation versions checked:", installed)
    return installed


def binary_dataset(X, y):
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return BinaryLabelDataset(
        df=df,
        label_names=["two_year_recid"],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"],
        privileged_protected_attributes=[{"race": 1}],
    )


def evaluate(test_bld, y_true, y_pred):
    pred_bld = test_bld.copy(deepcopy=True)
    pred_bld.labels = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    metric = ClassificationMetric(
        test_bld, pred_bld,
        unprivileged_groups=[{"race": 0}],
        privileged_groups=[{"race": 1}],
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "disparate_impact": float(metric.disparate_impact()),
        "statistical_parity_difference": float(metric.statistical_parity_difference()),
        "equal_opportunity_difference": float(metric.equal_opportunity_difference()),
        "predictive_equality_difference": float(metric.false_positive_rate_difference()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def assert_matches_reference(measured, reference, arm_name, seed):
    for metric in METRICS:
        if not np.isclose(measured[metric], float(reference[metric]), rtol=0, atol=1e-10):
            raise AssertionError(
                f"Seed {seed}: {arm_name} {metric} differs from frozen run: "
                f"{measured[metric]} vs {reference[metric]}")
    for field in ("tn", "fp", "fn", "tp"):
        if measured[field] != int(reference[field]):
            raise AssertionError(f"Seed {seed}: {arm_name} {field} differs from frozen run")


def fit_predict(egr, dataset_train, dataset_test, seed):
    # Same seed protocol used in original Runs 011 and 014.
    np.random.seed(seed)
    egr.fit(dataset_train)
    np.random.seed(seed)
    return egr.predict(dataset_test).labels.ravel().astype(int)


def new_egr(estimator):
    return ExponentiatedGradientReduction(
        estimator,
        constraints=CONSTRAINT,
        eps=EGR_EPS,
        max_iter=EGR_MAX_ITER,
        drop_prot_attr=False,
    )


def check_oracle_trace(traces, reweighting_on, seed, n, egr):
    if not traces:
        raise AssertionError(f"Seed {seed}: custom oracle never fitted")
    underlying = egr.model.model_  # AIF360 sklearn wrapper -> fairlearn EG
    n_oracle = int(underlying.n_oracle_calls_)
    n_dummy = int(underlying.n_oracle_calls_dummy_returned_)
    if len(traces) != n_oracle - n_dummy:
        raise AssertionError(f"Seed {seed}: recorded {len(traces)} oracle LR fits, "
                             f"expected {n_oracle - n_dummy} (excluding {n_dummy} dummy calls)")
    for t in traces:
        if t["n_rows"] != n:
            raise AssertionError("Oracle trained on unexpected number of observations")
        if not np.isclose(t["internal_weight_mass"], t["combined_weight_mass"], rtol=0, atol=1e-8):
            raise AssertionError("Combined weights changed EGR oracle total weight mass")
        if reweighting_on and (not t["changed_from_internal"] or t["ratio_std"] <= 0):
            raise AssertionError("Reweighing didn't modify an oracle training weight vector")
        if not reweighting_on and t["changed_from_internal"]:
            raise AssertionError("Unity oracle unexpectedly modified internal weights")
    return n_oracle, n_dummy


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    versions = check_versions()
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != EXPECTED_DATA_SHA256:
        raise AssertionError("COMPAS numeric dataset SHA256 mismatch")
    data = pd.read_csv(DATA)
    if data.shape != (5278, 12):
        raise AssertionError(f"Unexpected dataset shape {data.shape}")
    if data["race"].value_counts().to_dict() != {0: 3175, 1: 2103}:
        raise AssertionError("Unexpected race counts")
    if data["two_year_recid"].value_counts().to_dict() != {0: 2795, 1: 2483}:
        raise AssertionError("Unexpected target counts")
    X = data.drop(columns=["two_year_recid"])
    y = data["two_year_recid"]
    reference011 = pd.read_csv(P011).set_index("seed")
    reference014 = pd.read_csv(P014).set_index("seed")
    if sorted(reference011.index.tolist()) != list(range(101, 111)):
        raise AssertionError("Frozen EGR baseline missing seed(s)")
    for seed in reference011.index:
        for m in METRICS:
            if not np.isclose(reference011.loc[seed, m], reference014.loc[seed, m], rtol=0, atol=1e-12):
                raise AssertionError("Frozen Run 011 and 014 differ; recheck original integration")

    results, predictions, traces_all = [], [], []
    for seed in args.seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, shuffle=True, stratify=X["race"])
        original_test_row_ids = X_te.index.to_numpy(copy=True)
        X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
        y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)
        train_bld = binary_dataset(X_tr, y_tr)
        test_bld = binary_dataset(X_te, y_te)

        reweigher = Reweighing(unprivileged_groups=[{"race": 0}], privileged_groups=[{"race": 1}])
        re_bld = reweigher.fit_transform(train_bld)
        if not (np.array_equal(re_bld.features, train_bld.features) and
                np.array_equal(re_bld.labels, train_bld.labels) and
                np.array_equal(re_bld.protected_attributes, train_bld.protected_attributes)):
            raise AssertionError("Reweighing changed data beyond instance weights")
        ext = np.asarray(re_bld.instance_weights, dtype=float).reshape(-1)
        if not (np.isfinite(ext).all() and (ext > 0).all() and
                np.isclose(ext.sum(), len(ext), rtol=0, atol=1e-8) and
                not np.allclose(ext, 1.0, atol=1e-12, rtol=0)):
            raise AssertionError("Invalid or unchanged Reweighing weights")
        ref014 = reference014.loc[seed]
        for field, actual in {"re_weight_min": ext.min(), "re_weight_max": ext.max(),
                              "re_weight_sum": ext.sum()}.items():
            if not np.isclose(float(ref014[field]), actual, rtol=0, atol=1e-9):
                raise AssertionError(f"Seed {seed}: Reweighing {field} differs from frozen Run 014")
        train_cols = pd.DataFrame(train_bld.features, columns=train_bld.feature_names)
        digest = features_fingerprint(train_cols)
        weight_tuple = tuple(ext.tolist())
        unity_tuple = (1.0,) * len(ext)

        # Baseline: original AIF360 implementation (also represents original Re+EGR
        # because AIF360 0.6.1 ignores dataset.instance_weights).
        legacy = new_egr(LogisticRegression(solver="lbfgs", max_iter=BASE_LOGREG_MAX_ITER))
        pred_legacy = fit_predict(legacy, train_bld, test_bld, seed)
        m_legacy = evaluate(test_bld, y_te.to_numpy(), pred_legacy)
        assert_matches_reference(m_legacy, reference011.loc[seed], "legacy EGR / Run 011", seed)
        assert_matches_reference(m_legacy, reference014.loc[seed], "original Re+EGR / Run 014", seed)

        # Exact no-op custom oracle (critical implementation control).
        ExternalWeightOracleLR.FIT_TRACES.clear()
        unity = new_egr(ExternalWeightOracleLR(unity_tuple, digest))
        pred_unity = fit_predict(unity, train_bld, test_bld, seed)
        unity_traces = [dict(z, seed=seed, arm="unity_oracle") for z in ExternalWeightOracleLR.FIT_TRACES]
        oracle_n_unity, dummy_unity = check_oracle_trace(unity_traces, False, seed, len(ext), unity)
        if not np.array_equal(pred_unity, pred_legacy):
            raise AssertionError(f"Seed {seed}: unity oracle differs from legacy EGR predictions")
        m_unity = evaluate(test_bld, y_te.to_numpy(), pred_unity)
        assert_matches_reference(m_unity, reference011.loc[seed], "unity oracle", seed)

        # Approach A: Re weights multiplied into EACH EGR internal LR oracle call.
        ExternalWeightOracleLR.FIT_TRACES.clear()
        weighted = new_egr(ExternalWeightOracleLR(weight_tuple, digest))
        pred_weighted = fit_predict(weighted, re_bld, test_bld, seed)
        weighted_traces = [dict(z, seed=seed, arm="reweighted_oracle") for z in ExternalWeightOracleLR.FIT_TRACES]
        oracle_n_weighted, dummy_weighted = check_oracle_trace(weighted_traces, True, seed, len(ext), weighted)
        m_weighted = evaluate(test_bld, y_te.to_numpy(), pred_weighted)
        changed = int(np.sum(pred_weighted != pred_legacy))

        for arm, m, pred, n_oracle, n_dummy, logs in (
            ("legacy_egr", m_legacy, pred_legacy, int(legacy.model.model_.n_oracle_calls_),
             int(legacy.model.model_.n_oracle_calls_dummy_returned_), []),
            ("unity_oracle", m_unity, pred_unity, oracle_n_unity, dummy_unity, unity_traces),
            ("reweighted_oracle", m_weighted, pred_weighted, oracle_n_weighted, dummy_weighted, weighted_traces),
        ):
            mixture = getattr({"legacy_egr": legacy, "unity_oracle": unity,
                               "reweighted_oracle": weighted}[arm].model.model_, "weights_")
            row = {
                "seed": seed, "arm": arm, "train_rows": len(y_tr), "test_rows": len(y_te),
                "train_test_stratify": "race", "constraint": CONSTRAINT,
                "eps": EGR_EPS, "egr_max_iter": EGR_MAX_ITER, "drop_prot_attr": False,
                "oracle_calls": n_oracle, "dummy_oracle_calls": n_dummy,
                "logged_non_dummy_oracles": len(logs),
                "mixture_predictors_nonzero": int(np.sum(np.asarray(mixture, dtype=float) > 0)),
                "mixture_weight_sum": float(np.asarray(mixture, dtype=float).sum()),
                "re_weight_min": float(ext.min()), "re_weight_max": float(ext.max()),
                "re_weight_std": float(ext.std()), "re_weight_sum": float(ext.sum()),
                "changed_predictions_vs_legacy": int(np.sum(pred != pred_legacy)),
                "frozen_reference_match": bool(arm != "reweighted_oracle"),
                **m,
            }
            results.append(row)
            predictions.append(pd.DataFrame({
                "seed": seed, "arm": arm, "test_original_row_id": original_test_row_ids,
                "race": X_te["race"].to_numpy(dtype=int),
                "true_label": y_te.to_numpy(dtype=int), "predicted_label": pred,
            }))
            traces_all.extend(logs)

        print(f"Seed {seed}: frozen EGR match=True, unity oracle exact match=True; "
              f"Re-weighted oracle fits={len(weighted_traces)}; "
              f"prediction changes={changed}/{len(y_te)}; "
              f"accuracy {m_legacy['accuracy']:.6f} → {m_weighted['accuracy']:.6f}; "
              f"SPD {m_legacy['statistical_parity_difference']:+.6f} → "
              f"{m_weighted['statistical_parity_difference']:+.6f}", flush=True)
    frame = pd.DataFrame(results)
    metric_summary = frame.groupby("arm")[METRICS + ["changed_predictions_vs_legacy", "oracle_calls"]].agg(["mean", "std"]).reset_index()
    metric_summary.columns = ["arm" if c[0] == "arm" else f"{c[0]}_{c[1]}" for c in metric_summary.columns]
    frame.to_csv(OUT / f"per_seed_results_{args.tag}.csv", index=False, float_format="%.15g")
    metric_summary.to_csv(OUT / f"aggregate_summary_{args.tag}.csv", index=False, float_format="%.15g")
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / f"prediction_level_{args.tag}.csv", index=False, float_format="%.15g")
    pd.DataFrame(traces_all).to_csv(OUT / f"oracle_fit_traces_{args.tag}.csv", index=False, float_format="%.15g")
    manifest = {
        "run": 42, "tag": args.tag, "seeds": args.seeds, "source": "reweighing_handoff.zip",
        "dataset_sha256": EXPECTED_DATA_SHA256,
        "method": "Approach A: per-oracle EGR sample_weight * AIF360 Re weights; mass-normalized",
        "scope": "Only EGR internal LogisticRegression oracle fits; EGR fairness moments and objective remain unweighted",
        "control_requirement": "unity oracle predictions exactly match legacy EGR and frozen Run 011",
        "training_split": "70/30 race-stratified train/test, test unseen in EGR fitting",
        "settings": {"constraint": CONSTRAINT, "eps": EGR_EPS, "max_iter": EGR_MAX_ITER,
                     "base_classifier": "LogisticRegression(solver=lbfgs, max_iter=1000)",
                     "drop_prot_attr": False},
        "versions": versions,
    }
    (OUT / f"run_manifest_{args.tag}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nRUN 042 COMPLETE: outputs in", OUT)
    print(frame.groupby("arm")[METRICS + ["changed_predictions_vs_legacy"]].mean().round(6).to_string())


if __name__ == "__main__":
    main()
