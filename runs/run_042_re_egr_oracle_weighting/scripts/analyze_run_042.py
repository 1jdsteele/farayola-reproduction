"""Analyze Run 042, validate prediction-level results and oracle training traces."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "metadata"
ARMS = ["legacy_egr", "unity_oracle", "reweighted_oracle"]
METRICS = ["accuracy", "statistical_parity_difference", "disparate_impact",
           "equal_opportunity_difference", "predictive_equality_difference"]


def markdown_table(columns, rows):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.6f}"
        return str(v)
    result = ["| " + " | ".join(columns) + " |",
              "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        result.append("| " + " | ".join(f(v) for v in r) + " |")
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    d = pd.read_csv(OUT / f"per_seed_results_{args.tag}.csv")
    pred = pd.read_csv(OUT / f"prediction_level_{args.tag}.csv")
    traces = pd.read_csv(OUT / f"oracle_fit_traces_{args.tag}.csv")
    if sorted(d.arm.unique().tolist()) != sorted(ARMS):
        raise AssertionError("Run 042 must contain all three arms")
    expected_seeds = sorted(d.seed.unique().tolist())
    if len(d) != 3 * len(expected_seeds) or len(d) != len(d[["seed", "arm"]].drop_duplicates()):
        raise AssertionError("Missing or duplicated seed/arm rows")
    if len(pred) != 3 * len(expected_seeds) * 1584:
        raise AssertionError("Unexpected prediction-level record count")

    # Cross-arm predictions on same source rows, and independently recalc accuracy.
    summary_checks = []
    for seed in expected_seeds:
        subset = pred[pred.seed == seed]
        by_arm = {arm: subset[subset.arm == arm].sort_values("test_original_row_id") for arm in ARMS}
        for arm in ARMS:
            g = by_arm[arm]
            row = d[(d.seed == seed) & (d.arm == arm)].iloc[0]
            if len(g) != 1584:
                raise AssertionError(f"Seed {seed} arm {arm} not 1584 observations")
            acc = np.mean(g.predicted_label.to_numpy() == g.true_label.to_numpy())
            if not np.isclose(acc, row.accuracy, atol=1e-12, rtol=0):
                raise AssertionError(f"Seed {seed} arm {arm}: accuracy mismatch")
            if not np.array_equal(g.test_original_row_id.to_numpy(), by_arm["legacy_egr"].test_original_row_id.to_numpy()):
                raise AssertionError("Observation mismatch across arms")
            if not np.array_equal(g.true_label.to_numpy(), by_arm["legacy_egr"].true_label.to_numpy()):
                raise AssertionError("Truth mismatch across arms")
        legacy = by_arm["legacy_egr"].predicted_label.to_numpy()
        unity = by_arm["unity_oracle"].predicted_label.to_numpy()
        weighted = by_arm["reweighted_oracle"].predicted_label.to_numpy()
        if not np.array_equal(legacy, unity):
            raise AssertionError(f"Seed {seed} unity oracle predictions not identical")
        changed = int(np.sum(legacy != weighted))
        roww = d[(d.seed == seed) & (d.arm == "reweighted_oracle")].iloc[0]
        if changed != int(roww.changed_predictions_vs_legacy):
            raise AssertionError(f"Seed {seed}: prediction difference count mismatch")
        for arm in ("unity_oracle", "reweighted_oracle"):
            row = d[(d.seed == seed) & (d.arm == arm)].iloc[0]
            ts = traces[(traces.seed == seed) & (traces.arm == arm)]
            if len(ts) != int(row.logged_non_dummy_oracles):
                raise AssertionError(f"Seed {seed}, {arm}: missing oracle traces")
            if len(ts) != int(row.oracle_calls - row.dummy_oracle_calls):
                raise AssertionError(f"Seed {seed}, {arm}: oracle fit trace mismatch")
            if not np.allclose(ts.internal_weight_mass, ts.combined_weight_mass, atol=1e-8, rtol=0):
                raise AssertionError(f"Seed {seed}, {arm}: unexpected weight mass")
            changed_weights = ts.changed_from_internal.astype(str).str.lower().eq("true")
            if arm == "unity_oracle" and changed_weights.any():
                raise AssertionError("Unity control changed oracle weights")
            if arm == "reweighted_oracle" and not changed_weights.all():
                raise AssertionError("Weighted oracle failed to combine weights")
        summary_checks.append({"seed": seed, "changed_predictions": changed,
                               "changed_fraction": changed / 1584})
    checks = pd.DataFrame(summary_checks)
    checks.to_csv(OUT / f"prediction_change_checks_{args.tag}.csv", index=False)

    mean = d.groupby("arm")[METRICS + ["changed_predictions_vs_legacy", "oracle_calls"]].mean().reindex(ARMS)
    control = d[d.arm == "legacy_egr"].set_index("seed").sort_index()
    weighted = d[d.arm == "reweighted_oracle"].set_index("seed").sort_index()
    paired = []
    for metric in METRICS:
        diff = (weighted[metric] - control[metric]).to_numpy(float)
        n = len(diff)
        err = (t.ppf(.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        paired.append({"metric": metric, "n": n, "mean_delta": diff.mean(),
                       "sample_std": diff.std(ddof=1) if n > 1 else np.nan,
                       "ci95_lower": diff.mean() - err, "ci95_upper": diff.mean() + err,
                       "seeds_changed": int(np.count_nonzero(np.abs(diff) > 1e-12))})
    delta_df = pd.DataFrame(paired)
    delta_df.to_csv(OUT / f"paired_deltas_{args.tag}.csv", index=False, float_format="%.15g")

    print("RUN 042 — PREDICTION-LEVEL AND ORACLE-TRACE VALIDATION PASSED")
    print("\nMEAN TEST RESULTS")
    print(mean.round(6).to_string())
    print("\nPAIRED DELTAS: Re-weighted oracle minus legacy EGR")
    print(delta_df.round(6).to_string(index=False))
    print("\nPrediction changes vs. EGR baseline", checks.changed_predictions.mean(),
          "of 1584 per seed; seeds with changes:", int((checks.changed_predictions > 0).sum()),
          "/", len(checks))

    report = [
        "# Run 042 — Reweighing-integrated EGR oracle (Approach A)", "",
        "**Protocol:** COMPAS 5,278 observations; 70/30 race-stratified train/test; "
        f"seeds {', '.join(map(str, expected_seeds))}. EGR: DemographicParity, "
        "eps=0.02, max_iter=20, drop_prot_attr=False, base LR max_iter=1000.", "",
        "**Intervention:** For each EGR cost-sensitive logistic-regression oracle call, "
        "multiply its internal sample weights by AIF360 Reweighing instance weights. "
        "Renormalize to the original internal oracle weight mass. Do NOT externally "
        "weight EGR's fairness-moment constraints or mixture-selection objective.", "",
        "**Implementation controls:** Original EGR reproduces frozen Run 011 and "
        "Run 014; unity external weights reproduce baseline predictions exactly "
        "for every tested seed; nontrivial external weights reach every "
        "logged logistic-regression oracle training call.", "",
        "## Mean test metrics", "",
        markdown_table(["Configuration", *METRICS], [[arm, *[mean.loc[arm, m] for m in METRICS]] for arm in ARMS]), "",
        "## Paired difference: weighted oracle − original EGR", "",
        markdown_table(["Metric", "Mean delta", "95% seed t lower", "95% seed t upper", "Seeds changed"],
                       [[r.metric, r.mean_delta, r.ci95_lower, r.ci95_upper, r.seeds_changed]
                        for r in delta_df.itertuples()]), "",
        "## Prediction and oracle diagnostics", "",
        f"- Mean changed predictions versus EGR: {checks.changed_predictions.mean():.2f} / 1584.",
        f"- Seeds with changed predictions: {(checks.changed_predictions > 0).sum()}/{len(checks)}.",
        f"- Mean oracle fits, unity: {mean.loc['unity_oracle', 'oracle_calls']:.2f}.",
        f"- Mean oracle fits, reweighted: {mean.loc['reweighted_oracle', 'oracle_calls']:.2f}.", "",
        "## Interpretation cautions", "",
        "- This is an **oracle-level integration experiment**, NOT a fully externally "
        "weighted EGR optimization. EGR's empirical error and demographic-parity "
        "moments are still evaluated without Reweighing weights.",
        "- The altered oracle may no longer solve Fairlearn's original best-response "
        "objective exactly; do not assume the original EGR theoretical convergence "
        "or fairness guarantees apply to this variant.",
        "- EGR uses a **Demographic Parity** constraint; eps=0.02 is an optimization "
        "parameter and should not be interpreted as guaranteeing a particular "
        "test SPD bound or statistical fairness across all metrics.",
        "- Test observations were not used for EGR fitting (held-out 30% test). "
        "Do not tune on these test results; future robustness tests should be separate runs.",
        "- Seed-based intervals describe variation over overlapping splits "
        "of this COMPAS cohort, not independent population sampling.",
        "- Predictions from EGR are randomized. The historical seed protocol "
        "sets NumPy before fit/predict; dedicated EGR random-seed stability remains for future work.", "",
    ]
    report_path = OUT / f"analysis_report_{args.tag}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\nRUN 042 ANALYSIS COMPLETE:", report_path)


if __name__ == "__main__":
    main()
