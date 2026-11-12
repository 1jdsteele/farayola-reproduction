#!/usr/bin/env python3
"""Run 039: reproducible EO mechanism diagnostics from frozen Run 038.

Pure analysis: reads Run 038 saved CSVs; never retrains models or writes to
Run 038. It verifies the two observed EO-flip counts against the archived
Run 038 per-seed data, and checks preexisting Run 039 CSVs if present.

Usage (from repository root):
    python runs/run_039_eo_mechanism/scripts/analyze_run_039.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "runs/run_038_heldout_re_eo/metadata"
DEST = ROOT / "runs/run_039_eo_mechanism/metadata"
DEST.mkdir(parents=True, exist_ok=True)

PRED_FILE = SOURCE / "prediction_level_A.csv"
RESULT_FILE = SOURCE / "per_seed_results_A.csv"
for path in (PRED_FILE, RESULT_FILE):
    if not path.is_file():
        raise FileNotFoundError(f"Run 038 input missing: {path}")

pred = pd.read_csv(PRED_FILE)
results = pd.read_csv(RESULT_FILE)
assert len(results) == 10 and set(results.seed) == set(range(101, 111)), "Unexpected seeds"
assert set(pred.split) >= {"calibration", "test"}, "Missing splits"
assert not pred.duplicated(["seed", "split", "original_row_id"]).any(), "Duplicate observation"


def save_or_verify(frame: pd.DataFrame, filename: str) -> None:
    """Verify a previous output if present; only create missing outputs."""
    path = DEST / filename
    if path.exists():
        old = pd.read_csv(path)
        assert_frame_equal(old.reset_index(drop=True), frame.reset_index(drop=True),
                           check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
        print(f"Verified existing {path.name}")
    else:
        frame.to_csv(path, index=False, float_format="%.15g")
        print(f"Created {path.name}")


# 1. Realized changes, by race and branch, on the untouched test split.
by_group = []
test = pred.loc[pred.split == "test"].copy()
for (seed, race), group in test.groupby(["seed", "race"], sort=True):
    truth = group.true_label
    for branch in ("control", "weighted"):
        before = group[f"{branch}_pre_eo"]
        after = group[f"{branch}_post_eo"]
        changed = before != after
        by_group.append({
            "seed": seed, "race": race, "branch": branch, "n": len(group),
            "flips": int(changed.sum()), "flip_rate_pct": 100 * changed.mean(),
            "zero_to_one": int(((before == 0) & (after == 1)).sum()),
            "one_to_zero": int(((before == 1) & (after == 0)).sum()),
            "harmful": int((changed & (before == truth)).sum()),
            "helpful": int((changed & (after == truth)).sum()),
        })
by_group = pd.DataFrame(by_group)
save_or_verify(by_group, "group_adjustments_A.csv")

# 2. AIF360 0.6.1 EO policy parameters, as recorded by Run 038.
policy_rows = []
for _, row in results.iterrows():
    for branch in ("control", "weighted"):
        for race, pos_idx, neg_idx in ((1, 0, 1), (0, 2, 3)):
            policy_rows.append({
                "seed": int(row.seed), "race": race, "branch": branch,
                "policy_prob_1_to_0": 1 - row[f"{branch}_eo_parameter_{pos_idx}"],
                "policy_prob_0_to_1": row[f"{branch}_eo_parameter_{neg_idx}"],
            })
policies = pd.DataFrame(policy_rows)
assert policies[["policy_prob_1_to_0", "policy_prob_0_to_1"]].ge(-1e-10).all().all()
assert policies[["policy_prob_1_to_0", "policy_prob_0_to_1"]].le(1+1e-10).all().all()
save_or_verify(policies, "eo_policy_probabilities_A.csv")

# 3. Analytic flip counts for every predictor/policy combination.
# int(...) per subgroup replicates AIF360's observed flip-count rounding.
cross_rows = []
for _, row in results.iterrows():
    seed = int(row.seed)
    data = test[test.seed == seed]
    for classifier in ("control", "weighted"):
        for policy in ("control", "weighted"):
            flips = 0
            for race, group in data.groupby("race"):
                before = group[f"{classifier}_pre_eo"]
                positives = int((before == 1).sum())
                negatives = int((before == 0).sum())
                i = 0 if int(race) == 1 else 2
                keep_positive = row[f"{policy}_eo_parameter_{i}"]
                promote_negative = row[f"{policy}_eo_parameter_{i+1}"]
                flips += int(positives * (1 - keep_positive))
                flips += int(negatives * promote_negative)
            cross_rows.append({"seed": seed, "classifier": classifier,
                               "eo_policy": policy, "flips": flips})
cross = pd.DataFrame(cross_rows)
for branch in ("control", "weighted"):
    estimated = cross[(cross.classifier == branch) & (cross.eo_policy == branch)].set_index("seed")["flips"]
    observed = results.set_index("seed")[f"test_{branch}_eo_changed_predictions"]
    assert estimated.sort_index().equals(observed.sort_index()), f"Actual-match failed: {branch}"
    print(f"Actual-match ({branch}): True, all 10 seeds")
save_or_verify(cross, "policy_prediction_cross_A.csv")

# 4. NEW: compute race-specific TPR / FPR on EO calibration data, before EO.
calibration_rows = []
cal = pred.loc[pred.split == "calibration"].copy()
for (seed, race), group in cal.groupby(["seed", "race"], sort=True):
    y = group.true_label
    pos = y == 1
    neg = y == 0
    assert pos.sum() > 0 and neg.sum() > 0
    for branch in ("control", "weighted"):
        p = group[f"{branch}_pre_eo"]
        calibration_rows.append({
            "seed": seed, "race": race, "branch": branch,
            "n_positive": int(pos.sum()), "n_negative": int(neg.sum()),
            "TPR": float(((p == 1) & pos).sum() / pos.sum()),
            "FPR": float(((p == 1) & neg).sum() / neg.sum()),
        })
calibration_rates = pd.DataFrame(calibration_rows)
save_or_verify(calibration_rates, "calibration_rates_by_group_A.csv")
wide = calibration_rates.pivot(index=["seed", "branch"], columns="race", values=["TPR", "FPR"])
calibration_gaps = pd.DataFrame({
    "absolute_TPR_gap": (wide["TPR"][0] - wide["TPR"][1]).abs(),
    "absolute_FPR_gap": (wide["FPR"][0] - wide["FPR"][1]).abs(),
}).reset_index()
save_or_verify(calibration_gaps, "calibration_gaps_A.csv")

# A final self-contained summary, with reproducibility references.
change_means = by_group.groupby("branch")[["flips", "harmful", "helpful"]].sum().div(10)
policy_mean = cross.groupby(["classifier", "eo_policy"]).flips.mean().unstack()
gap_mean = calibration_gaps.groupby("branch")[["absolute_TPR_gap", "absolute_FPR_gap"]].mean()
print("\nCalibration TPR/FPR absolute gaps (mean across 10 seeds):")
print(gap_mean.round(4).to_string())
print("\nTest flips and correctness changes (mean per seed):")
print(change_means.round(2).to_string())
print("\nClassifier × EO policy counterfactual flips (mean per seed):")
print(policy_mean.round(2).to_string())

assert np.isclose(change_means.loc["control", "flips"], 82.8)
assert np.isclose(change_means.loc["weighted", "flips"], 29.0)
assert np.isclose(gap_mean.loc["control", "absolute_TPR_gap"], .2214, atol=.0001)
assert np.isclose(gap_mean.loc["weighted", "absolute_TPR_gap"], .0355, atol=.0001)
assert np.isclose(gap_mean.loc["control", "absolute_FPR_gap"], .1605, atol=.0001)
assert np.isclose(gap_mean.loc["weighted", "absolute_FPR_gap"], .0428, atol=.0001)

hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (PRED_FILE, RESULT_FILE)}
summary = f"""# Run 039 — EO mechanism diagnostic (analysis of frozen Run 038)

Run 039 **does not retrain models or refit Equalized Odds**. It analyzes saved
Run 038 predictions and EO adjustment parameters from seeds 101–110.
Both branches use 70% training / 15% calibration / 15% held-out test data.

## Calibration disparities, **before** EO (mean absolute gap across seeds)

| Metric | Without Re | With Re |
|---|---:|---:|
| TPR gap | {gap_mean.loc['control', 'absolute_TPR_gap']:.4f} | {gap_mean.loc['weighted', 'absolute_TPR_gap']:.4f} |
| FPR gap | {gap_mean.loc['control', 'absolute_FPR_gap']:.4f} | {gap_mean.loc['weighted', 'absolute_FPR_gap']:.4f} |

## EO changes on held-out test observations (mean per seed, 792 test rows)

| Diagnostic | EO alone | Re + EO |
|---|---:|---:|
| Changed predictions | {change_means.loc['control', 'flips']:.1f} | {change_means.loc['weighted', 'flips']:.1f} |
| Correct → incorrect | {change_means.loc['control', 'harmful']:.1f} | {change_means.loc['weighted', 'harmful']:.1f} |
| Incorrect → correct | {change_means.loc['control', 'helpful']:.1f} | {change_means.loc['weighted', 'helpful']:.1f} |
| Net additional errors | {change_means.loc['control', 'harmful'] - change_means.loc['control', 'helpful']:.1f} | {change_means.loc['weighted', 'harmful'] - change_means.loc['weighted', 'helpful']:.1f} |

## Classifier / EO-policy substitution: calculated changed predictions

| Classifier predictions | Policy learned without Re | Policy learned with Re |
|---|---:|---:|
| Unweighted LR predictions | {policy_mean.loc['control', 'control']:.1f} (observed) | {policy_mean.loc['control', 'weighted']:.1f} (counterfactual) |
| Weighted LR predictions | {policy_mean.loc['weighted', 'control']:.1f} (counterfactual) | {policy_mean.loc['weighted', 'weighted']:.1f} (observed) |

Off-diagonal counts are *analytic counterfactual calculations* from
predictions and saved policy probabilities; they are **not** separately
executed pipelines or claims about their accuracy or fairness. Diagonal
counts are checked against the original Run 038 observed counts.

## Interpretation

Reweighing substantially reduced TPR/FPR disparities on EO's calibration
set. EO then learned a different adjustment policy, made fewer changes on
the held-out test set, and introduced fewer net errors. This is evidence
consistent with the smaller EO accuracy penalty in the weighted pipeline;
it does **not** establish that reducing calibration disparities is the sole
causal mechanism. The ten seeds share observations, and realized EO
predictions depend on randomization.

## Frozen Run 038 source SHA-256

- `prediction_level_A.csv`: `{hashes['prediction_level_A.csv']}`
- `per_seed_results_A.csv`: `{hashes['per_seed_results_A.csv']}`

To reproduce: `python runs/run_039_eo_mechanism/scripts/analyze_run_039.py`
"""
report_path = DEST / "analysis_report_A.md"
if report_path.exists():
    assert report_path.read_text() == summary, "Existing report differs; do not overwrite"
    print("Verified existing analysis_report_A.md")
else:
    report_path.write_text(summary)
    print("Created analysis_report_A.md")
print("\nRUN 039 DIAGNOSTIC REPRODUCTION PASSED")
