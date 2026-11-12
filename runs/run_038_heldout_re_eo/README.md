# Run 038 — Held-out Reweighing + Equalized Odds (COMPAS)

## Research question

Does the weight-aware Reweighing correction from Run 037 change fairness and predictive performance **on held-out observations**, when EO's policy is learned on a separate calibration set?

**This is a new experimental protocol. Do not overwrite or relabel Runs 018, 037, or their output files.**

## Controlled comparison within Run 038

| Branch | Training | EO calibration | Held-out test |
|---|---|---|---|
| Control EO | Unweighted logistic regression | Fit EO to control predictions on calibration set | Apply learned control EO policy to test predictions |
| Corrected Re + EO | Reweigh training observations, pass `sample_weight` to logistic regression | Fit EO to weighted model predictions on *same* calibration observations | Apply learned weighted EO policy to test predictions on *same* test observations |

The single treatment difference between the branches is using Reweighing's instance weights during classifier training. EO is then re-fitted separately in each branch: this is an intended downstream consequence of the treatment, **not** another manually changed hyperparameter.

### Fixed design

- Input: provided 5,278-row, 11-feature COMPAS numeric dataset; exact SHA-256 checked.
- Seeds: 101–110 (10 runs). Each seed uses a different partition, with paired branches on identical data.
- **Outer split:** 70% training (3,694 rows), 30% holdout (1,584 rows), stratified by race, `random_state=seed`. This is the **same training subset/order as Run 037**.
- **Inner holdout split:** divide the outer holdout into EO calibration (792 rows) and final test (792 rows), `random_state=seed+10000`, stratified by the joint category `(race, two_year_recid)` to protect class/group coverage. This is deliberately different from Run 037's protocol.
- Race 0 unprivileged; race 1 privileged; positive/favorable label 1, following the reproduction (not a normative claim that recidivism is desirable).
- Classifier: `LogisticRegression(solver='lbfgs', max_iter=1000)`.
- EO: AIF360 `EqOddsPostprocessing(seed=seed)`. **`fit` only receives calibration truth + calibration model predictions; `predict` receives test model predictions without test truth.**
- Training subset never receives validation/test labels. Reweighing is fitted on training data only.
- No hyperparameter tuning, threshold selection, or choice between methods is performed on test data.

### Comparability and interpretation

Run 037's EO policy was fitted on its entire 30% evaluation set and evaluated on that same set. Run 038 uses separate 15% EO calibration and 15% test sets. **Do not take the numerical difference between the Run 037 and Run 038 metrics as a paired causal estimate.** The within-Run-038 paired EO versus corrected Re + EO difference is the comparison of interest.

The training-cohort identity is checked against frozen Run 008 pre-EO metrics and Run 010 Reweighing weights for every seed. If either check fails, **do not remove assertions**; diagnose environment/data mismatch first.

These are held-out estimates for this COMPAS cohort, not estimates of performance on a different population or external dataset. The 10 seeds reuse many observations; their t intervals are descriptive seed-to-seed summaries, not independent population confidence intervals. EO uses seeded randomized postprocessing; the realized EO fairness metrics need not be exactly zero.

## Included files

- `scripts/run_038_heldout_re_eo_10seed.py`: canonical 10-seed AIF360 experiment.
- `scripts/check_run_038_design.py`: split integrity check, can run without AIF360; **does not run the models**.
- `scripts/analyze_run_038.py`: validates generated result files and produces a Markdown comparison report.
- `data/compas_farayola_numeric.csv`: exact data file from prior handoff.
- `provenance/run_008_none_per_seed_results_A.csv` and `run_010_reweighing_per_seed_results_A.csv`: frozen reference tables.
- `metadata/`: generated results live here; initially contains no official results.

## Install at the correct path

From the `farayola-reproduction` repository root (the folder containing `runs/` and `.venv/`):

```bash
source .venv/bin/activate
unzip ~/Downloads/run_038_heldout_re_eo.zip -d runs/
```

This yields `runs/run_038_heldout_re_eo/scripts/`. **The `runs/` prefix is required** in all commands below.

## Execute on your Mac (same pinned `.venv` as Run 037)

```bash
python runs/run_038_heldout_re_eo/scripts/check_run_038_design.py
python runs/run_038_heldout_re_eo/scripts/run_038_heldout_re_eo_10seed.py --tag A
python runs/run_038_heldout_re_eo/scripts/analyze_run_038.py --tag A
```

The AIF360 step requires your thesis environment (AIF360 0.6.1, scikit-learn 1.9.1, etc.). Warnings about optional TensorFlow / inFairness algorithms are not relevant to Run 038. **An EO optimization failure or a frozen reference mismatch is a real blocker—do not simply skip it.**

Optional exact-repeat check (identical settings, separate outputs):

```bash
python runs/run_038_heldout_re_eo/scripts/run_038_heldout_re_eo_10seed.py --tag B
python runs/run_038_heldout_re_eo/scripts/analyze_run_038.py --tag B
```

### Saved output files

- `metadata/per_seed_results_A.csv`: 10 rows; train/calibration/test sample sizes; all pre- and post-EO metrics for both branches, all seed-level deltas, confusion counts, weight statistics, EO linear-program parameters.
- `metadata/aggregate_summary_A.csv`: after-EO test metrics by branch, across seeds.
- `metadata/paired_deltas_A.csv`: paired final test differences with descriptive 95% t intervals.
- `metadata/prediction_level_A.csv`: individual calibration and test predictions for both branches, before and after EO, including original row IDs.
- `metadata/split_membership_A.csv`: full per-seed train/calibration/test membership (original data row IDs).
- `metadata/group_label_counts_A.csv`: per-split race/label counts, to audit EO feasibility.
- `metadata/run_manifest_A.json`: versions, input hashes, script hash, protocol, random seeds.
- `metadata/analysis_report_A.md`: post-EO held-out comparison, upstream model comparison, prediction-change counts, absolute fairness distances, and calibration-vs-test descriptive accuracy.

Once finished, preserve the entire `runs/run_038_heldout_re_eo/` folder (including the generated metadata) in your Git workflow. Review files before pushing to a public GitHub repository.
