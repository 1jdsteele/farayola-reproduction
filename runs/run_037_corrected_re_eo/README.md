# Run 037 — Corrected Reweighing + Equalized Odds (COMPAS)

## Research question

Does *explicitly passing Reweighing's instance weights to logistic regression*
change the pre-EO predictions and the final EO outputs, relative to the original
Run 018 that generated but discarded the weights?

This experiment is a narrowly scoped **integration correction**, not a replication
of Farayola's reported paper table and **not** a valid out-of-sample EO evaluation.

## Controlled comparison

- Data: 5,278 observations, 11 numeric predictors, same SHA-256 as Runs 008/010/012/018.
- Seeds: 101–110, 70/30 split, race-stratified.
- Groups: race 0 unprivileged; race 1 privileged; target 1 favorable.
- Model: `LogisticRegression(solver='lbfgs', max_iter=1000)`.
- EO: `EqOddsPostprocessing(..., seed=seed)` calibrated and evaluated on the **same test set**, exactly as Run 018. This is **label leakage**, retained here *solely to isolate the implementation change*. Run a separate train/calibration/test experiment for generalization claims.
- Control branch: Run 018 behavior (`model.fit(X_train, y_train)`).
- Corrected branch: same inputs/parameters, but `model.fit(X_train, y_train, sample_weight=re_weights)`.

Both branches are executed within Run 037 on identical splits. The control must
reproduce saved Run 008 (pre-EO), Run 012 (post-EO), and Run 018 (post-EO)
**per seed**; otherwise execution stops without a misleading result.

The weight values are also checked against saved Run 010. Weighted pre-EO metrics
are *compared* to Run 010 but are not required to match, because Run 010 trains on
pandas features while Run 018/037 use the AIF360 feature array; versions and
numeric solver details can matter near the decision boundary.

## Files

- `scripts/run_037_corrected_re_eo_10seed.py` — main AIF360 experiment.
- `scripts/analyze_run_037.py` — generates Markdown report from actual result CSVs.
- `scripts/pre_eo_independent_sanity_check.py` — noncanonical weight/LR sanity check; **does not run EO**.
- `data/` — exact COMPAS CSV copied from the November 11 handoff.
- `provenance/` — frozen A-version per-seed tables for Runs 008, 010, 012, 018.
- `metadata/` — output files. No original runs are modified.

## Execute in your original Mac project `.venv`

From your reproduction project root:

```bash
source .venv/bin/activate
python -c "import aif360, sklearn, pandas, scipy; print('AIF360', getattr(aif360, '__version__', 'unreported'), 'sklearn', sklearn.__version__)"
```

Extract this package directly inside your reproduction repository (e.g.,
`~/Desktop/gradSchool/fairness_thesis/repos/farayola-reproduction/`). This creates
a self-contained `run_037_corrected_re_eo/` subdirectory. From the repository root:

```bash
python run_037_corrected_re_eo/scripts/run_037_corrected_re_eo_10seed.py --tag A
python run_037_corrected_re_eo/scripts/analyze_run_037.py --tag A
```

**Optional reproducibility rerun:**

```bash
python run_037_corrected_re_eo/scripts/run_037_corrected_re_eo_10seed.py --tag B
```

The main experiment outputs:

- `metadata/per_seed_results_A.csv` — ten rows, final/post EO and pre-EO metrics for both branches, seed-level diagnostics, paired differences.
- `metadata/aggregate_summary_A.csv` — mean/sd/min/max/95% t interval of corrected final results.
- `metadata/paired_deltas_A.csv` — paired seed-level metric differences, corrected minus original, plus descriptive intervals.
- `metadata/prediction_level_A.csv` — all 1,584 per-test predictions per seed for each branch/stage, plus original dataset row index and probability scores.
- `metadata/run_manifest_A.json` — exact environment versions and file digests.
- `metadata/analysis_report_A.md` — automatic comparison and interpretation cautions.

## If it fails

- `ModuleNotFoundError: aif360`: run in the thesis project's `.venv` with AIF360 installed, not an unrelated environment.
- `Run 008`, `012`, or `018` reference assertion: **do not remove the assertions**. Compare your environment versions to the original runs and inspect the diagnostic; the frozen experiment is not reproducing exactly.
- Run 010 pre-EO mismatch recorded in output: inspect feature order, pandas/NumPy solver behavior, and versions before claiming exact equality to the independent standalone run. It does not invalidate an otherwise controlled within-Run-037 weight intervention.

## Planned next research step

When the corrected/uncorrected comparison is confirmed, create an additional
**train/calibration/test** variant in a new run. Do not repurpose these test-set
calibrated outputs as held-out fairness/performance estimates.
