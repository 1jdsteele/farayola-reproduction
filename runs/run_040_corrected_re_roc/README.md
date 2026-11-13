# Run 040 — Reweighing + Reject Option Classification integration

## Research question

Does the apparently inactive Re+ROC implementation reflect (i) Reweighing
instance weights not reaching logistic regression, (ii) ROC being supplied hard
0/1 predictions instead of the continuous confidence scores it requires, or
(iii) both?

This is a **2 × 2 implementation-isolation experiment**. Every arm contains
ROC, and every arm uses the same COMPAS records, split, and model family.

| Arm | LR training weights | Scores supplied to ROC | Meaning |
| --- | --- | --- | --- |
| A_hard_unweighted | All 1 | Hard predicted labels | Historical ROC-alone control |
| B_hard_weighted | AIF360 Reweighing | Hard predicted labels | Weight propagation correction only |
| C_prob_unweighted | All 1 | `LR.predict_proba` | ROC score correction only |
| D_prob_weighted | AIF360 Reweighing | `LR.predict_proba` | Both corrections |

**Important:** "NONE" means no fairness mitigation at all. All four arms here
apply ROC; the classifier used by each arm is logistic regression. The hard-
label inputs in A/B replicate the original ROC integration. When you review
upstream metrics for A/C, those are the NONE classifier, before ROC.

## Method and correctness checks

- 5,278 observations from the canonical numeric COMPAS dataset; SHA-256 pinned.
- 11 features, target `two_year_recid`, positive label 1.
- Privileged race 1, unprivileged race 0.
- Train/test 70/30, race-stratified, seed 101–110, identical to Runs 008/010/037.
- `LogisticRegression(solver="lbfgs", max_iter=1000)` in every arm.
- Reweighing changes only training instance weights. Weighted arms pass `sample_weight` to `.fit()`.
- Frozen Run 008 NONE metrics/confusion matrices must match for every seed.
- Frozen Run 010 weight statistics must match for every seed; pre-ROC weighted predictions are additionally compared with Run 010 and any difference reported.
- When original Run 013 and Run 019 per-seed result CSVs exist in their conventional local paths, the original hard-score arm is checked against both.
- In A/B, `ROC.scores` is explicitly set to hard predicted labels; in C/D it is explicitly set to continuous positive-class probabilities. **No true labels are put in ROC scores.**
- Reproduces ROC configuration: `metric_name="Equal opportunity difference"`, `low_class_thresh=.3`, `high_class_thresh=.8`, `num_class_thresh=100`, `num_ROC_margin=50`, bounds ±.05.
- Outputs test-set changes (from LR's ordinary 0.5 predictions), threshold/margin, ROC constraint attainment and warnings, confusion matrices, SPD/DI/EOD/PED/accuracy, and row-level predictions.

## Methodological limitation — important

**ROC is calibrated and evaluated on the same test labels in Run 040**, consistent
with the original released experiments. This isolates implementation changes but
is NOT a held-out generalization estimate. A subsequent held-out 70/15/15
investigation, fitting ROC only on calibration rows, will be needed to assess
out-of-sample performance and any changes in fairness. This is analogous to
Run 037 followed by Run 038.

The AIF360 ROC implementation searches 100 classification thresholds × 50
ROC margins for each arm/seed. It optimizes **balanced accuracy**, subject to
an **EOD-only** constraint if any feasible option exists; otherwise it has an
implementation-specific fallback and issues a warning. Its selected threshold
need not be 0.5. ROC requires continuous scores to interpret uncertainty.

## Folder layout

```
run_040_corrected_re_roc/
├── README.md
├── data/compas_farayola_numeric.csv
├── provenance/
│   ├── run_008_none_per_seed_results_A.csv
│   └── run_010_reweighing_per_seed_results_A.csv
├── scripts/
│   ├── check_run_040_design.py
│   ├── run_040_corrected_re_roc_10seed.py
│   └── analyze_run_040.py
└── metadata/  # generated results go here
```

## Commands (from the thesis repository root)

```bash
cd ~/Desktop/gradSchool/fairness_thesis/repos/farayola-reproduction
source .venv/bin/activate

# If ZIP was downloaded to ~/Downloads:
unzip ~/Downloads/run_040_corrected_re_roc.zip -d runs/

# Preflight (does not require AIF360 to execute):
python runs/run_040_corrected_re_roc/scripts/check_run_040_design.py

# Optional one-seed four-arm smoke test (do not use it as scientific results):
python runs/run_040_corrected_re_roc/scripts/run_040_corrected_re_roc_10seed.py --seeds 101 --tag SMOKE

# Full ten-seed experiment:
python runs/run_040_corrected_re_roc/scripts/run_040_corrected_re_roc_10seed.py --tag A

# Analysis:
python runs/run_040_corrected_re_roc/scripts/analyze_run_040.py --tag A
```

If an assertion fails, stop and send the output. Don't bypass checks. The
optional original ROC references may have different filenames in some repos;
the script prints a note if it cannot find them, but frozen Run 008 is required.

**Do not freeze Run 040 until the ten-seed analysis has been reviewed.**

## Source of ROC behavior

- AIF360 0.6.1 ROC source:
  https://github.com/Trusted-AI/AIF360/blob/v0.6.1/aif360/algorithms/postprocessing/reject_option_classification.py
- AIF360 0.6.1 BinaryLabelDataset scores default:
  https://github.com/Trusted-AI/AIF360/blob/v0.6.1/aif360/datasets/structured_dataset.py
- Farayola authors' `Git_COMPAS_PP_Re+EO_ROC.ipynb` in the handoff:
  calls `ROC.fit_predict()` after constructing a predicted-label BLD, without
  explicit probabilities, and does not pass Reweighing weights to LR.
