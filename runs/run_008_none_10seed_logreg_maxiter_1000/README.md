# Run 008 — Candidate NONE 10-Seed Logistic Regression Convergence Diagnostic

## Parent

runs/run_007_none_race_stratified_10seed/

## Research question

Does allowing the candidate NONE Logistic Regression models additional
LBFGS optimization iterations materially change the ten-seed
race-stratified baseline, particularly seed 105 or the aggregate
fairness metrics?

## Motivation

Run 007 retains LogisticRegression() with its default max_iter=100.

Across seeds 101 through 110, seed 105 reached the 100-iteration limit
and emitted a scikit-learn LBFGS ConvergenceWarning.

The other nine seeds terminated below 100 iterations.

Run 008 tests whether that convergence limitation materially affects the
candidate NONE sensitivity experiment.

## Single scientific change

Run 007:

classifier = LogisticRegression()

Run 008:

classifier = LogisticRegression(max_iter=1000)

## Held constant

- reconstructed COMPAS input
- best-supported 11-feature matrix
- seeds 101 through 110
- 70:30 split
- shuffle=True
- race stratification
- no fairness intervention
- AIF360 protected-group definitions
- favorable_label=1 convention
- accuracy and fairness metrics
- aggregate summary calculations
- software environment

## Important reporting note

The inherited script's human-readable protocol output may still say:

Classifier: LogisticRegression()

The executed constructor is actually:

LogisticRegression(max_iter=1000)

The exact source difference is preserved in:

metadata/run007_vs_run008_code_diff.txt

## Repeatability

Run 008 is executed twice.

Per-seed results, aggregate results, and stdout are compared between
executions before freezing.

## Comparison

A machine-readable Run 007 versus Run 008 per-seed comparison is
preserved in:

metadata/run007_vs_run008_metric_comparison.csv

## Interpretation policy

Run 008 is checksummed and frozen before interpreting whether increased
Logistic Regression optimization materially changes the results.
