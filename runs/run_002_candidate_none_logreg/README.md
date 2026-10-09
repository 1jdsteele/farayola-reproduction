# Run 002 — Candidate NONE Baseline Using Logistic Regression

## Purpose

Test whether the reconstructed COMPAS model input can approximately
reproduce the paper's untreated COMPAS `NONE` condition using Logistic
Regression as a candidate baseline classifier.

## Important limitation

The paper does not clearly specify which classifier produced the `NONE`
baseline.

Logistic Regression is therefore an inferred candidate, not a confirmed
reconstruction of the authors' baseline classifier.

It was selected because Logistic Regression is explicitly identified as
the base classifier used by EGR in the paper and IF-EA implementation.

## Input

Uses the exact same `compas_farayola_numeric.csv` preserved in Run 001.

The input contains:

- 5,278 observations
- 11 reconstructed predictors
- `two_year_recid` target
- 3,175 African-American observations
- 2,103 Caucasian observations
- 2,795 target = 0
- 2,483 target = 1

The 5,278-person cohort is exactly reconstructed from the ProPublica
COMPAS data. The exact 11-feature model matrix remains inferred because
the authors did not publish the processed input CSV or complete feature
list.

## Split

- test_size = 0.30
- random_state = 101
- shuffle = True
- stratify = two_year_recid

These parameters match the visible IF-EA COMPAS notebook.

## Classifier

`LogisticRegression()` with scikit-learn defaults.

No feature scaling or convergence correction was introduced.

## Fairness interventions

NONE.

Specifically, this run does NOT apply:

- Reweighing
- Disparate Impact Remover
- Adversarial Learning
- Exponentiated Gradient Reduction
- Equalized Odds post-processing
- Reject Option Classification

## Protected group definitions

- race = 0: African-American / unprivileged
- race = 1: Caucasian / privileged

The IF-EA convention `favorable_label = 1` is retained even though
`two_year_recid = 1` denotes recidivism.

## Evaluated metrics

- Accuracy
- Statistical Parity Difference
- Disparate Impact
- Equal Opportunity Difference
- Predictive Equality Difference

## Interpretation rule

This run should be treated as a candidate reconstruction of the paper's
NONE condition, not as proof of the authors' exact baseline procedure.

Do not modify files in this directory after the run is frozen.
